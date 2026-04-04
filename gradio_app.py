"""
Gradio UI for the Ukrainian paragraph pipeline (OpenAI + Meilisearch + local dictionaries).
Run: python gradio_app.py
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any

import gradio as gr
from openai import AsyncOpenAI, RateLimitError

from batching import DEFAULT_BATCH_CHARS, compute_batch_spans, load_csv_column, load_txt_file, split_into_batches
from paragraph_pipeline import analyze_paragraph


def _normalize_upload_path(upload_file: Any) -> str | None:
    if upload_file is None:
        return None
    if isinstance(upload_file, (list, tuple)) and upload_file:
        return str(upload_file[0])
    return str(upload_file)


def _upload_dependent_visibility(upload_file: Any):
    """
    Paragraph: visible only when no file is uploaded.
    CSV column: visible only when the uploaded file is .csv.
    """
    path_str = _normalize_upload_path(upload_file)
    if not path_str:
        return gr.update(visible=True), gr.update(visible=False)
    suf = Path(path_str).suffix.lower()
    if suf == ".csv":
        return gr.update(visible=False), gr.update(visible=True)
    return gr.update(visible=False), gr.update(visible=False)


def _fixed_output_visibility(fix_paragraph: bool):
    """Hide fixed-text output and download when 'Fix paragraph' is off."""
    vis = bool(fix_paragraph)
    return gr.update(visible=vis), gr.update(visible=vis)


def resolve_input_text(
    paragraph: str,
    upload_file: Any,
    csv_column: str,
) -> tuple[str | None, str | None]:
    """Returns (text, error_message)."""
    path_str = _normalize_upload_path(upload_file)
    if path_str:
        path = Path(path_str)
        if not path.is_file():
            return None, f"Upload not found: {path_str}"
        suf = path.suffix.lower()
        if suf == ".csv":
            col = (csv_column or "").strip()
            if not col:
                return None, "CSV column name is required when uploading a .csv file."
            return load_csv_column(path, col)
        return load_txt_file(path)

    text = (paragraph or "").strip()
    if not text:
        return None, "Paragraph is empty (or upload a .txt / .csv file)."
    return text, None


def _write_fixed_temp(content: str) -> str:
    fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="fixed_paragraph_", text=False)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(content)
        return tmp_path
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


_STOPPED_LOG = (
    "\n\n[Generation stopped by user. "
    "The current batch may still finish on the server before stopping. "
    "Partial fixed text and download (if any) are kept.]"
)


def _buttons_for_running(is_running: bool):

    return (
        gr.update(visible=not is_running),  # Run
        gr.update(visible=is_running),      # Stop
    )


def _prepare_run_outputs():
    # Clear previous run outputs so old results do not linger.
    run_btn, stop_btn = _buttons_for_running(True)
    return gr.update(value=""), gr.update(value=""), gr.update(value=None), run_btn, stop_btn


async def run_pipeline_stream(
    paragraph: str,
    upload_file: Any,
    csv_column: str,
    max_chars_per_batch: float | int | None,
    fix_paragraph: bool,
    account_for_untypical_usage: bool,
    api_key: str,
):
    if not api_key or not str(api_key).strip():
        run_btn, stop_btn = _buttons_for_running(False)
        yield (
            "",
            "Error: OpenAI API key is required. Paste your key in the field above.",
            gr.update(value=None),
            run_btn,
            stop_btn,
        )
        return

    text, err = resolve_input_text(paragraph, upload_file, csv_column)
    if err:
        run_btn, stop_btn = _buttons_for_running(False)
        yield ("", err, gr.update(value=None), run_btn, stop_btn)
        return
    assert text is not None

    max_chars = int(max_chars_per_batch) if max_chars_per_batch else DEFAULT_BATCH_CHARS
    batches = split_into_batches(text, max_chars)
    if not batches:
        run_btn, stop_btn = _buttons_for_running(False)
        yield (
            "",
            "Error: no text to process after splitting.",
            gr.update(value=None),
            run_btn,
            stop_btn,
        )
        return

    spans = compute_batch_spans(text, batches)
    if spans is not None:
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            if not (e1 <= s2 and s1 <= e1 and s2 <= e2):
                raise RuntimeError("Batch spans overlap; refusing to run parallel analysis.")

    client = AsyncOpenAI(api_key=str(api_key).strip())
    fixed_chunks: list[str] = []
    log_chunks: list[str] = []
    n = len(batches)

    max_parallel_batches = 5
    sem = asyncio.Semaphore(max_parallel_batches)

    pending_fixed: dict[int, str] = {}
    next_to_emit = 0

    async def _run_one(batch_idx: int, batch_text: str):
        async with sem:
            out_buf = io.StringIO()
            err_buf = io.StringIO()
            try:
                res = None
                while True:
                    try:
                        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                            res = await analyze_paragraph(
                                batch_text,
                                client,
                                account_for_untypical_usage=account_for_untypical_usage,
                                fix_paragraph=fix_paragraph,
                            )
                        break
                    except RateLimitError as e:
                        note = (
                            f"[rate_limit] batch {batch_idx + 1}/{n}: {e}\n"
                            f"Waiting 30s before retrying the same batch from the start...\n"
                        )
                        err_buf.write(note)
                        print(note, end="", flush=True)
                        await asyncio.sleep(30)
                fixed = res.fixed_paragraph if fix_paragraph else ""
                status = "ok"
                return batch_idx, fixed, out_buf.getvalue() + err_buf.getvalue(), status
            except asyncio.CancelledError:
                raise
            except Exception as e:
                err_buf.write(f"{type(e).__name__}: {e}\n")
                err_buf.write(traceback.format_exc())
                return batch_idx, "", out_buf.getvalue() + err_buf.getvalue(), "failed"

    tasks = [asyncio.create_task(_run_one(i, b)) for i, b in enumerate(batches)]

    try:
        for coro in asyncio.as_completed(tasks):
            batch_idx, fixed_text, chunk_logs, status = await coro
            human_idx = batch_idx + 1

            if status == "failed":
                log_chunks.append(f"=== Batch {human_idx}/{n} (failed) ===\n{chunk_logs}")
                run_btn, stop_btn = _buttons_for_running(False)
                yield (
                    "\n\n".join(fixed_chunks),
                    "\n\n".join(log_chunks),
                    gr.update(value=None),
                    run_btn,
                    stop_btn,
                )
                return

            log_chunks.append(f"=== Batch {human_idx}/{n} ===\n{chunk_logs}")

            if fix_paragraph:
                pending_fixed[batch_idx] = fixed_text

                advanced = False
                while next_to_emit in pending_fixed:
                    fixed_chunks.append(pending_fixed.pop(next_to_emit))
                    next_to_emit += 1
                    advanced = True

                if advanced:
                    yield (
                        "\n\n".join(fixed_chunks),
                        "\n\n".join(log_chunks),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                    )
            else:
                yield (
                    "",
                    "\n\n".join(log_chunks),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                )

    except asyncio.CancelledError:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        cumulative_fixed = "\n\n".join(fixed_chunks)
        cumulative_logs = "\n\n".join(log_chunks) + _STOPPED_LOG
        run_btn, stop_btn = _buttons_for_running(False)
        if fix_paragraph and cumulative_fixed.strip():
            dl_path = _write_fixed_temp(cumulative_fixed)
            yield (cumulative_fixed, cumulative_logs, dl_path, run_btn, stop_btn)
        else:
            yield (
                cumulative_fixed,
                cumulative_logs,
                gr.update(value=None),
                run_btn,
                stop_btn,
            )
        return
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    cumulative_fixed = "\n\n".join(fixed_chunks)
    cumulative_logs = "\n\n".join(log_chunks)
    run_btn, stop_btn = _buttons_for_running(False)
    if fix_paragraph and cumulative_fixed.strip():
        tmp_path = _write_fixed_temp(cumulative_fixed)
        yield (cumulative_fixed, cumulative_logs, tmp_path, run_btn, stop_btn)
    else:
        yield (
            cumulative_fixed,
            cumulative_logs,
            gr.update(value=None),
            run_btn,
            stop_btn,
        )


def build_ui():
    with gr.Blocks(title="Ukrainian paragraph pipeline") as demo:
        gr.Markdown(
            "Enter text or upload a **.txt** / **.csv** file. For CSV, set the column name to read "
            "(all non-empty cells are joined with blank lines). Long input is split into batches; "
            "results stream as each batch finishes. Use **Stop** to cancel; the **current batch** may "
            "still complete before the run ends. Requires Meilisearch (`MEILI_HOST` / `MEILI_API_KEY`) "
            "and spaCy `uk_core_news_sm`."
        )
        with gr.Row():
            upload = gr.File(
                label="Optional: upload .txt or .csv",
                file_types=[".txt", ".csv"],
                type="filepath",
            )
            csv_col = gr.Textbox(
                label="CSV column name (required for .csv)",
                placeholder="e.g. text",
                lines=1,
                visible=False,
            )
        max_batch = gr.Number(
            label="Max characters per batch",
            value=DEFAULT_BATCH_CHARS,
            minimum=500,
            step=100,
            precision=0,
        )
        paragraph_in = gr.Textbox(
            label="Paragraph (used when no file is uploaded)",
            lines=12,
            placeholder="Вставте текст українською…",
        )
        fix_cb = gr.Checkbox(label="Fix paragraph", value=True)
        untypical_cb = gr.Checkbox(
            label="Account for untypical usage",
            value=False,
        )
        api_key_in = gr.Textbox(
            label="OpenAI API key (required)",
            type="password",
            placeholder="sk-…",
        )
        with gr.Row():
            submit_btn = gr.Button("Run", variant="primary")
            stop_btn = gr.Button("Stop", visible=False)

        fixed_out = gr.Textbox(
            label="Fixed paragraph (streams as batches complete)",
            lines=12,
            interactive=False,
        )
        logs_out = gr.Textbox(
            label="Logs — includes batches and llm_search_logs.jsonl lines",
            lines=18,
            interactive=False,
        )
        download_file = gr.File(
            label="Download fixed text (.txt)",
            interactive=False,
        )

        upload.change(
            fn=_upload_dependent_visibility,
            inputs=[upload],
            outputs=[paragraph_in, csv_col],
            show_progress="hidden",
        )
        fix_cb.change(
            fn=_fixed_output_visibility,
            inputs=[fix_cb],
            outputs=[fixed_out, download_file],
            show_progress="hidden",
        )

        prep_event = submit_btn.click(
            fn=_prepare_run_outputs,
            inputs=[],
            outputs=[fixed_out, logs_out, download_file, submit_btn, stop_btn],
            show_progress="hidden",
        )
        run_event = prep_event.then(
            fn=run_pipeline_stream,
            inputs=[
                paragraph_in,
                upload,
                csv_col,
                max_batch,
                fix_cb,
                untypical_cb,
                api_key_in,
            ],
            outputs=[fixed_out, logs_out, download_file, submit_btn, stop_btn],
            show_progress="minimal",
            show_progress_on=[fixed_out, logs_out, download_file],
        )
        stop_btn.click(fn=None, cancels=[run_event])

    demo.queue(default_concurrency_limit=1)
    return demo


if __name__ == "__main__":
    build_ui().launch()
