"""
One-off: translate `polluted_text` to English in several ways:

  --mode direct     Plain translation (same idea as translate_clean_text.py).
  --mode rag        RAG on polluted only (spaCy/VESUM, Meilisearch, web search;
                    search_unknown_words uses persist=False).
  --mode rag-clean  Same RAG blocks as rag, plus the paired `clean_text` as a
                    standard-register Ukrainian hint (to align meaning with gold
                    clean_text_en when scoring).

Requires: OPENAI_API_KEY, Meilisearch (for custom_dict lookup), spaCy uk model,
and dic_data/ — same as paragraph_pipeline.

Run while translate_clean_text.py runs in another terminal (different process).

Examples:
  set OPENAI_API_KEY=...
  python translate_polluted_text.py corpus_output/generated_pairs.csv --mode direct
  python translate_polluted_text.py corpus_output/generated_pairs_with_en.csv --mode rag \\
      -o corpus_output/generated_pairs_with_en_polluted.csv

  # RAG + clean paragraph as register hint (for comparing to golden clean_text_en):
  python translate_polluted_text.py corpus_output/generated_pairs_with_en.csv --mode rag-clean
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI

from paragraph_pipeline import extract_candidates, json_dumps, search_unknown_words

DEFAULT_TRANSLATION_MODEL = os.environ.get("TRANSLATION_MODEL", "gpt-4o")


def _insert_field_after(fieldnames: list[str], after: str, new: str) -> list[str]:
    if new in fieldnames:
        return list(fieldnames)
    try:
        i = fieldnames.index(after)
    except ValueError:
        return fieldnames + [new]
    return fieldnames[: i + 1] + [new] + fieldnames[i + 1 :]


async def _translate_async(
    client: AsyncOpenAI,
    model: str,
    system: str,
    user: str,
) -> str:
    resp = await client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    out = resp.choices[0].message.content
    return (out or "").strip()


_DIRECT_SYSTEM = (
    "You are an expert translator. Translate Ukrainian text into natural, fluent "
    "English. Preserve paragraph breaks and sentence boundaries. Output only the "
    "English translation, with no preface or notes."
)


_RAG_SYSTEM = (
    "You are an expert translator. You will receive a Ukrainian paragraph plus "
    "lexical reference material (dictionary entries, VESUM suggestions, and web "
    "search summaries). Use that material only to disambiguate non-standard or "
    "informal words; translate the paragraph faithfully into natural English. "
    "Output only the English translation, with no preface or notes."
)

_RAG_CLEAN_SYSTEM = (
    "You are an expert translator. You will receive: (1) a Ukrainian paragraph in a "
    "non-standard / mixed register, (2) the same content in standard literary "
    "Ukrainian for meaning alignment, (3) lexical RAG material. Use the standard "
    "Ukrainian line only to fix sense and register when translating the first "
    "paragraph into English — do not ignore the informal surface of the first "
    "paragraph unless the task clearly calls for neutralization. Output only the "
    "English translation of the first (polluted) paragraph, with no preface or notes."
)


async def translate_polluted_direct(client: AsyncOpenAI, model: str, text: str) -> str:
    if not (text or "").strip():
        return ""
    return await _translate_async(client, model, _DIRECT_SYSTEM, text.strip())


async def translate_polluted_rag(client: AsyncOpenAI, model: str, polluted: str) -> str:
    text = (polluted or "").strip()
    if not text:
        return ""

    candidates = await extract_candidates(text)
    search_results = await search_unknown_words(
        candidates["unknown"],
        candidates["known_bad_no_repl"],
        text,
        client,
        persist=False,
    )

    needs_context = (
        candidates["unknown"]
        or candidates["known_bad"]
        or candidates["known_bad_no_repl"]
        or candidates["in_custom_dict"]
    )

    if not needs_context:
        return await translate_polluted_direct(client, model, text)

    user = f"""Paragraph to translate to English:
{text}

Words in our custom dictionary with known meanings and replacements:
{json_dumps(candidates["in_custom_dict"])}

Non-standard words with VESUM suggested replacements (list — pick best for context):
{json_dumps(candidates["known_bad"])}

Non-standard words with no replacements + unknown words:
{json_dumps(candidates["known_bad_no_repl"] + candidates["unknown"])}

Search results from web (already context-appropriate):
{json_dumps(search_results)}
"""
    return await _translate_async(client, model, _RAG_SYSTEM, user)


async def translate_polluted_rag_clean(
    client: AsyncOpenAI, model: str, polluted: str, clean: str
) -> str:
    """RAG on polluted text plus the paired clean_text as a standard-register hint."""
    text = (polluted or "").strip()
    clean_s = (clean or "").strip()
    if not text:
        return ""
    if not clean_s:
        return await translate_polluted_rag(client, model, text)

    candidates = await extract_candidates(text)
    search_results = await search_unknown_words(
        candidates["unknown"],
        candidates["known_bad_no_repl"],
        text,
        client,
        persist=False,
    )

    needs_context = (
        candidates["unknown"]
        or candidates["known_bad"]
        or candidates["known_bad_no_repl"]
        or candidates["in_custom_dict"]
    )

    if not needs_context:
        user = f"""Paragraph to translate (non-standard register):
{text}

Standard literary Ukrainian (same passage; use for meaning/register only):
{clean_s}
"""
        return await _translate_async(client, model, _RAG_CLEAN_SYSTEM, user)

    user = f"""Paragraph to translate to English (non-standard register):
{text}

Standard literary Ukrainian (same passage; use for meaning/register only):
{clean_s}

Words in our custom dictionary with known meanings and replacements:
{json_dumps(candidates["in_custom_dict"])}

Non-standard words with VESUM suggested replacements (list — pick best for context):
{json_dumps(candidates["known_bad"])}

Non-standard words with no replacements + unknown words:
{json_dumps(candidates["known_bad_no_repl"] + candidates["unknown"])}

Search results from web (already context-appropriate):
{json_dumps(search_results)}
"""
    return await _translate_async(client, model, _RAG_CLEAN_SYSTEM, user)


async def _run_rows(
    mode: str,
    rows: list[dict],
    model: str,
    api_key: str,
    sleep_s: float,
    out_col: str,
) -> None:
    client = AsyncOpenAI(api_key=api_key)
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        src = row.get("polluted_text") or ""
        if mode == "direct":
            row[out_col] = await translate_polluted_direct(client, model, src)
        elif mode == "rag":
            row[out_col] = await translate_polluted_rag(client, model, src)
        else:
            row[out_col] = await translate_polluted_rag_clean(
                client, model, src, row.get("clean_text") or ""
            )

        if sleep_s > 0 and i < total:
            await asyncio.sleep(sleep_s)
        if i % 10 == 0 or i == total:
            print(f"  ... {i}/{total}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Translate polluted_text: direct vs RAG-augmented (same model)"
    )
    p.add_argument("input_csv", type=Path, help="CSV with polluted_text column")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <stem>_polluted_<mode>.csv)",
    )
    p.add_argument(
        "--mode",
        choices=("direct", "rag", "rag-clean"),
        required=True,
        help=(
            "direct = raw; rag = RAG on polluted only; "
            "rag-clean = RAG + clean_text column as standard-register hint (needs clean_text)"
        ),
    )
    p.add_argument("--model", default=DEFAULT_TRANSLATION_MODEL, help="Chat model for English")
    p.add_argument("--sleep", type=float, default=0.15, help="Pause between rows (seconds)")
    p.add_argument(
        "--output-column",
        default=None,
        help="CSV column for English (defaults: polluted_text_en / polluted_text_en_rag / polluted_text_en_rag_clean)",
    )
    args = p.parse_args()

    inp = args.input_csv.resolve()
    if not inp.is_file():
        print(f"File not found: {inp}", file=sys.stderr)
        sys.exit(1)

    out = args.output
    if out is None:
        out = inp.with_name(f"{inp.stem}_polluted_{args.mode}{inp.suffix}")
    else:
        out = out.resolve()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    default_cols = {
        "direct": "polluted_text_en",
        "rag": "polluted_text_en_rag",
        "rag-clean": "polluted_text_en_rag_clean",
    }
    col = args.output_column or default_cols[args.mode]

    with open(inp, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "polluted_text" not in reader.fieldnames:
            print("CSV must contain polluted_text.", file=sys.stderr)
            sys.exit(1)
        if args.mode == "rag-clean" and "clean_text" not in reader.fieldnames:
            print("rag-clean mode requires a clean_text column in the CSV.", file=sys.stderr)
            sys.exit(1)
        fieldnames = _insert_field_after(list(reader.fieldnames), "polluted_text", col)
        rows = list(reader)

    print(
        f"[translate_polluted] mode={args.mode!r} model={args.model!r} "
        f"rows={len(rows)} -> {out}",
        flush=True,
    )

    asyncio.run(_run_rows(args.mode, rows, args.model, api_key, args.sleep, col))

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"[translate_polluted] Done: {out}", flush=True)


if __name__ == "__main__":
    main()
