"""
Load text/CSV and split long documents into batches for the paragraph pipeline.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

MIN_BATCH_CHARS = 500
DEFAULT_BATCH_CHARS = 2000


def load_txt_file(path: str | Path) -> tuple[str | None, str | None]:
    """Read UTF-8 text file. Returns (text, error_message)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
        return text, None
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="utf-8")
            return text, None
        except Exception as e:
            return None, f"Failed to read text file: {e}"
    except Exception as e:
        return None, f"Failed to read text file: {e}"


def load_csv_column(path: str | Path, column_name: str) -> tuple[str | None, str | None]:
    """Load CSV with pandas, take one column, join non-empty cells with blank lines."""
    path = Path(path)
    name = column_name.strip()
    if not name:
        return None, "CSV column name is empty."

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception as e:
        return None, f"Failed to read CSV: {e}"

    if name not in df.columns:
        return None, f"Column {name!r} not found. Available columns: {df.columns.tolist()}"

    col = df[name].dropna()
    vals: list[str] = []
    for v in col:
        t = str(v).strip()
        if t and t.lower() != "nan":
            vals.append(t)

    if not vals:
        return None, f"No non-empty values in column {name!r}."

    return "\n\n".join(vals), None


def _hard_chunks(s: str, max_chars: int) -> list[str]:
    return [s[i : i + max_chars] for i in range(0, len(s), max_chars)]


def _split_oversized_paragraph(p: str, max_chars: int) -> list[str]:
    if len(p) <= max_chars:
        return [p]
    raw = re.split(r"(?<=[.!?…])\s+", p)
    sentences = [x.strip() for x in raw if x.strip()]
    if not sentences:
        return _hard_chunks(p, max_chars)

    out: list[str] = []
    buf: list[str] = []
    bl = 0

    for s in sentences:
        if len(s) > max_chars:
            if buf:
                out.append(" ".join(buf))
                buf = []
                bl = 0
            out.extend(_hard_chunks(s, max_chars))
            continue
        gap = 1 if buf else 0
        if buf and bl + gap + len(s) > max_chars:
            out.append(" ".join(buf))
            buf = [s]
            bl = len(s)
        else:
            buf.append(s)
            bl += gap + len(s)
    if buf:
        out.append(" ".join(buf))
    return out


def split_into_batches(text: str, max_chars: int) -> list[str]:
    """
    Split text into chunks up to max_chars, preferring paragraph boundaries,
    then sentence boundaries, then hard slices.
    """
    if max_chars < MIN_BATCH_CHARS:
        max_chars = MIN_BATCH_CHARS

    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    batches: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        if buf:
            batches.append("\n\n".join(buf))
            buf = []

    for p in paragraphs:
        if len(p) > max_chars:
            flush()
            batches.extend(_split_oversized_paragraph(p, max_chars))
            continue

        trial = "\n\n".join(buf + [p])
        if buf and len(trial) > max_chars:
            flush()

        buf.append(p)

    flush()
    return batches


def compute_batch_spans(text: str, batches: list[str]) -> list[tuple[int, int]] | None:
    """
    Best-effort mapping of each batch back to a (start, end) span in the original text.

    This is used as a non-overlap guard for parallel processing. If mapping fails
    (e.g. batches were normalized/rewrapped such that exact substring match is impossible),
    returns None.
    """
    if not text or not batches:
        return []

    spans: list[tuple[int, int]] = []
    cursor = 0
    for b in batches:
        if not b:
            spans.append((cursor, cursor))
            continue
        start = text.find(b, cursor)
        if start < 0:
            return None
        end = start + len(b)
        spans.append((start, end))
        cursor = end
    return spans
