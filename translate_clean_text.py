"""
One-off: add English translations for the clean_text column in generated_pairs CSV.

Uses OpenAI gpt-4o by default (strong for Ukrainian → English). Writes a new file;
does not modify the input CSV.

Usage:
  set OPENAI_API_KEY=...
  python translate_clean_text.py corpus_output/generated_pairs.csv
  python translate_clean_text.py corpus_output/generated_pairs.csv -o corpus_output/generated_pairs_en.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# Best general OpenAI model for high-quality Ukrainian→English translation.
DEFAULT_TRANSLATION_MODEL = os.environ.get("TRANSLATION_MODEL", "gpt-5.4-nano")


def _insert_field_after(fieldnames: list[str], after: str, new: str) -> list[str]:
    if new in fieldnames:
        return list(fieldnames)
    try:
        i = fieldnames.index(after)
    except ValueError:
        return fieldnames + [new]
    return fieldnames[: i + 1] + [new] + fieldnames[i + 1 :]


def translate_one(client: OpenAI, model: str, ukrainian: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert translator. Translate Ukrainian literary text "
                    "into natural, fluent English. Preserve paragraph breaks and "
                    "sentence boundaries. Output only the English translation, with "
                    "no preface or notes."
                ),
            },
            {"role": "user", "content": ukrainian},
        ],
    )
    out = resp.choices[0].message.content
    return (out or "").strip()


def main() -> None:
    p = argparse.ArgumentParser(description="Translate clean_text to English (one-off CSV)")
    p.add_argument(
        "input_csv",
        type=Path,
        help="Path to generated_pairs.csv (or similar)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <input_stem>_with_en.csv next to input)",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_TRANSLATION_MODEL,
        help=f"OpenAI chat model (default: {DEFAULT_TRANSLATION_MODEL})",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.15,
        help="Seconds between API calls (rate limits; default 0.15)",
    )
    args = p.parse_args()

    inp = args.input_csv.resolve()
    if not inp.is_file():
        print(f"File not found: {inp}", file=sys.stderr)
        sys.exit(1)

    out = args.output
    if out is None:
        out = inp.with_name(f"{inp.stem}_with_en{inp.suffix}")
    else:
        out = out.resolve()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    with open(inp, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("CSV has no header row.", file=sys.stderr)
            sys.exit(1)
        if "clean_text" not in reader.fieldnames:
            print("CSV must contain a 'clean_text' column.", file=sys.stderr)
            sys.exit(1)

        fieldnames = _insert_field_after(list(reader.fieldnames), "clean_text", "clean_text_en")
        rows = list(reader)

    total = len(rows)
    print(f"[translate] {total} rows, model={args.model!r}, -> {out}", flush=True)

    for i, row in enumerate(rows, start=1):
        src = row.get("clean_text") or ""
        row["clean_text_en"] = translate_one(client, args.model, src) if src else ""
        if args.sleep > 0 and i < total:
            time.sleep(args.sleep)
        if i % 25 == 0 or i == total:
            print(f"  ... {i}/{total}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"[translate] Done: {out}", flush=True)


if __name__ == "__main__":
    main()
