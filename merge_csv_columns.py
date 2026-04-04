"""
Join extra translation columns into a base CSV by matching `id`.

Example (golden EN in generated_pairs_with_en; polluted translations in separate files):

  python merge_csv_columns.py corpus_output/generated_pairs_with_en.csv \\
    -o corpus_output/merged_for_scoring.csv \\
    --from corpus_output/polluted_direct.csv polluted_text_en \\
    --from corpus_output/polluted_rag.csv polluted_text_en_rag

Then:

  python score_translations.py corpus_output/merged_for_scoring.csv \\
    --source-column polluted_text --reference-column clean_text_en \\
    --hypothesis polluted_text_en polluted_text_en_rag ...
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Merge CSV columns by id into base CSV")
    p.add_argument(
        "base_csv",
        type=Path,
        help="Base file (must have id + columns you keep, e.g. clean_text_en gold)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output CSV path",
    )
    p.add_argument(
        "--from",
        dest="joins",
        action="append",
        nargs=2,
        metavar=("FILE", "COLUMN"),
        required=True,
        help="Extra CSV and the column name to copy (repeat for each file)",
    )
    args = p.parse_args()

    base_path = args.base_csv.resolve()
    if not base_path.is_file():
        print(f"Not found: {base_path}", file=sys.stderr)
        sys.exit(1)

    with open(base_path, newline="", encoding="utf-8") as f:
        base_rows = list(csv.DictReader(f))
    if not base_rows:
        print("Base CSV is empty.", file=sys.stderr)
        sys.exit(1)
    if "id" not in base_rows[0]:
        print("Base CSV must have an 'id' column.", file=sys.stderr)
        sys.exit(1)

    by_id = {str(r["id"]): r for r in base_rows}

    for path_str, col in args.joins:
        path = Path(path_str).resolve()
        if not path.is_file():
            print(f"Not found: {path}", file=sys.stderr)
            sys.exit(1)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if "id" not in (rows[0] if rows else {}):
            print(f"{path} must have an 'id' column.", file=sys.stderr)
            sys.exit(1)
        if col not in rows[0]:
            print(f"{path} has no column {col!r}.", file=sys.stderr)
            sys.exit(1)
        n = 0
        for r in rows:
            i = str(r["id"])
            if i not in by_id:
                print(f"Warning: id {i!r} in {path.name} not in base — skipped", file=sys.stderr)
                continue
            by_id[i][col] = r.get(col, "")
            n += 1
        print(f"[merge] {path.name}: copied {col!r} for {n} rows", flush=True)

    out_rows = [by_id[str(r["id"])] for r in base_rows]
    fieldnames = list(base_rows[0].keys())
    for _path, col in args.joins:
        if col not in fieldnames:
            fieldnames.append(col)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"[merge] Wrote {args.output.resolve()} ({len(out_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
