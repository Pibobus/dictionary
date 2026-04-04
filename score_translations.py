"""
Score English hypotheses against a gold reference using COMET (reference-based)
and BLEURT.

COMET: Unbabel/wmt22-comet-da (src + machine translation + reference).
BLEURT: Hugging Face `evaluate` + google-research BLEURT checkpoint (default
bleurt-base-128; use --bleurt-config bleurt-large-512 for longer texts).

Typical setup: gold English from gpt-5.4-nano in one column (e.g. clean_text_en_ref),
hypotheses from other runs in other columns (e.g. clean_text_en_alt, polluted_text_en).

Dependencies (heavy; install once — on Python 3.13 you may need `pip install
unbabel-comet --no-deps` then compatible torch/transformers/torchmetrics):

  pip install evaluate datasets
  pip install unbabel-comet --no-deps
  pip install torch pytorch-lightning "transformers>=4.17,<5" "torchmetrics>=0.10.2,<0.11" \\
      "protobuf>=4.24.4,<5" "huggingface-hub>=0.19.3,<1" sacrebleu entmax jsonargparse
  pip install "git+https://github.com/google-research/bleurt.git"

Example:

  python score_translations.py corpus_output/generated_pairs_with_en.csv \\
    --source-column clean_text \\
    --reference-column clean_text_en \\
    --hypothesis polluted_text_en polluted_text_en_rag \\
    -o corpus_output/translation_scores.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path


def _slug(name: str) -> str:
    s = re.sub(r"[^\w]+", "_", name.strip()).strip("_")
    return s or "col"


def _load_comet():
    from comet import download_model, load_from_checkpoint

    path = download_model("Unbabel/wmt22-comet-da")
    return load_from_checkpoint(path)


def _load_bleurt(config_name: str | None):
    import evaluate

    if config_name:
        return evaluate.load("bleurt", config_name=config_name)
    return evaluate.load("bleurt")


def main() -> None:
    p = argparse.ArgumentParser(description="COMET + BLEURT vs gold reference column")
    p.add_argument("input_csv", type=Path, help="CSV with source, reference, hypotheses")
    p.add_argument(
        "--source-column",
        default="clean_text",
        help="Ukrainian source (for COMET; default: clean_text)",
    )
    p.add_argument(
        "--reference-column",
        required=True,
        help="Gold English reference (e.g. translations from gpt-5.4-nano)",
    )
    p.add_argument(
        "--hypothesis",
        nargs="+",
        required=True,
        metavar="COL",
        help="One or more columns to score (machine translations)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="JSON summary path (default: <input>_scores.json)",
    )
    p.add_argument(
        "--per-row-csv",
        type=Path,
        default=None,
        help="Optional CSV with per-row COMET/BLEURT scores",
    )
    p.add_argument(
        "--comet-batch-size",
        type=int,
        default=8,
        help="Batch size for COMET predict",
    )
    p.add_argument(
        "--bleurt-config",
        default=None,
        help="e.g. bleurt-large-512 (default: HF default bleurt-base-128)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Score only the first N usable rows (debug / quick check)",
    )
    args = p.parse_args()

    inp = args.input_csv.resolve()
    if not inp.is_file():
        print(f"File not found: {inp}", file=sys.stderr)
        sys.exit(1)

    need = {args.source_column, args.reference_column, *args.hypothesis}
    with open(inp, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("Empty CSV", file=sys.stderr)
            sys.exit(1)
        missing = need - set(reader.fieldnames)
        if missing:
            print(f"Missing columns: {sorted(missing)}", file=sys.stderr)
            sys.exit(1)
        rows = list(reader)

    usable: list[dict] = []
    for row in rows:
        ref = (row.get(args.reference_column) or "").strip()
        src = (row.get(args.source_column) or "").strip()
        if not ref or not src:
            continue
        hyps = {h: (row.get(h) or "").strip() for h in args.hypothesis}
        if not all(hyps.values()):
            continue
        usable.append({**row, "_src": src, "_ref": ref, "_hyps": hyps})

    n_skip = len(rows) - len(usable)
    if args.max_rows is not None:
        usable = usable[: args.max_rows]
    if not usable:
        print("No rows with non-empty source, reference, and all hypotheses.", file=sys.stderr)
        sys.exit(1)

    print(f"[score] Using {len(usable)} rows (skipped {n_skip} incomplete).", flush=True)

    try:
        comet_model = _load_comet()
    except Exception as e:
        print("Failed to load COMET. Install unbabel-comet and deps — see docstring.", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    try:
        bleurt = _load_bleurt(args.bleurt_config)
    except Exception as e:
        print("Failed to load BLEURT. Install BLEURT + TensorFlow — see docstring.", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    summary: dict = {
        "input": str(inp),
        "rows_used": len(usable),
        "rows_skipped_incomplete": n_skip,
        "source_column": args.source_column,
        "reference_column": args.reference_column,
        "comet_model": "Unbabel/wmt22-comet-da",
        "bleurt_config": args.bleurt_config or "default (bleurt-base-128)",
        "systems": {},
    }

    per_row_out: list[dict] = []
    for r in usable:
        row_out: dict = {}
        if r.get("id") is not None and str(r.get("id", "")).strip() != "":
            row_out["id"] = r["id"]
        per_row_out.append(row_out)

    for hyp_col in args.hypothesis:
        slug = _slug(hyp_col)
        comet_data = [
            {"src": r["_src"], "mt": r["_hyps"][hyp_col], "ref": r["_ref"]}
            for r in usable
        ]
        pred = comet_model.predict(comet_data, batch_size=args.comet_batch_size, gpus=0)
        comet_scores = list(pred["scores"])

        bleurt_res = bleurt.compute(
            predictions=[r["_hyps"][hyp_col] for r in usable],
            references=[r["_ref"] for r in usable],
        )
        bleurt_scores = list(bleurt_res["scores"])

        summary["systems"][hyp_col] = {
            "comet_mean": float(statistics.mean(comet_scores)),
            "comet_std": float(statistics.pstdev(comet_scores) if len(comet_scores) > 1 else 0.0),
            "comet_system": float(pred["system_score"]),
            "bleurt_mean": float(statistics.mean(bleurt_scores)),
            "bleurt_std": float(statistics.pstdev(bleurt_scores) if len(bleurt_scores) > 1 else 0.0),
        }

        for i in range(len(usable)):
            per_row_out[i][f"comet_{slug}"] = comet_scores[i]
            per_row_out[i][f"bleurt_{slug}"] = bleurt_scores[i]

    out = args.output
    if out is None:
        out = inp.with_name(f"{inp.stem}_scores.json")
    else:
        out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[score] Wrote {out}", flush=True)

    if args.per_row_csv:
        path = args.per_row_csv.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["id"] if any("id" in r for r in per_row_out) else []
        for pr in per_row_out:
            for k in pr:
                if k not in fieldnames:
                    fieldnames.append(k)
        if not fieldnames:
            fieldnames = list(per_row_out[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            w.writeheader()
            w.writerows(per_row_out)
        print(f"[score] Per-row: {path}", flush=True)


if __name__ == "__main__":
    main()
