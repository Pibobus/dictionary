from __future__ import annotations

import csv
import os
from pathlib import Path

from .config import OUTPUT_DIR
from .models import GeneratedPair, RejectedPair

_PAIRS_FIELDS = [
    "id",
    "clean_text",
    "polluted_text",
    "injected",
    "model",
    "prompt_version",
    "source_entry_ids",
    "generated_at",
    "coverage_score",
    "semantic_cosine",
]

_REJECTED_FIELDS = _PAIRS_FIELDS + ["rejection_reason"]


class CsvStorage:
    """Append-only CSV storage for generated and rejected pairs."""

    def __init__(self, output_dir: str = OUTPUT_DIR) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._pairs_path = self._dir / "generated_pairs.csv"
        self._rejected_path = self._dir / "rejected_pairs.csv"

        self._ensure_header(self._pairs_path, _PAIRS_FIELDS)
        self._ensure_header(self._rejected_path, _REJECTED_FIELDS)

    @staticmethod
    def _ensure_header(path: Path, fields: list[str]) -> None:
        if not path.exists():
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
                writer.writeheader()

    def save_pair(self, pair: GeneratedPair) -> None:
        row = pair.to_csv_row()
        with open(self._pairs_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=_PAIRS_FIELDS, quoting=csv.QUOTE_ALL
            )
            writer.writerow(row)

    def save_rejected(self, pair: RejectedPair) -> None:
        row = pair.to_csv_row()
        with open(self._rejected_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=_REJECTED_FIELDS, quoting=csv.QUOTE_ALL
            )
            writer.writerow(row)
