from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class DictEntry(BaseModel):
    lemma: str
    classification: Literal["slang", "borrowed", "archaic", "neologism", "bad"]
    replacements: list[str]
    word_forms: list[str]
    meaning: str
    examples: list[str]
    meaning_embedding: list[float] | None = None
    examples_embedding: list[float] | None = None
    added_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class InjectedEntry(BaseModel):
    lemma: str
    classification: str
    span_start: int
    span_end: int
    replacement_used: str


class GeneratedPair(BaseModel):
    id: str
    clean_text: str
    polluted_text: str
    injected: list[InjectedEntry]
    model: str
    prompt_version: str
    source_entry_ids: list[str]
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    coverage_score: float
    semantic_cosine: float | None = None

    def to_csv_row(self) -> dict:
        data = self.model_dump()
        data["injected"] = json.dumps(
            [e.model_dump() for e in self.injected], ensure_ascii=False
        )
        data["source_entry_ids"] = json.dumps(
            self.source_entry_ids, ensure_ascii=False
        )
        if data.get("semantic_cosine") is None:
            data["semantic_cosine"] = ""
        return data


class RejectedPair(BaseModel):
    id: str
    clean_text: str
    polluted_text: str
    injected: list[InjectedEntry]
    model: str
    prompt_version: str
    source_entry_ids: list[str]
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    coverage_score: float
    semantic_cosine: float | None = None
    rejection_reason: str

    def to_csv_row(self) -> dict:
        data = self.model_dump()
        data["injected"] = json.dumps(
            [e.model_dump() for e in self.injected], ensure_ascii=False
        )
        data["source_entry_ids"] = json.dumps(
            self.source_entry_ids, ensure_ascii=False
        )
        if data.get("semantic_cosine") is None:
            data["semantic_cosine"] = ""
        return data
