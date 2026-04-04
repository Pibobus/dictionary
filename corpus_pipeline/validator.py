from __future__ import annotations

import re

from .models import DictEntry, InjectedEntry

_COVERAGE_THRESHOLD = 0.8


def _find_span(text_lower: str, token: str) -> tuple[int, int] | None:
    """Find token as a whole-word match and return (start, end) char offsets."""
    pattern = re.compile(re.escape(token), re.IGNORECASE)
    m = pattern.search(text_lower)
    if m:
        return m.start(), m.end()
    return None


def compute_coverage(
    entries: list[DictEntry],
    clean_text: str,
    polluted_text: str,
) -> tuple[float, list[InjectedEntry]]:
    """Check that each entry's lemma / word_forms appear in polluted_text.

    Returns (coverage_score, list_of_injected_entries).
    """
    polluted_lower = polluted_text.lower()
    clean_lower = clean_text.lower()
    matched = 0
    injected: list[InjectedEntry] = []

    for entry in entries:
        candidates = [entry.lemma] + entry.word_forms
        found_span: tuple[int, int] | None = None
        found_form: str | None = None

        for form in candidates:
            span = _find_span(polluted_lower, form.lower())
            if span is not None:
                found_span = span
                found_form = form
                break

        if found_span is None:
            continue

        matched += 1

        replacement_used = ""
        for repl in entry.replacements:
            if repl.lower() in clean_lower:
                replacement_used = repl
                break

        injected.append(
            InjectedEntry(
                lemma=entry.lemma,
                classification=entry.classification,
                span_start=found_span[0],
                span_end=found_span[1],
                replacement_used=replacement_used,
            )
        )

    score = matched / len(entries) if entries else 0.0
    return score, injected


def validate(
    entries: list[DictEntry],
    clean_text: str,
    polluted_text: str,
) -> tuple[bool, float, list[InjectedEntry], str | None]:
    """Coverage check: each sampled entry must appear in the polluted text.

    Returns (passed, coverage_score, injected, reason).
    """
    coverage_score, injected = compute_coverage(entries, clean_text, polluted_text)

    if coverage_score < _COVERAGE_THRESHOLD:
        return (
            False,
            coverage_score,
            injected,
            f"coverage {coverage_score:.2f} < {_COVERAGE_THRESHOLD}",
        )

    return True, coverage_score, injected, None
