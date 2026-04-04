from __future__ import annotations

from .models import DictEntry

_CLEAN_SYSTEM = (
    "You are a Ukrainian language dataset generator. "
    "Write one paragraph (2–4 sentences) in literary standard Ukrainian "
    "(літературна норма): neutral or mildly formal register, as in a short "
    "encyclopedia article, news commentary, or formal blog — not chat, not "
    "social media voice.\n\n"
    "Each sentence must be short: aim for roughly 6–12 words per sentence; "
    "never exceed about 15 words in a single sentence. "
    "One simple predicate per sentence; no semicolons; split ideas instead of "
    "chaining with commas. Avoid subordinate clauses except a single short one "
    "if unavoidable.\n\n"
    "Surrounding prose must stay strictly standard: no slang, no internet "
    "abbreviations or chatspeak, no anglicisms where a normal Ukrainian word "
    "exists, no deliberately colloquial particles or fillers used for "
    "authenticity.\n\n"
    "You will be given senses with STANDARD Ukrainian forms to use. "
    "For each item, use at least one of the listed standard forms naturally "
    "(match the grammatical case). "
    "Distribute them across different sentences — do not cluster. "
    "Do not name the task, the list, or meta-comment on word choice. "
    "Output only the paragraph, no title or preamble."
)


_POLLUTE_SYSTEM = (
    "You are a Ukrainian language dataset generator. "
    "You will receive a paragraph written in clean standard Ukrainian and a list "
    "of non-standard words (slang, borrowed, archaic, etc.) together with the "
    "standard equivalents they should replace.\n\n"
    "Rewrite the paragraph by substituting the standard equivalents with the "
    "corresponding non-standard words, as a native informal speaker would use them. "
    "Distribute the non-standard words across different sentences — do not cluster them. "
    "Each non-standard word must appear at least once. "
    "Preserve the same sentence count and boundaries as the source; do not merge "
    "sentences or split one sentence into two unless you must for grammar. "
    "Keep every sentence roughly the same length or shorter; never produce "
    "sentences longer than the originals.\n\n"
    "Wrap your output in XML tags exactly like this:\n"
    "<polluted>\n[rewritten paragraph here]\n</polluted>"
)


def _format_entry_for_clean(entry: DictEntry) -> str:
    examples = entry.examples[:2]
    example_str = examples[0] if examples else "—"
    replacements_str = ", ".join(entry.replacements)
    return (
        f"Standard Ukrainian (use in text): {replacements_str}\n"
        f"Meaning: {entry.meaning}\n"
        f"Example usage: {example_str}"
    )


def _format_entry_for_pollute(entry: DictEntry) -> str:
    examples = entry.examples[:2]
    example_str = examples[0] if examples else "—"
    replacements_str = ", ".join(entry.replacements)
    return (
        f"Non-standard: {entry.lemma}\n"
        f"Meaning: {entry.meaning}\n"
        f"Standard equivalents: {replacements_str}\n"
        f"Example usage: {example_str}"
    )


def build_clean_prompt(entries: list[DictEntry]) -> tuple[str, str]:
    """Build (system_msg, user_msg) for clean-text generation (Call 1)."""
    word_blocks = "\n\n".join(
        _format_entry_for_clean(e) for e in entries
    )
    user_msg = f"Standard forms to weave into the paragraph:\n\n{word_blocks}"
    return _CLEAN_SYSTEM, user_msg


def build_pollute_prompt(
    clean_text: str, entries: list[DictEntry]
) -> tuple[str, str]:
    """Build (system_msg, user_msg) for pollution rewriting (Call 2)."""
    word_blocks = "\n\n".join(
        _format_entry_for_pollute(e) for e in entries
    )
    user_msg = (
        f"Clean paragraph:\n{clean_text}\n\n"
        f"Words to substitute:\n\n{word_blocks}"
    )
    return _POLLUTE_SYSTEM, user_msg
