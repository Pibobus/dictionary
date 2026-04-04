from __future__ import annotations

import re

from openai import AsyncOpenAI

from .config import GENERATION_MODEL
from .models import DictEntry
from .prompt_builder import build_clean_prompt, build_pollute_prompt

_POLLUTED_RE = re.compile(r"<polluted>(.*?)</polluted>", re.DOTALL)
_MAX_RETRIES = 2


async def _chat(
    client: AsyncOpenAI,
    system: str,
    user: str,
    model: str = GENERATION_MODEL,
) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.9,
    )
    return resp.choices[0].message.content or ""


def _parse_polluted(text: str) -> str | None:
    m = _POLLUTED_RE.search(text)
    return m.group(1).strip() if m else None


async def generate_pair(
    entries: list[DictEntry],
    client: AsyncOpenAI,
) -> tuple[str, str]:
    """Generate a (clean_text, polluted_text) pair using two LLM calls.

    Call 1: produce clean standard Ukrainian text.
    Call 2: rewrite it with non-standard substitutions.
    Retries up to _MAX_RETRIES times on polluted-tag parse failure.
    """
    sys_clean, usr_clean = build_clean_prompt(entries)
    clean_text = (await _chat(client, sys_clean, usr_clean)).strip()

    sys_poll, usr_poll = build_pollute_prompt(clean_text, entries)

    for attempt in range(_MAX_RETRIES + 1):
        if attempt > 0:
            usr_poll_retry = (
                usr_poll
                + "\n\nYou MUST wrap your output in <polluted>...</polluted> XML tags."
            )
        else:
            usr_poll_retry = usr_poll

        raw = await _chat(client, sys_poll, usr_poll_retry)
        polluted_text = _parse_polluted(raw)
        if polluted_text is not None:
            return clean_text, polluted_text

    raise ValueError(
        f"Failed to parse <polluted> tags after {_MAX_RETRIES + 1} attempts. "
        f"Last response: {raw[:300]}"
    )
