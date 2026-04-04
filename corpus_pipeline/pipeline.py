"""CLI entrypoint: python -m corpus_pipeline.pipeline --target-count 100"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from openai import AsyncOpenAI

from .config import (
    GENERATION_MODEL,
    OPENAI_API_KEY,
    OUTPUT_DIR,
    PIPELINE_CONCURRENCY,
    PROMPT_VERSION,
)
from .generator import generate_pair
from .models import GeneratedPair, RejectedPair
from .sampler import sample_entries
from .storage import CsvStorage
from .validator import validate


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_rejected(
    pair_id: str,
    clean: str,
    polluted: str,
    injected: list,
    coverage: float,
    entries: list,
    prompt_version: str,
    reason: str,
) -> RejectedPair:
    return RejectedPair(
        id=pair_id,
        clean_text=clean,
        polluted_text=polluted,
        injected=injected,
        model=GENERATION_MODEL,
        prompt_version=prompt_version,
        source_entry_ids=[e.lemma for e in entries],
        generated_at=_now_iso(),
        coverage_score=coverage,
        semantic_cosine=None,
        rejection_reason=reason,
    )


@dataclass
class _AttemptResult:
    pair_id: str
    entries: list
    lemmas: list[str]
    passed: bool
    clean: str
    polluted: str
    coverage: float
    injected: list
    reason: str | None


async def _attempt_one_pair(
    client: AsyncOpenAI,
    batch_size: int,
    prompt_version: str,
) -> _AttemptResult:
    pair_id = str(uuid.uuid4())
    entries = sample_entries(batch_size)
    lemmas = [e.lemma for e in entries]
    clean = polluted = ""
    coverage = 0.0
    injected: list = []
    reason: str | None = None
    passed = False

    generation_attempts = 0
    while generation_attempts < 2 and not passed:
        generation_attempts += 1
        try:
            clean, polluted = await generate_pair(entries, client)
        except ValueError as exc:
            reason = str(exc)
            break

        (
            passed,
            coverage,
            injected,
            reason,
        ) = validate(entries, clean, polluted)
        if not passed:
            print(
                f"  [{pair_id[:8]}] attempt {generation_attempts} rejected: "
                f"{reason} (coverage={coverage:.2f})",
                flush=True,
            )

    return _AttemptResult(
        pair_id=pair_id,
        entries=entries,
        lemmas=lemmas,
        passed=passed,
        clean=clean,
        polluted=polluted,
        coverage=coverage,
        injected=injected,
        reason=reason,
    )


async def run(
    target_count: int,
    batch_size: int = 3,
    prompt_version: str = PROMPT_VERSION,
    concurrency: int = PIPELINE_CONCURRENCY,
) -> None:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    storage = CsvStorage(OUTPUT_DIR)
    sem = asyncio.Semaphore(concurrency)
    state_lock = asyncio.Lock()
    accepted = 0
    total_attempts = 0

    print(
        f"[pipeline] Starting: target={target_count}, batch_size={batch_size}, "
        f"concurrency={concurrency}, prompt_version={prompt_version}",
        flush=True,
    )

    async def worker() -> None:
        nonlocal accepted, total_attempts
        while True:
            async with state_lock:
                if accepted >= target_count:
                    return
            async with sem:
                r = await _attempt_one_pair(client, batch_size, prompt_version)
            async with state_lock:
                total_attempts += 1
                if accepted >= target_count:
                    return
                if r.passed:
                    pair = GeneratedPair(
                        id=r.pair_id,
                        clean_text=r.clean,
                        polluted_text=r.polluted,
                        injected=r.injected,
                        model=GENERATION_MODEL,
                        prompt_version=prompt_version,
                        source_entry_ids=r.lemmas,
                        generated_at=_now_iso(),
                        coverage_score=r.coverage,
                        semantic_cosine=None,
                    )
                    storage.save_pair(pair)
                    accepted += 1
                    print(
                        f"  -> ACCEPTED [{accepted}/{target_count}] "
                        f"sample={r.lemmas} "
                        f"(coverage={r.coverage:.2f})",
                        flush=True,
                    )
                else:
                    storage.save_rejected(
                        _build_rejected(
                            r.pair_id,
                            r.clean,
                            r.polluted,
                            r.injected,
                            r.coverage,
                            r.entries,
                            prompt_version,
                            r.reason or "unknown",
                        )
                    )
                    print(
                        f"  -> REJECTED ({r.reason}) sample={r.lemmas}",
                        flush=True,
                    )
                if accepted >= target_count:
                    return

    await asyncio.gather(*(worker() for _ in range(concurrency)))

    print(
        f"\n[pipeline] Done. {accepted} pairs generated in {total_attempts} attempts.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate parallel Ukrainian corpus pairs"
    )
    parser.add_argument(
        "--target-count", type=int, required=True,
        help="Total number of accepted pairs to generate",
    )
    parser.add_argument(
        "--batch-size", type=int, default=3,
        help="Number of dictionary entries per pair (default: 3)",
    )
    parser.add_argument(
        "--prompt-version", type=str, default=PROMPT_VERSION,
        help=f"Prompt version tag (default: {PROMPT_VERSION})",
    )
    parser.add_argument(
        "--concurrency", type=int, default=PIPELINE_CONCURRENCY,
        help=f"Parallel workers (default: {PIPELINE_CONCURRENCY})",
    )
    args = parser.parse_args()

    try:
        asyncio.run(
            run(
                args.target_count,
                args.batch_size,
                args.prompt_version,
                concurrency=args.concurrency,
            )
        )
    except KeyboardInterrupt:
        print("\n[pipeline] Interrupted — data already flushed to CSV.")
        sys.exit(0)


if __name__ == "__main__":
    main()
