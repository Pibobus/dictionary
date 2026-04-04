from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime, timezone
from typing import Any

import meilisearch
from openai import AsyncOpenAI


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _examples_text(examples: Any) -> str:
    if not examples:
        return ""
    if isinstance(examples, list):
        parts = [str(x).strip() for x in examples if str(x).strip()]
        return "\n".join(parts).strip()
    return str(examples).strip()

def _doc_get(doc: Any, key: str, default: Any = None) -> Any:
    if isinstance(doc, dict):
        return doc.get(key, default)
    if hasattr(doc, key):
        return getattr(doc, key)
    # Some meilisearch client versions wrap documents with a ._Document__doc dict
    if hasattr(doc, "__dict__"):
        d = getattr(doc, "__dict__", {}) or {}
        if isinstance(d, dict) and key in d:
            return d.get(key, default)
    return default


async def _embed_one(
    client: AsyncOpenAI,
    doc: Any,
    *,
    model: str,
    sem: asyncio.Semaphore,
) -> dict | None:
    doc_id = _doc_get(doc, "id")
    meaning = str(_doc_get(doc, "meaning") or "").strip()
    examples = _examples_text(_doc_get(doc, "examples"))

    if not doc_id:
        return None

    if _doc_get(doc, "meaning_embedding") and _doc_get(doc, "examples_embedding"):
        return None

    if not meaning and not examples:
        # Nothing to embed; keep doc unchanged.
        return None

    async with sem:
        resp = await client.embeddings.create(model=model, input=[meaning, examples])

    return {
        "id": doc_id,
        "meaning_embedding": list(resp.data[0].embedding),
        "examples_embedding": list(resp.data[1].embedding),
        "embeddings_backfilled_at": _utc_now(),
    }


async def main() -> int:
    p = argparse.ArgumentParser(description="Backfill meaning/examples embeddings in Meilisearch custom_dict.")
    p.add_argument("--meili-host", default=os.environ.get("MEILI_HOST1", "http://127.0.0.1:7700"))
    p.add_argument("--meili-api-key", default=os.environ.get("MEILI_API_KEY1", "masterKey123"))
    p.add_argument("--index", default="custom_dict")
    p.add_argument("--model", default="text-embedding-3-small")
    p.add_argument("--page-size", type=int, default=200)
    p.add_argument("--max-concurrency", type=int, default=5)
    args = p.parse_args()

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY env var.")

    ms = meilisearch.Client(args.meili_host, args.meili_api_key)
    index = ms.index(args.index)
    client = AsyncOpenAI(api_key=api_key)

    offset = 0
    total_seen = 0
    total_updated = 0
    sem = asyncio.Semaphore(max(1, args.max_concurrency))

    print(f"[{_utc_now()}] Starting backfill on index={args.index!r} host={args.meili_host!r}")

    while True:
        docs = await asyncio.to_thread(
            index.get_documents,
            {
                "offset": offset,
                "limit": args.page_size,
            },
        )
        docs_list = getattr(docs, "results", None)
        if docs_list is None:
            docs_list = docs
        docs_list = list(docs_list)
        if not docs_list:
            break

        total_seen += len(docs_list)
        to_update: list[dict] = []

        tasks = [asyncio.create_task(_embed_one(client, d, model=args.model, sem=sem)) for d in docs_list]
        for t in asyncio.as_completed(tasks):
            upd = await t
            if upd:
                to_update.append(upd)

        if to_update:
            task = await asyncio.to_thread(index.update_documents, to_update)
            await asyncio.to_thread(ms.wait_for_task, task.task_uid)
            total_updated += len(to_update)

        print(
            f"[{_utc_now()}] offset={offset} seen={total_seen} updated={total_updated} (+{len(to_update)})",
            flush=True,
        )

        offset += len(docs_list)

    print(f"[{_utc_now()}] Done. seen={total_seen} updated={total_updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

