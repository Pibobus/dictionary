from __future__ import annotations

import math
import random
from collections import defaultdict

import meilisearch

from .config import MEILI_API_KEY, MEILI_HOST
from .models import DictEntry

_client = meilisearch.Client(MEILI_HOST, MEILI_API_KEY)
_index = _client.index("custom_dict")

_cached_entries: list[DictEntry] = []


def _get_all_entries() -> list[DictEntry]:
    """Paginate through the custom_dict index and cache all entries."""
    global _cached_entries
    if _cached_entries:
        return _cached_entries

    entries: list[DictEntry] = []
    offset = 0
    limit = 100
    while True:
        result = _index.get_documents({"limit": limit, "offset": offset})
        docs = result.results
        if not docs:
            break
        for doc in docs:
            d = doc if isinstance(doc, dict) else dict(doc)
            d.pop("meaning_embedding", None)
            d.pop("examples_embedding", None)
            d.pop("id", None)
            entries.append(DictEntry(**d))
        if len(docs) < limit:
            break
        offset += limit

    _cached_entries = entries
    return _cached_entries


def sample_entries(n: int = 3) -> list[DictEntry]:
    """Return *n* entries with stratified sampling across classification types."""
    all_entries = _get_all_entries()
    if not all_entries:
        raise RuntimeError("No entries found in Meilisearch custom_dict index")

    by_class: dict[str, list[DictEntry]] = defaultdict(list)
    for entry in all_entries:
        by_class[entry.classification].append(entry)

    classes = list(by_class.keys())
    num_classes = len(classes)
    max_per_class = math.ceil(n / num_classes)

    selected: list[DictEntry] = []
    for cls in classes:
        pool = by_class[cls]
        take = min(max_per_class, len(pool), n - len(selected))
        if take <= 0:
            continue
        selected.extend(random.sample(pool, take))
        if len(selected) >= n:
            break

    if len(selected) < n:
        remaining_pool = [e for e in all_entries if e not in selected]
        extra = min(n - len(selected), len(remaining_pool))
        selected.extend(random.sample(remaining_pool, extra))

    return selected[:n]
