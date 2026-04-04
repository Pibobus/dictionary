"""
Paragraph analysis pipeline extracted from test words.ipynb.
AsyncOpenAI client must be passed per request (e.g. from Gradio with user API key).
"""

from __future__ import annotations

import asyncio
import glob
import json
import math
import os
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import emoji
import meilisearch
import spacy
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# --- paths & data ---

PROJECT_ROOT = Path(__file__).resolve().parent

invalid_tags = linguistic_tags = [
    "bad",
    "rare",
    "coll",
    "arch",
    "slang",
    "vulg",
    "obsc",
    "subst",
]


def load_vesum(path: str | Path | None = None) -> dict:
    if path is None:
        path = PROJECT_ROOT / "dic_data" / "dict_corp_lt.txt"
    path = Path(path)
    vesum: dict = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            form, lemma, tags = parts

            is_good = set(invalid_tags).isdisjoint(tags.split(":"))
            if form.lower() not in vesum:
                vesum[form.lower()] = [{lemma}, set(tags.split(":")).intersection(set(invalid_tags)), is_good]
            else:
                vesum[form.lower()][0].add(lemma)
                if not is_good:
                    vesum[form.lower()][1].add(tags.split(":")[-1])
                else:
                    vesum[form.lower()][2] = True

    for key in vesum:
        vesum[key][0] = list(vesum[key][0])
        vesum[key][1] = list(vesum[key][1])

    return vesum


def load_replacements(dict_dir: str | Path | None = None) -> dict:
    if dict_dir is None:
        dict_dir = PROJECT_ROOT / "dic_data" / "replacements"
    dict_dir = Path(dict_dir)
    replacements: dict = {}
    for path in glob.glob(str(dict_dir / "*.lst")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if "#>" not in line:
                    continue

                word_part, replacement_part = line.split("#>", 1)

                word = word_part.strip().split()[0]

                suggestions = [s.strip() for s in replacement_part.strip().split("|")]

                if word:
                    replacements[word] = replacements.get(word, []) + suggestions

    return replacements


vesum = load_vesum()
replacements = load_replacements()


def query(word: str) -> dict:
    entry = vesum.get(word)

    if not entry:
        return {"known": False}

    lemma, tags, is_possibly_good = entry

    return {
        "known": True,
        "lemmas": lemma,
        "is_bad": not set(linguistic_tags).isdisjoint(tags),
        "is_possibly_good": is_possibly_good,
        "tags": tags,
    }


# --- spaCy ---

nlp = spacy.load("uk_core_news_sm")


# --- Meilisearch ---

MEILI_HOST = os.environ.get("MEILI_HOST", "http://127.0.0.1:7700")
MEILI_API_KEY = os.environ.get("MEILI_API_KEY", "masterKey123")

ms_client = meilisearch.Client(MEILI_HOST, MEILI_API_KEY)


def _ensure_index(uid: str, primary_key: str = "id") -> None:
    try:
        ms_client.get_index(uid)
    except Exception:
        try:
            ms_client.create_index(uid, {"primaryKey": primary_key})
        except Exception as e:
            print(f"[meilisearch] could not create index {uid}: {e}")


_ensure_index("custom_dict")
_ensure_index("rejected_words")

index = ms_client.index("custom_dict")

try:
    index.update_settings(
        {
            "searchableAttributes": ["lemma", "word_forms"],
            "filterableAttributes": ["classification"],
            "typoTolerance": {
                "enabled": True,
                "minWordSizeForTypos": {"oneTypo": 4, "twoTypos": 8},
            },
            "prefixSearch": "disabled",
        }
    )
except Exception as e:
    print(f"[meilisearch] custom_dict settings: {e}")


def make_id() -> str:
    return str(uuid.uuid4())


class DictEntry(BaseModel):
    lemma: str
    classification: Literal["slang", "borrowed", "archaic", "neologism", "bad"]
    replacements: list[str]
    word_forms: list[str]
    meaning: str
    examples: list[str]
    meaning_embedding: list[float] | None = None
    examples_embedding: list[float] | None = None
    added_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ParagraphResult(BaseModel):
    fixed_paragraph: str


async def add_to_meilisearch(
    entry: DictEntry,
    client: AsyncOpenAI,
    *,
    lookup_word: str | None = None,
    similarity_threshold: float = 0.75,
):
    if client is None:
        raise ValueError("OpenAI client is required to add dictionary entries (embeddings).")

    meaning_text = (entry.meaning or "").strip()
    examples_text = "\n".join([e.strip() for e in (entry.examples or []) if str(e).strip()]).strip()

    if not meaning_text and not examples_text:
        raise ValueError("Cannot embed empty meaning and examples.")

    if meaning_text or examples_text:
        emb_resp = await client.embeddings.create(
            model="text-embedding-3-small",
            input=[meaning_text, examples_text],
        )
        entry.meaning_embedding = list(emb_resp.data[0].embedding)
        entry.examples_embedding = list(emb_resp.data[1].embedding)

    # Standard lookup (as usual) to find same-word candidates.
    compare_word = (lookup_word or entry.lemma or "").strip()
    hits: list[dict] = []
    if compare_word:
        hits = await lookup_custom_dict(compare_word, limit=20)

    best_score = 0.0
    best_hit: dict | None = None
    if entry.meaning_embedding and entry.examples_embedding and hits:
        for h in hits:
            hm = h.get("meaning_embedding")
            he = h.get("examples_embedding")
            if not isinstance(hm, list) or not isinstance(he, list):
                continue
            try:
                hm_f = [float(x) for x in hm]
                he_f = [float(x) for x in he]
            except Exception:
                continue
            meaning_sim = _cosine_similarity(entry.meaning_embedding, hm_f)
            examples_sim = _cosine_similarity(entry.examples_embedding, he_f)
            score = 0.8 * meaning_sim + 0.2 * examples_sim
            if score > best_score:
                best_score = score
                best_hit = h
            if score >= similarity_threshold:
                _log_embedding_duplicate(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "lookup_word": lookup_word,
                        "lemma_new": entry.lemma,
                        "score": round(score, 6),
                        "meaning_cosine": round(meaning_sim, 6),
                        "examples_cosine": round(examples_sim, 6),
                        "threshold": similarity_threshold,
                        "new": {
                            "meaning": entry.meaning,
                            "examples": entry.examples,
                            "classification": entry.classification,
                            "replacements": entry.replacements,
                            "word_forms": entry.word_forms,
                        },
                        "existing": {
                            "id": h.get("id"),
                            "lemma": h.get("lemma"),
                            "meaning": h.get("meaning"),
                            "examples": h.get("examples"),
                            "classification": h.get("classification"),
                            "replacements": h.get("replacements"),
                            "word_forms": h.get("word_forms"),
                        },
                    }
                )
                return None

    doc = entry.model_dump()
    doc["id"] = make_id()

    def _add():
        task = index.add_documents([doc])
        return ms_client.wait_for_task(task.task_uid)

    if best_hit is not None and best_score > 0:
        print(
            f"[meilisearch] best embedding score for {compare_word!r}: {best_score:.3f} (kept, below threshold)",
            flush=True,
        )

    return await asyncio.to_thread(_add)


async def lookup_custom_dict(word: str, limit: int = 3) -> list[dict]:
    return await asyncio.to_thread(_lookup_sync, word, limit)


def _lookup_sync(word: str, limit: int) -> list[dict]:
    results = index.search(
        word,
        {
            "limit": limit,
            "attributesToSearchOn": ["lemma", "word_forms"],
        },
    )
    return results.get("hits", [])


rejected_index = ms_client.index("rejected_words")
try:
    rejected_index.update_settings(
        {
            "searchableAttributes": ["word", "reason"],
            "filterableAttributes": ["expires_at", "word"],
            "sortableAttributes": ["expires_at"],
        }
    )
except Exception as e:
    print(f"[meilisearch] rejected_words settings: {e}")

REJECTION_TTL_DAYS = 30


async def add_to_rejected(word: str, skip_reason: str):
    expires_at = (datetime.now() + timedelta(days=REJECTION_TTL_DAYS)).isoformat()

    await asyncio.to_thread(
        rejected_index.add_documents,
        [
            {
                "id": make_id(),
                "word": word,
                "skip_reason": skip_reason,
                "rejected_at": datetime.now().isoformat(),
                "expires_at": expires_at,
            }
        ],
    )


async def is_rejected(word: str) -> bool:
    try:
        results = await asyncio.to_thread(
            rejected_index.search,
            word,
            {
                "filter": f"word = '{word}' AND expires_at > '{datetime.now().isoformat()}'",
                "attributesToSearchOn": ["word"],
                "limit": 1,
            },
        )
        return len(results["hits"]) > 0
    except Exception:
        return False


# --- text heuristics ---

URL_REGEX = re.compile(r"https?://\S+|www\.\S+")

TECH_REGEX = re.compile(
    r"""
    ^[a-zA-Z0-9\.-]+\.(com|net|org|io|js|py|dev)$ |
    ^[a-zA-Z][a-zA-Z0-9_\.-]*$
""",
    re.VERBOSE,
)


def is_url(word: str) -> bool:
    return bool(URL_REGEX.match(word))


def is_tech_token(word: str) -> bool:
    return bool(TECH_REGEX.match(word))


def is_mixed(word: str) -> bool:
    return any(c.isalpha() for c in word) and any(c.isdigit() for c in word)


def has_special_tech_chars(word: str) -> bool:
    return any(c in word for c in "+#.")


def is_named_entity_like(word: str) -> bool:
    return (
        is_url(word)
        or is_mixed(word)
        or is_tech_token(word)
        or has_special_tech_chars(word)
        or any("a" <= c.lower() <= "z" for c in word)
    )


def remove_emojis(text: str) -> str:
    return emoji.replace_emoji(text, replace="")


def normalize_apostrophes(text: str) -> str:
    return (
        text.replace("’", "'")
        .replace("`", "'")
        .replace("ʼ", "'")
        .replace("ʹ", "'")
    )


def normalize_text(text: str) -> str:
    text = remove_emojis(text)
    text = normalize_apostrophes(text)
    return text


# --- candidates ---


async def extract_candidates(paragraph: str, account_for_untypical_usage: bool = True) -> dict:
    paragraph = normalize_text(paragraph)
    doc = nlp(paragraph)
    named_entities = {ent.text.lower() for ent in doc.ents}

    words = set()
    for token in doc:
        if token.is_punct or token.is_space or token.like_num:
            continue
        words.add(token.text.lower())

    result = {
        "entities": [],
        "standard": [],
        "known_bad": [],
        "known_bad_no_repl": [],
        "in_custom_dict": [],
        "unknown": [],
    }

    for word in words:
        clean = word.strip(".,!?;:")

        if clean in named_entities or is_named_entity_like(clean):
            result["entities"].append(clean)
            continue

        vesum_result = query(clean)
        if vesum_result["known"] and not vesum_result["is_bad"]:
            result["standard"].append(clean)
            continue

        rejected, custom_matches = await asyncio.gather(
            is_rejected(clean),
            lookup_custom_dict(clean),
        )

        if rejected:
            result["standard"].append(clean)
            continue

        if custom_matches:
            result["in_custom_dict"].append(
                {
                    "word": clean,
                    "entries": [
                        {
                            "replacements": w["replacements"],
                            "meaning": w["meaning"],
                            "examples": w["examples"],
                        }
                        for w in custom_matches
                    ],
                }
            )
            continue

        if not vesum_result["known"]:
            result["unknown"].append(clean)
            continue

        if vesum_result["is_bad"] and not (vesum_result["is_possibly_good"] and not account_for_untypical_usage):
            lemmas = vesum_result.get("lemmas", [])

            word_replacements = list(
                {
                    repl
                    for lemma in lemmas
                    for repl in replacements.get(lemma, [])
                }
            )

            entry = {
                "word": clean,
                "vesum": vesum_result,
                "replacements": word_replacements,
            }

            if word_replacements:
                result["known_bad"].append(entry)
            else:
                result["known_bad_no_repl"].append(entry)

            continue

        result["standard"].append(clean)

    return result


LOG_FILE = PROJECT_ROOT / "llm_search_logs.jsonl"
EMBEDDING_DUPLICATES_LOG_FILE = PROJECT_ROOT / "embedding_duplicates.jsonl"


def log_event(event: dict):
    line = json.dumps(event, ensure_ascii=False) + "\n"
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line)
    # Mirror to stdout so UIs (e.g. Gradio) can show the same records as the jsonl file
    print(f"[llm_search_logs.jsonl] {line.rstrip()}", flush=True)


def json_dumps(obj) -> str:
    def default(o):
        if isinstance(o, set):
            return list(o)
        if isinstance(o, BaseModel):
            return o.model_dump()
        if hasattr(o, "__dict__"):
            return o.__dict__
        raise TypeError(f"Not serializable: {type(o)}")

    return json.dumps(obj, ensure_ascii=False, default=default)


def _log_embedding_duplicate(event: dict) -> None:
    line = json.dumps(event, ensure_ascii=False) + "\n"
    with EMBEDDING_DUPLICATES_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line)
    print(f"[embedding_duplicates.jsonl] {line.rstrip()}", flush=True)


def _l2_norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    na = _l2_norm(a)
    nb = _l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (na * nb)


class SearchResult(BaseModel):
    worth_adding: bool
    skip_reason: str | None = None
    leave_as_is: bool = False

    lemma: str | None = None
    classification: Literal["slang", "borrowed", "archaic", "neologism", "bad"] | None = None
    replacements: list[str] = []
    word_forms: list[str] = []
    meaning: str | None = None
    examples: list[str] = []
    added_at: str = Field(default_factory=lambda: datetime.now().isoformat())


def extract_sentence_with_neighbors(text: str, word: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for i, s in enumerate(sentences):
        if word.lower() in s.lower():
            context = [s]

            if i > 0:
                context.insert(0, sentences[i - 1])

            if i < len(sentences) - 1:
                context.append(sentences[i + 1])

            return " ".join(context).strip()

    return text[:200]


async def search_unknown_words(
    unknown_words: list,
    no_repl_words: list,
    paragraph: str,
    client: AsyncOpenAI,
    *,
    max_concurrent: int = 20,
) -> dict:
    words_to_search = []

    for word in unknown_words:
        words_to_search.append(
            {
                "word": word,
                "type": "unknown",
                "possible_lemmas": [],
                "is_possibly_good": False,
                "vesum_tags": None,
            }
        )

    for w in no_repl_words:
        words_to_search.append(
            {
                "word": w["word"],
                "type": "known_bad",
                "possible_lemmas": w["vesum"]["lemmas"],
                "is_possibly_good": w["vesum"]["is_possibly_good"],
                "vesum_tags": w["vesum"]["tags"],
            }
        )

    if not words_to_search:
        return {}

    sem = asyncio.Semaphore(max_concurrent)

    async def search_one(item):
        async with sem:
            return await _search_one_word(item)

    async def _search_one_word(item):
        word = item["word"]
        context = extract_sentence_with_neighbors(paragraph, word)
        tags = item["vesum_tags"]
        is_possibly_good = item["is_possibly_good"]
        possible_lemmas = item["possible_lemmas"]
        word_type = item["type"]

        start_time = time.time()

        try:
            response = await client.responses.parse(
                model="gpt-5.4",
                tools=[
                    {
                        "type": "web_search",
                    }
                ],
                temperature=0.1,
                max_output_tokens=1000,
                tool_choice="required",
                text_format=SearchResult,
                input=f"""
You are analyzing a Ukrainian word for inclusion in a custom dictionary. Entries may be classified as slang, borrowed (loanwords), archaic, neologism, or bad — use the definitions below; do not label loanwords as slang unless informality is the main signal.
When searching, prefer results from:
- reddit.com
- goroh.pp.ua
- myslovo.com
- dou.ua
- twitter.com
- wikipedia.org
Word: "{word}"

Metadata:
- Source: VESUM dictionary
- Tags: {tags}
- Possible lemmas: {possible_lemmas}
- Possibly standard/acceptable: {is_possibly_good}
- Word type: {word_type}

Context:
The word appears in the following paragraph:
"{context}"

Instructions:

1. Determine the meaning of the word strictly based on:
   - the provided context
   - real-world usage (via search)

2. Use the web search tool to:
   - find real usage examples of "{word}" (Ukrainian or Russian, Cyrillic)
   - extract 1-3 full example sentences (no links)
   - include the provided paragraph as one example

3. Infer the meaning from actual usage. Do NOT guess if evidence is weak.

4. Decide whether the word is worth adding to the dictionary:

   Mark as NOT worth adding if:
   - it is a standard Ukrainian word
   - it is a named entity (person, place, brand, etc.)
   - it has no clear or consistent meaning
   - it is noise or malformed text

   Mark as worth adding if:
   - it is non-standard in a way worth recording (slang register, loanword worth documenting, archaic, neologism, or bad form)
   - it has clear meaning supported by multiple usages

5. Special rule:
   - If `is_possibly_good = True` AND the word is a known standard Ukrainian form → mark as NOT worth adding

6. If the word IS worth adding:
   - provide a clear meaning
   - suggest the best standard Ukrainian replacement (if applicable)
   - Return all inflected forms of the word (same lemma only). Do not include phrases, prepositions, or derived words.
   - Set `classification` to EXACTLY ONE of: slang | borrowed | archaic | neologism | bad (see definitions below).


   Classification definitions (choose carefully; do not confuse borrowed with slang):
   - **borrowed**: a **loanword / internationalism** — foreign origin or calque widely adopted into Ukrainian (including Latin/Greek-based terms, technical terms, everyday loans like кава, менеджер). Often neutral or formal register; etymology is the main reason it is “non-native”. Use this when the item is primarily a **standard or conventional borrowing**, not informal by nature.
   - **slang**: **informal / colloquial / jargon / internet or subculture register** — stylistically marked as non-neutral (жаргон, просторіччя, молодіжне мовлення). Use slang when the **register** is informal or group-specific, even if the word once came from another language. Do NOT default foreign-looking words to slang.
   - **archaic**: outdated or rarely used in modern Ukrainian.
   - **neologism**: newly coined or very recent word/formation.
   - **bad**: prescriptively incorrect or non-standard spelling/form (align with “incorrect” usage), not merely borrowed.

   Disambiguation (borrowed vs slang):
   - If evidence shows **neutral dictionary adoption** or **international/technical term** in ordinary or formal text → **borrowed**, not slang.
   - If evidence shows **deliberately informal, playful, or in-group** usage as the main trait → **slang**.
   - If both foreign origin and informality apply: prefer **borrowed** when the word is listed/used as a normal loan; prefer **slang** only when informality is the defining feature in context.

7. If the word is NOT worth adding:
   - explain why briefly
   - set `leave_as_is = true`

Rules:
- Base conclusions ONLY on evidence (context + search)
- Do NOT hallucinate meanings or examples
- Prefer rejecting over guessing
- Be concise and factual
- Use Ukrainian language only.

""",
            )

            duration = time.time() - start_time

            try:
                raw = json.loads(response.output_text)
                search_result = SearchResult(**raw)
            except Exception:
                search_result = SearchResult(
                    worth_adding=False,
                    skip_reason="failed to parse model response",
                    leave_as_is=True,
                )

            if search_result.worth_adding:
                entry = DictEntry(
                    lemma=search_result.lemma,
                    classification=search_result.classification,
                    replacements=search_result.replacements,
                    word_forms=search_result.word_forms,
                    meaning=search_result.meaning,
                    examples=search_result.examples,
                )
                await add_to_meilisearch(entry, client, lookup_word=word)
                log_event(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "word": word,
                        "status": "success",
                        "duration_sec": round(duration, 3),
                        "worth_adding": search_result.worth_adding,
                        "leave_as_is": search_result.leave_as_is,
                        "output_text": response.output_text,
                    }
                )
            else:
                log_event(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "word": word,
                        "status": "rejected",
                        "leave_as_is": search_result.leave_as_is,
                        "reason": search_result.skip_reason,
                    }
                )

                await add_to_rejected(word, search_result.skip_reason or "")

            return word, {
                "search_result": search_result,
                "output_text": response.output_text,
            }

        except Exception as e:
            duration = time.time() - start_time

            log_event(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "word": word,
                    "status": "error",
                    "duration_sec": round(duration, 3),
                    "error": str(e),
                }
            )

            return word, None

    tasks = [search_one(item) for item in words_to_search]
    results_list = await asyncio.gather(*tasks)

    return {word: result for word, result in results_list}


class WordNeedsSearch(BaseModel):
    word: str
    reason: str


class AnalyzeOnlyAnalysis(BaseModel):
    words_needing_clarification: list[WordNeedsSearch]


class ParagraphAnalysis(BaseModel):
    fixed_paragraph: str
    words_needing_clarification: list[WordNeedsSearch]


async def analyze_paragraph(
    paragraph: str,
    client: AsyncOpenAI,
    account_for_untypical_usage: bool = False,
    fix_paragraph: bool = True,
) -> ParagraphResult:
    candidates = await extract_candidates(paragraph, account_for_untypical_usage)

    search_results = await search_unknown_words(
        candidates["unknown"],
        candidates["known_bad_no_repl"],
        paragraph,
        client,
    )

    needs_llm = (
        candidates["unknown"]
        or candidates["known_bad"]
        or candidates["known_bad_no_repl"]
        or candidates["in_custom_dict"]
    )

    if not needs_llm:
        return ParagraphResult(
            fixed_paragraph=paragraph,
        )

    system_content = f"""You are a Ukrainian language expert.

For each word from that specific list {candidates["unknown"] + [w["word"] for w in candidates["known_bad"]] + [w["word"] for w in candidates["known_bad_no_repl"]] + list(set([w["word"] for w in candidates["in_custom_dict"]]))}:
1. If word comes from search results — it already fits context, replacement is appropriate
2. If word is marked leave_as_is=True — leave it unchanged
3. If the word has replacements from the custom dictionary:
   - Evaluate each replacement using the provided meanings and examples of the original word.
   - Select the replacement that best matches the meaning in this specific context.
   - The dictionary may return multiple entries for the same word, each with different meanings and corresponding replacements.
   - For each occurrence of the word in context:
     - Analyze all dictionary entries and their meanings.
     - Select the replacement that best matches the intended meaning in the given context.
     - If no single replacement is clearly appropriate after considering all meanings and options:
       - Add the word to words_needing_clarification.
       - Include a short reason explaining the ambiguity (e.g., multiple plausible meanings, insufficient context, conflicting replacements).
4. If the word has replacements from VESUM:
   - Evaluate each replacement in the given context.
   - Select the most contextually appropriate replacement.
   - If no replacement clearly fits the context, add the word to words_needing_clarification with a short reason.


When multiple replacements exist pick the most contextually appropriate one.
Use correct grammatical forms for all replacements.
Use Ukrainian language only.
Do NOT invent replacements — only use provided ones.
"""

    if fix_paragraph:
        system_content += "\nReturn the fully fixed paragraph in fixed_paragraph field. Leave the words_needing_clarification untouched. Return the flagged words in the words_needing_clarification field."
    else:
        system_content += "\nReturn only the words that cannot be confidently resolved or replaced based on the given context and available information in the words_needing_clarification field."

    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": f"""
Paragraph: {paragraph}

Words in our custom dictionary with known meanings and replacements:
{json_dumps(candidates["in_custom_dict"])}

Non-standard words with VESUM suggested replacements (list — pick best for context):
{json_dumps(candidates["known_bad"])}

Non-standard words with no replacements + unknown words:
{json_dumps(candidates["known_bad_no_repl"] + candidates["unknown"])}

Search results from web (already context-appropriate):
{json_dumps(search_results)}
""",
        },
    ]

    output_format = ParagraphAnalysis if fix_paragraph else AnalyzeOnlyAnalysis

    result = await client.responses.parse(
        model="gpt-5.4",
        temperature=0.3,
        input=messages,
        text_format=output_format,
    )

    analysis = result.output_parsed

    if not analysis.words_needing_clarification:
        print(f"Resolved in 1 iteration(s)")
        if analysis is None:
            return ParagraphResult(
                fixed_paragraph=paragraph,
            )

        return ParagraphResult(
            fixed_paragraph=analysis.fixed_paragraph if fix_paragraph else paragraph,
        )

    flagged = [w.word.lower() for w in analysis.words_needing_clarification]
    print(f"Iteration 2: needs clarification for {flagged}")

    new_results = await search_unknown_words(
        unknown_words=flagged,
        no_repl_words=[],
        paragraph=paragraph,
        client=client,
    )

    if fix_paragraph:
        messages[-1] = {
            "role": "user",
            "content": f"""
                Partially fixed paragraph: {analysis.fixed_paragraph}
                Search results from web for marked words (already context-appropriate):
                {json_dumps(new_results)}

                Fix the partially fixed paragraph using new search results and return the completely fixed paragraph.
                """,
        }
        result = await client.responses.parse(
            model="gpt-5.4",
            input=messages,
            temperature=0.3,
            text_format=ParagraphResult,
        )

        analysis = result.output_parsed

    if analysis is None:
        return ParagraphResult(
            fixed_paragraph=paragraph,
        )

    return ParagraphResult(
        fixed_paragraph=analysis.fixed_paragraph if fix_paragraph else paragraph,
    )
