"""Phase B memory upgrades: the real implementations of C2/C3/C4.

Each class subclasses its starter counterpart (so the corruption injector
and analysis code work unchanged) and replaces the starter mechanism with
the real one:

  EmbeddingRAG   C2: cosine retrieval over embedded chunks
  LLMSummary     C3: LLM rolling summarization (replaces truncation)
  LLMFacts       C4: LLM fact extraction (replaces domain regexes)

Every LLM/embedding call made by a memory object is METERED on the object
(`meter` dict) so the pilot runner can charge memory-system cost into CAMU —
the point of the benchmark is that extraction/summarization cost is part of
the architecture's price.

All model calls go through injectable functions (constructor args), so unit
tests run at $0 with fakes; the default functions call LiteLLM with a disk
cache under .memcache/ keyed by content hash, making reruns cheap and
stable.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

from .memory import KeywordRAG, RollingSummary, StructuredFacts, Hybrid

CACHE_DIR = Path(os.environ.get("MERIT_MEM_CACHE", ".memcache"))


def _cached(kind: str, key_text: str, compute):
    key = hashlib.sha256(f"{kind}|{key_text}".encode()).hexdigest()[:32]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text()), True
    value = compute()
    path.write_text(json.dumps(value))
    return value, False


def default_embed_fn(model: str):
    """Returns embed(texts) -> (vectors, total_tokens), disk-cached."""
    def embed(texts: list[str]):
        import litellm
        vectors, tokens = [], 0

        for t in texts:
            def compute():
                resp = litellm.embedding(model=model, input=[t])
                return {"v": resp.data[0]["embedding"],
                        "tokens": resp.usage.prompt_tokens}
            hit, was_cached = _cached(f"emb|{model}", t, compute)
            vectors.append(hit["v"])
            tokens += 0 if was_cached else hit["tokens"]
        return vectors, tokens
    return embed


def default_llm_fn(model: str):
    """Returns llm(system, user) -> (text, prompt_toks, completion_toks),
    disk-cached, temperature 0."""
    def llm(system: str, user: str):
        import litellm

        def compute():
            resp = litellm.completion(
                model=model, temperature=0,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            return {"text": resp.choices[0].message.content,
                    "pt": resp.usage.prompt_tokens,
                    "ct": resp.usage.completion_tokens}
        hit, was_cached = _cached(f"llm|{model}", f"{system}\x00{user}",
                                  compute)
        if was_cached:
            return hit["text"], 0, 0
        return hit["text"], hit["pt"], hit["ct"]
    return llm


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class EmbeddingRAG(KeywordRAG):
    """C2 upgraded: chunks embedded at write time; cosine top-k at read."""
    name = "C2_rag"

    def __init__(self, embed_fn, chunk_chars: int = 600,
                 top_k: int = 4) -> None:
        super().__init__(chunk_chars=chunk_chars, top_k=top_k)
        self.embed_fn = embed_fn
        self._vectors: list[list[float]] = []
        self.meter = {"prompt_tokens": 0, "completion_tokens": 0,
                      "embedding_tokens": 0}

    def write(self, episode_id: str, transcript: str) -> None:
        start = len(self.records)
        super().write(episode_id, transcript)
        new_texts = [r.text for r in self.records[start:]]
        if new_texts:
            vecs, toks = self.embed_fn(new_texts)
            self._vectors.extend(vecs)
            self.meter["embedding_tokens"] += toks

    def read(self, current_context: str, budget_chars: int = 4000) -> str:
        if not self.records:
            return ""
        # corruption may append records without vectors; embed stragglers
        if len(self._vectors) < len(self.records):
            vecs, toks = self.embed_fn(
                [r.text for r in self.records[len(self._vectors):]])
            self._vectors.extend(vecs)
            self.meter["embedding_tokens"] += toks
        (qvec,), toks = self.embed_fn([current_context])
        self.meter["embedding_tokens"] += toks
        ranked = sorted(range(len(self.records)),
                        key=lambda i: -_cosine(qvec, self._vectors[i]))
        out, used = [], 0
        for i in ranked[: self.top_k]:
            text = self.records[i].text
            if used + len(text) > budget_chars:
                break
            out.append(text)
            used += len(text)
        return "\n---\n".join(out)


SUMMARY_SYSTEM = (
    "You maintain the agent's running notes across customer/ops sessions. "
    "Summarize the episode transcript in at most 8 short lines. You MUST "
    "preserve verbatim every identifier, amount, address, version, room, "
    "place, day and time, and every agreed-but-not-yet-executed action. "
    "If a value supersedes an earlier one, state only the latest.")


class LLMSummary(RollingSummary):
    """C3 upgraded: LLM summarization instead of truncation."""
    name = "C3_summary"

    def __init__(self, llm_fn, max_chars: int = 4000) -> None:
        super().__init__(max_chars=max_chars)
        self.llm_fn = llm_fn
        self.meter = {"prompt_tokens": 0, "completion_tokens": 0,
                      "embedding_tokens": 0}

    def summarize(self, transcript: str) -> str:
        text, pt, ct = self.llm_fn(SUMMARY_SYSTEM, transcript)
        self.meter["prompt_tokens"] += pt
        self.meter["completion_tokens"] += ct
        return text.strip()


EXTRACT_SYSTEM = (
    "Extract durable facts from the episode transcript as JSON lines, one "
    "object per line: {\"entity\": ..., \"attribute\": ..., \"value\": ...}. "
    "Entities are IDs (CUST-*, ORD-*), services, people, or \"user\". "
    "Capture agreed amounts, addresses, config fixes, rollback targets, "
    "usual rooms/times, and commitments (value = \"<place> on <day> at "
    "<time>\"). If a fact supersedes an earlier one, emit the latest value. "
    "Output ONLY JSON lines, nothing else.")


class LLMFacts(StructuredFacts):
    """C4 upgraded: LLM extraction instead of domain regexes.
    Update-on-write semantics are unchanged: a newer (entity, attribute)
    REPLACES the old value."""
    name = "C4_structured"

    def __init__(self, llm_fn) -> None:
        super().__init__()
        self.llm_fn = llm_fn
        self.meter = {"prompt_tokens": 0, "completion_tokens": 0,
                      "embedding_tokens": 0}

    def write(self, episode_id: str, transcript: str) -> None:
        text, pt, ct = self.llm_fn(EXTRACT_SYSTEM, transcript)
        self.meter["prompt_tokens"] += pt
        self.meter["completion_tokens"] += ct
        for line in text.splitlines():
            line = line.strip().strip("`")
            if not line.startswith("{"):
                continue
            try:
                fact = json.loads(line)
                entity = str(fact["entity"])
                attr = re.sub(r"\W+", "_", str(fact["attribute"])).lower()
                self.facts[(entity, attr)] = (str(fact["value"]), episode_id)
            except (json.JSONDecodeError, KeyError):
                continue  # malformed line: skip, never crash the run


class LLMHybrid(Hybrid):
    """C5 upgraded: LLMFacts + EmbeddingRAG."""
    name = "C5_hybrid"

    def __init__(self, llm_fn, embed_fn) -> None:
        super().__init__()
        self.struct = LLMFacts(llm_fn)
        self.rag = EmbeddingRAG(embed_fn)

    @property
    def meter(self):
        return {k: self.struct.meter[k] + self.rag.meter[k]
                for k in self.struct.meter}


def upgraded_conditions(llm_model: str, embed_model: str) -> dict:
    """Drop-in replacement for memory.CONDITIONS with the Phase B
    implementations bound to real models (LiteLLM strings)."""
    from .memory import NoMemory, FullReplay
    llm = default_llm_fn(llm_model)
    emb = default_embed_fn(embed_model)
    return {
        "C0": NoMemory,
        "C1": FullReplay,
        "C2": lambda: EmbeddingRAG(emb),
        "C3": lambda: LLMSummary(llm),
        "C4": lambda: LLMFacts(llm),
        "C5": lambda: LLMHybrid(llm, emb),
    }
