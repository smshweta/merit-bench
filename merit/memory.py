"""Memory conditions C0–C5 behind a single interface, plus the corruption
injector. All conditions receive identical episode traces; only what they
store/retrieve differs. Token costs of memory reads/writes are metered by
the runner via the returned strings' lengths and any LLM calls made here.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class MemoryRecord:
    episode_id: str
    text: str
    corrupted: bool = False  # ground-truth flag, never shown to the agent


class MemoryBase(ABC):
    """memory.write(trace) after each episode; memory.read(ctx) before each turn."""

    name: str = "base"

    def __init__(self) -> None:
        self.records: list[MemoryRecord] = []

    @abstractmethod
    def write(self, episode_id: str, transcript: str) -> None: ...

    @abstractmethod
    def read(self, current_context: str, budget_chars: int = 4000) -> str: ...


class NoMemory(MemoryBase):
    name = "C0_none"

    def write(self, episode_id: str, transcript: str) -> None:
        pass

    def read(self, current_context: str, budget_chars: int = 4000) -> str:
        return ""


class FullReplay(MemoryBase):
    """C1: prepend all prior transcripts (upper bound on info AND cost)."""
    name = "C1_full_replay"

    def write(self, episode_id: str, transcript: str) -> None:
        self.records.append(MemoryRecord(episode_id, transcript))

    def read(self, current_context: str, budget_chars: int = 60_000) -> str:
        blob = "\n\n---\n\n".join(r.text for r in self.records)
        return blob[-budget_chars:]  # keep most recent if over budget; document this


class KeywordRAG(MemoryBase):
    """C2: retrieval memory. Starter version uses BM25-ish keyword overlap over
    chunks; swap `score` for embedding cosine similarity in Phase 1 step 5
    (interface stays identical, so results pipelines don't change)."""
    name = "C2_rag"

    def __init__(self, chunk_chars: int = 600, top_k: int = 4) -> None:
        super().__init__()
        self.chunk_chars = chunk_chars
        self.top_k = top_k

    def write(self, episode_id: str, transcript: str) -> None:
        for i in range(0, len(transcript), self.chunk_chars):
            self.records.append(
                MemoryRecord(episode_id, transcript[i:i + self.chunk_chars]))

    @staticmethod
    def _tokens(s: str) -> set[str]:
        return set(re.findall(r"[A-Za-z0-9\-]+", s.lower()))

    def read(self, current_context: str, budget_chars: int = 4000) -> str:
        q = self._tokens(current_context)
        scored = sorted(self.records,
                        key=lambda r: -len(q & self._tokens(r.text)))
        out, used = [], 0
        for r in scored[: self.top_k]:
            if used + len(r.text) > budget_chars:
                break
            out.append(r.text)
            used += len(r.text)
        return "\n---\n".join(out)


class RollingSummary(MemoryBase):
    """C3: one bounded summary document. Starter version does extractive
    truncation; replace `summarize` with an LLM call in Phase 1 (the runner
    already meters LLM usage inside memory objects via the shared client)."""
    name = "C3_summary"

    def __init__(self, max_chars: int = 4000) -> None:
        super().__init__()
        self.doc = ""
        self.max_chars = max_chars

    def summarize(self, transcript: str) -> str:
        return transcript[:800]  # TODO(Phase 1): LLM summarization

    def write(self, episode_id: str, transcript: str) -> None:
        self.doc = (self.doc + f"\n[Episode {episode_id}] "
                    + self.summarize(transcript))[-self.max_chars:]

    def read(self, current_context: str, budget_chars: int = 4000) -> str:
        return self.doc[-budget_chars:]


class StructuredFacts(MemoryBase):
    """C4: (entity, attribute, value, episode) tuples with update-on-write —
    a newer value for the same (entity, attribute) REPLACES the old one.
    Starter extraction is regex over tool results; upgrade to LLM extraction
    in Phase 1. Update-on-write is the mechanism H3 tests."""
    name = "C4_structured"

    def __init__(self) -> None:
        super().__init__()
        self.facts: dict[tuple[str, str], tuple[str, str]] = {}

    _PATTERNS = [
        (re.compile(r'"customer_id":\s*"([^"]+)".*?"address":\s*"([^"]+)"'),
         "address"),
        (re.compile(r'"order_id":\s*"([^"]+)".*?"refunded_cents":\s*(\d+)'),
         "refunded_cents"),
    ]

    def write(self, episode_id: str, transcript: str) -> None:
        for pattern, attr in self._PATTERNS:
            for m in pattern.finditer(transcript):
                entity, value = m.group(1), m.group(2)
                self.facts[(entity, attr)] = (value, episode_id)

    def read(self, current_context: str, budget_chars: int = 4000) -> str:
        lines = [f"{e}.{a} = {v}  (ep {ep})"
                 for (e, a), (v, ep) in self.facts.items()]
        return "\n".join(lines)[:budget_chars]


class Hybrid(MemoryBase):
    """C5: structured facts + RAG fallback."""
    name = "C5_hybrid"

    def __init__(self) -> None:
        super().__init__()
        self.struct = StructuredFacts()
        self.rag = KeywordRAG()

    def write(self, episode_id: str, transcript: str) -> None:
        self.struct.write(episode_id, transcript)
        self.rag.write(episode_id, transcript)

    def read(self, current_context: str, budget_chars: int = 4000) -> str:
        s = self.struct.read(current_context, budget_chars // 2)
        r = self.rag.read(current_context, budget_chars // 2)
        return f"[facts]\n{s}\n[related transcripts]\n{r}"


# ---------------- corruption injector ----------------

def corrupt_records(memory: MemoryBase, rate: float, rng,
                    mode: str = "stale") -> int:
    """Corrupt a fraction of stored records IN PLACE. Returns count corrupted.
    stale: mutate digits (old order amounts / addresses look plausible but wrong)
    contradiction: append a conflicting duplicate
    distractor: inject entity-matching but irrelevant text
    """
    n = 0
    for rec in memory.records:
        if rng.random() < rate:
            if mode == "stale":
                rec.text = re.sub(r"\d", lambda m: str((int(m.group()) + 3) % 10),
                                  rec.text)
            elif mode == "contradiction":
                rec.text += "\n[NOTE] Correction: the previous values were updated."
            rec.corrupted = True
            n += 1
    return n


CONDITIONS = {
    "C0": NoMemory,
    "C1": FullReplay,
    "C2": KeywordRAG,
    "C3": RollingSummary,
    "C4": StructuredFacts,
    "C5": Hybrid,
}
