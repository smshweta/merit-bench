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
        self.doc_corrupted = False  # ground-truth flag, never shown to agent

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
        self.corrupted_keys: set[tuple[str, str]] = set()  # ground truth only

    # (pattern, attribute, entity_group, value_group)
    # entity_group may be a literal string (e.g. "user") when the fact's
    # owner has no ID in the text (D3 personal-assistant facts).
    _PATTERNS = [
        # D1 customer support
        (re.compile(r'"customer_id":\s*"([^"]+)".*?"address":\s*"([^"]+)"'),
         "address", 1, 2),
        (re.compile(r'"order_id":\s*"([^"]+)".*?"refunded_cents":\s*(\d+)'),
         "refunded_cents", 1, 2),
        (re.compile(r'agreed amount of (\d+) cents for order (ORD-\d+)'),
         "agreed_refund_cents", 2, 1),
        # D2 IT ops
        (re.compile(r'planned fix: (\w+=\d+) for ([a-z-]+-api)'),
         "planned_fix", 2, 1),
        (re.compile(r'rollback target (v[\d.]+) for ([a-z-]+-api)'),
         "rollback_target", 2, 1),
        (re.compile(r'"service":\s*"([^"]+)".*?"version":\s*"(v[\d.]+)"'),
         "version", 1, 2),
        # D3 personal assistant
        (re.compile(r'usual room is (Room \d+[A-Z])'),
         "usual_room", "user", 1),
        (re.compile(r'dinner with (\w+) at (.+? on \w+ at \d\d:\d\d)'),
         "dinner", 1, 2),
    ]

    def write(self, episode_id: str, transcript: str) -> None:
        for pattern, attr, eg, vg in self._PATTERNS:
            for m in pattern.finditer(transcript):
                entity = eg if isinstance(eg, str) else m.group(eg)
                value = m.group(vg)
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

_ENTITY_RE = re.compile(r"\b(CUST-\d+|ORD-\d+|[a-z]+-api|Room \d+[A-Z])\b")

_DISTRACTOR_TEMPLATES = [
    "{e} asked whether gift wrapping is available for future orders.",
    "{e} was sent the seasonal newsletter and clicked one link.",
    "{e} inquired about loyalty points; none were applied.",
    "{e} briefly viewed the FAQ page about shipping carriers.",
]


def _mutate_digits(s: str) -> str:
    return re.sub(r"\d", lambda m: str((int(m.group()) + 3) % 10), s)


def corrupt_records(memory: MemoryBase, rate: float, rng,
                    mode: str = "stale") -> int:
    """Corrupt a fraction of stored memory IN PLACE. Returns count corrupted.
    Ground-truth corruption flags are kept internally, never shown to agents.

    stale:         mutate digits — superseded-looking but wrong values
    contradiction: append a conflicting duplicate record
    distractor:    inject entity-matching but task-irrelevant records
    Works on every condition's store: records (C1/C2), the summary doc (C3),
    and the fact table (C4); C5 delegates to both sub-stores.
    """
    if isinstance(memory, Hybrid):
        return (corrupt_records(memory.struct, rate, rng, mode)
                + corrupt_records(memory.rag, rate, rng, mode))

    n = 0
    # --- structured fact table (C4) ---
    if isinstance(memory, StructuredFacts):
        for key in list(memory.facts):
            if rng.random() < rate:
                value, ep = memory.facts[key]
                if mode == "stale":
                    memory.facts[key] = (_mutate_digits(value), ep)
                elif mode == "contradiction":
                    entity, attr = key
                    memory.facts[(entity, attr + "_prior")] = (
                        _mutate_digits(value), ep)
                elif mode == "distractor":
                    entity, _ = key
                    memory.facts[(entity, f"note{rng.randint(1, 99)}")] = (
                        rng.choice(_DISTRACTOR_TEMPLATES).format(e=entity), ep)
                memory.corrupted_keys.add(key)
                n += 1
        return n

    # --- rolling summary doc (C3) ---
    if isinstance(memory, RollingSummary):
        if memory.doc and rng.random() < rate:
            if mode == "stale":
                memory.doc = _mutate_digits(memory.doc)
            elif mode == "contradiction":
                memory.doc += ("\n[NOTE] Correction: some previous values "
                               "were updated; treat older values as current.")
            elif mode == "distractor":
                ents = _ENTITY_RE.findall(memory.doc) or ["the customer"]
                memory.doc += "\n" + rng.choice(_DISTRACTOR_TEMPLATES).format(
                    e=rng.choice(ents))
            memory.doc_corrupted = True
            n += 1
        return n

    # --- record stores (C1/C2) ---
    new_records: list[MemoryRecord] = []
    for rec in memory.records:
        if rng.random() < rate:
            if mode == "stale":
                rec.text = _mutate_digits(rec.text)
                rec.corrupted = True
            elif mode == "contradiction":
                rec.text += ("\n[NOTE] Correction: the previous values were "
                             "updated.")
                rec.corrupted = True
            elif mode == "distractor":
                ents = _ENTITY_RE.findall(rec.text) or ["the customer"]
                new_records.append(MemoryRecord(
                    rec.episode_id,
                    rng.choice(_DISTRACTOR_TEMPLATES).format(e=rng.choice(ents)),
                    corrupted=True))
            n += 1
    memory.records.extend(new_records)
    return n


CONDITIONS = {
    "C0": NoMemory,
    "C1": FullReplay,
    "C2": KeywordRAG,
    "C3": RollingSummary,
    "C4": StructuredFacts,
    "C5": Hybrid,
}
