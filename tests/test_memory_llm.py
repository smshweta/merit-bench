"""Tests for the Phase B memory implementations (merit/memory_llm.py) with
injected fake model functions — no API calls, $0."""
import random

from merit.memory import corrupt_records
from merit.memory_llm import EmbeddingRAG, LLMFacts, LLMSummary, LLMHybrid


# ---------------- fakes ----------------

VOCAB = ["refund", "address", "rollback", "dinner", "room", "config"]


def fake_embed(texts):
    """Deterministic keyword one-hot embeddings; 1 token per text."""
    vecs = []
    for t in texts:
        low = t.lower()
        vecs.append([float(w in low) for w in VOCAB] + [1e-6])
    return vecs, len(texts)


def fake_llm_summary(system, user):
    lines = [l for l in user.splitlines() if "agreed" in l or "room" in l]
    return "\n".join(lines[:8]) or "(nothing durable)", 100, 20


def fake_llm_extract(system, user):
    out = []
    if "4287" in user:
        out.append('{"entity": "ORD-1", "attribute": "agreed refund cents", '
                   '"value": "4287"}')
    if "Room 47D" in user:
        out.append('{"entity": "user", "attribute": "usual_room", '
                   '"value": "Room 47D"}')
    out.append("not json, should be skipped")
    return "\n".join(out), 150, 30


# ---------------- EmbeddingRAG ----------------

def test_embedding_rag_ranks_by_cosine_and_meters():
    m = EmbeddingRAG(fake_embed, chunk_chars=10_000, top_k=1)
    m.write("e0", "the agreed refund was recorded")
    m.write("e1", "the meeting room was confirmed")
    got = m.read("process the refund now")
    assert "refund" in got and "room" not in got
    got = m.read("book the room")
    assert "room" in got
    assert m.meter["embedding_tokens"] > 0


def test_embedding_rag_embeds_corruption_stragglers():
    m = EmbeddingRAG(fake_embed, chunk_chars=10_000, top_k=2)
    m.write("e0", "refund note for ORD-123456 " * 20)
    n = corrupt_records(m, rate=1.0, rng=random.Random(0), mode="distractor")
    assert n > 0
    m.read("refund")  # must not crash on records added without vectors
    assert len(m._vectors) == len(m.records)


# ---------------- LLMSummary ----------------

def test_llm_summary_uses_model_and_meters():
    m = LLMSummary(fake_llm_summary)
    m.write("e0", "user said hello\nthe agreed amount of 4287 cents\nbye")
    assert "4287" in m.doc
    assert m.meter["prompt_tokens"] == 100
    corrupt_records(m, rate=1.0, rng=random.Random(0), mode="stale")
    assert m.doc_corrupted


# ---------------- LLMFacts ----------------

def test_llm_facts_extracts_updates_and_skips_garbage():
    m = LLMFacts(fake_llm_extract)
    m.write("e0", "colleague agreed 4287 cents for the damaged item")
    assert m.facts[("ORD-1", "agreed_refund_cents")][0] == "4287"
    # update-on-write: a later value for the same key replaces the old one
    m.facts[("ORD-1", "agreed_refund_cents")] = ("1111", "e0")
    m.write("e1", "re-negotiated: 4287 cents stands")
    assert m.facts[("ORD-1", "agreed_refund_cents")][0] == "4287"
    assert m.meter["completion_tokens"] > 0


def test_llm_facts_attribute_normalization():
    m = LLMFacts(fake_llm_extract)
    m.write("e0", "my usual room is Room 47D and 4287 cents")
    assert ("user", "usual_room") in m.facts
    # "agreed refund cents" -> snake_case
    assert ("ORD-1", "agreed_refund_cents") in m.facts


# ---------------- Hybrid + corruption compatibility ----------------

def test_llm_hybrid_combines_and_meters():
    m = LLMHybrid(fake_llm_extract, fake_embed)
    m.write("e0", "agreed 4287 cents refund for ORD-1")
    out = m.read("refund")
    assert "4287" in out
    assert m.meter["prompt_tokens"] > 0
    assert m.meter["embedding_tokens"] > 0
    assert corrupt_records(m, rate=1.0, rng=random.Random(0),
                           mode="stale") > 0