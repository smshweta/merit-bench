"""Tests for Phase 1 components: arc generator + leak check, simulated user,
distractor corruption, mock model, and the offline pilot pipeline."""
import json
import random

import pytest

from merit.arcs import Arc, EpisodeSpec, LeakError, TaskSpec, generate_arc, \
    generate_suite, leak_check
from merit.memory import (CONDITIONS, FullReplay, KeywordRAG, RollingSummary,
                          StructuredFacts, Hybrid, corrupt_records)
from merit.mockmodel import completion, find_address, find_agreed_cents
from merit.user_sim import SimulatedUser


# ---------------- arc generator ----------------

def test_arc_generation_is_deterministic():
    a1 = generate_arc("arcX", seed=3, n_episodes=5, dep_ratio=0.5)
    a2 = generate_arc("arcX", seed=3, n_episodes=5, dep_ratio=0.5)
    assert [e.task.task_id for e in a1.episodes] == \
           [e.task.task_id for e in a2.episodes]
    assert [e.task.user_messages for e in a1.episodes] == \
           [e.task.user_messages for e in a2.episodes]


def test_dep_ratio_controls_dependent_count():
    for ratio, expected in [(0.0, 0), (0.5, 2), (1.0, 4)]:
        arc = generate_arc("arcR", seed=1, n_episodes=5, dep_ratio=ratio)
        assert sum(e.task.dependent for e in arc.episodes) == expected


def test_plants_strictly_precede_probes():
    for arc in generate_suite(n_arcs=5, dep_ratio=0.75):
        for ep in arc.episodes:
            if ep.task.dependent:
                assert 0 <= ep.task.plant_episode < ep.index


def test_leak_check_catches_gold_in_probe_input():
    arc = Arc(arc_id="bad", seed=0)
    arc.episodes = [
        EpisodeSpec(0, TaskSpec("t0", False, "I_addr", ["set up"], "c", {})),
        EpisodeSpec(1, TaskSpec(
            "t1", True, "T_addr",
            ["my address is 12 Leak St, please confirm 12 Leak St"],
            "check_message_sent", {}, gold_fact_value="12 Leak St",
            plant_episode=0)),
    ]
    with pytest.raises(LeakError):
        leak_check(arc, world_json="{}")


def test_leak_check_catches_gold_in_world_state():
    arc = Arc(arc_id="bad2", seed=0)
    arc.episodes = [
        EpisodeSpec(0, TaskSpec("t0", False, "I_addr", ["hello"], "c", {})),
        EpisodeSpec(1, TaskSpec(
            "t1", True, "T_addr", ["confirm my address"],
            "check_message_sent", {}, gold_fact_value="99 Known Ave",
            plant_episode=0)),
    ]
    with pytest.raises(LeakError):
        leak_check(arc, world_json='{"customers": "99 Known Ave"}')


def test_generated_arcs_pass_leak_check():
    generate_suite(n_arcs=10, dep_ratio=0.5)  # raises LeakError on failure


# ---------------- simulated user ----------------

def test_scripted_user_is_verbatim_and_deterministic():
    u = SimulatedUser(script=["a", "b"], persona_idx=1)
    assert u.turns() == ["a", "b"] == u.turns()


# ---------------- corruption ----------------

def _rng():
    return random.Random(0)


def test_distractor_adds_entity_matching_records():
    m = KeywordRAG()
    m.write("e0", "[tool refund] {'order_id': 'ORD-123456'} -> ok " * 30)
    before = len(m.records)
    n = corrupt_records(m, rate=1.0, rng=_rng(), mode="distractor")
    assert n > 0 and len(m.records) == before + n
    added = m.records[before:]
    assert all(r.corrupted for r in added)
    assert any("ORD-123456" in r.text for r in added)


def test_stale_corruption_mutates_structured_facts():
    m = StructuredFacts()
    m.write("e0", 'agreed amount of 4200 cents for order ORD-111111')
    assert m.facts[("ORD-111111", "agreed_refund_cents")][0] == "4200"
    corrupt_records(m, rate=1.0, rng=_rng(), mode="stale")
    assert m.facts[("ORD-111111", "agreed_refund_cents")][0] != "4200"
    assert ("ORD-111111", "agreed_refund_cents") in m.corrupted_keys


def test_corruption_reaches_summary_and_hybrid():
    s = RollingSummary()
    s.write("e0", "value 12345")
    corrupt_records(s, rate=1.0, rng=_rng(), mode="stale")
    assert s.doc_corrupted and "12345" not in s.doc

    h = Hybrid()
    h.write("e0", 'agreed amount of 4200 cents for order ORD-222222')
    assert corrupt_records(h, rate=1.0, rng=_rng(), mode="stale") > 0


# ---------------- mock model ----------------

def test_mock_memory_readers():
    c4 = "ORD-123456.agreed_refund_cents = 4200  (ep e0)"
    assert find_agreed_cents(c4, "ORD-123456") == "4200"
    raw = "agreed amount of 3300 cents for order ORD-777777"
    assert find_agreed_cents(raw, "ORD-777777") == "3300"
    addr = "CUST-1001.address = 55 Oak Ave Apt 2, Riverton  (ep e1)"
    assert find_address(addr, "CUST-1001") == "55 Oak Ave Apt 2, Riverton"
    assert find_address("", "CUST-1001") is None


def test_mock_completion_calls_tools_and_finishes():
    msgs = [{"role": "system", "content": "no notes"},
            {"role": "user", "content":
             "I'd like a full refund on order ORD-123456, please."}]
    r = completion("mock", msgs)
    tc = r.choices[0].message.tool_calls[0]
    assert tc.function.name == "get_order"
    assert json.loads(tc.function.arguments) == {"order_id": "ORD-123456"}
    assert r.usage.prompt_tokens > 0


# ---------------- offline pilot pipeline ----------------

def test_mock_pilot_end_to_end(tmp_path):
    import argparse
    import sys
    sys.path.insert(0, "scripts")
    from run_pilot import run

    args = argparse.Namespace(
        model="mock", arcs=4, episodes=5, seeds=1, dep_ratio=0.5,
        conditions="C0,C1,C4", corrupt=True,
        corrupt_modes="stale", user_mode="scripted", out=str(tmp_path))
    results = run(args)
    rows = [json.loads(l) for l in results.read_text().splitlines()]
    assert rows, "pilot wrote no rows"

    def tsr(cond, dep, corrupt="none"):
        sel = [r for r in rows if r["condition"] == cond
               and r["dependent"] == dep and r["corrupt_mode"] == corrupt]
        return sum(r["success"] for r in sel) / len(sel)

    # pipeline sanity: memory (C1) must beat no-memory (C0) on dependent
    # tasks, and corruption must not IMPROVE dependent-task success
    assert tsr("C0", dep=True) < tsr("C1", dep=True)
    assert tsr("C1", dep=True, corrupt="stale") <= tsr("C1", dep=True)
    # independent tasks solvable without memory
    assert tsr("C0", dep=False) == 1.0
