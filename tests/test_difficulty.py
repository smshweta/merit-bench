"""Tests for the difficulty ladder (easy / medium / hard) across all three
domains: generation determinism, leak checks over ALL gold values, hard-tier
plant<update<probe ordering, and $0 mock pilots per (domain, difficulty)."""
import argparse
import json
import sys

import pytest

from merit import arcs as d1, d2, d3
from merit.metrics import memory_utilized

DOMAIN_MODULES = {"d1": d1, "d2": d2, "d3": d3}


@pytest.mark.parametrize("dom", ["d1", "d2", "d3"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generation_deterministic_and_leak_free(dom, difficulty):
    mod = DOMAIN_MODULES[dom]
    s1 = mod.generate_suite(n_arcs=4, dep_ratio=0.75, difficulty=difficulty)
    s2 = mod.generate_suite(n_arcs=4, dep_ratio=0.75, difficulty=difficulty)
    ids1 = [e.task.task_id for a in s1 for e in a.episodes]
    ids2 = [e.task.task_id for a in s2 for e in a.episodes]
    assert ids1 == ids2  # leak check runs inside generate_arc


@pytest.mark.parametrize("dom", ["d1", "d2", "d3"])
def test_medium_probes_have_multiple_golds(dom):
    mod = DOMAIN_MODULES[dom]
    probes = [e.task for a in mod.generate_suite(n_arcs=5, dep_ratio=0.75,
                                                 difficulty="medium")
              for e in a.episodes if e.task.dependent]
    assert probes
    for t in probes:
        assert len(t.golds()) == 2
        assert all(g for g in t.golds())


@pytest.mark.parametrize("dom", ["d1", "d2", "d3"])
def test_hard_probes_need_the_updated_value(dom):
    mod = DOMAIN_MODULES[dom]
    suites = mod.generate_suite(n_arcs=5, dep_ratio=0.75, difficulty="hard")
    probes = [(a, e) for a in suites for e in a.episodes if e.task.dependent]
    assert probes
    for arc, ep in probes:
        t = ep.task
        # the UPDATED gold value is delivered in some episode strictly
        # between plant and probe (task ids can be hidden by _merge, so
        # detect by message content)
        update_eps = [e for e in arc.episodes
                      if t.plant_episode < e.index < ep.index
                      and any(t.gold_fact_value in m
                              for m in e.task.user_messages)]
        assert update_eps, f"no update episode for {t.task_id}"
        # the gold is the UPDATED value: it must NOT be what the plant said
        plant_msgs = " ".join(
            m for e in arc.episodes if e.index == t.plant_episode
            for m in e.task.user_messages)
        assert t.gold_fact_value not in plant_msgs


def test_memory_utilized_requires_all_golds():
    calls = [{"name": "refund", "args": {"amount_cents": 4287}},
             {"name": "send_message", "args": {"body": "12 Oak St"}}]
    assert memory_utilized(calls, ["4287", "12 Oak St"])
    assert not memory_utilized(calls, ["4287", "99 Elm Ave"])
    assert memory_utilized(calls, "4287")  # str form still works


@pytest.mark.parametrize("dom", ["d1", "d2", "d3"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_mock_pilot_grid(tmp_path, dom, difficulty):
    sys.path.insert(0, "scripts")
    from run_pilot import run

    args = argparse.Namespace(
        model="mock", api_base=None, domain=dom, difficulty=difficulty,
        arcs=4, episodes=5, seeds=1, dep_ratio=0.5,
        conditions="C0,C1,C4", corrupt=False,
        corrupt_modes="stale", user_mode="scripted",
        out=str(tmp_path / f"{dom}-{difficulty}"))
    rows = [json.loads(l)
            for l in run(args).read_text().splitlines()]

    def tsr(cond, dep):
        sel = [r for r in rows if r["condition"] == cond
               and r["dependent"] == dep]
        return sum(r["success"] for r in sel) / len(sel) if sel else None

    # no-memory agents must NEVER pass dependent probes at any difficulty
    assert tsr("C0", dep=True) == 0.0
    # memory must help; C4 (update-on-write) must ace the hard tier
    assert tsr("C1", dep=True) > 0.0
    assert tsr("C4", dep=True) > 0.0
    if difficulty == "hard":
        assert tsr("C4", dep=True) == 1.0
    # independent tasks are solvable without memory
    assert tsr("C0", dep=False) == 1.0
    # delta scoring invariant
    assert not any(r["success"] and r["pre_satisfied"] for r in rows)