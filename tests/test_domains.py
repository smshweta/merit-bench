"""Tests for domains D2 (IT ops) and D3 (personal assistant): world
determinism, arc generation + leak check, checkers, and C4 fact extraction
for the new domains' fact shapes."""
from merit import d2, d3
from merit import metrics as M
from merit.memory import StructuredFacts


# ---------------- D2 ----------------

def test_d2_world_is_deterministic():
    w1, w2 = d2.D2World.create(seed=7), d2.D2World.create(seed=7)
    assert w1.dump_json() == w2.dump_json()


def test_d2_arc_generation_and_leak_check():
    for arc in d2.generate_suite(n_arcs=5, dep_ratio=0.75):
        for ep in arc.episodes:
            if ep.task.dependent:
                assert 0 <= ep.task.plant_episode < ep.index
                assert ep.task.gold_fact_value


def test_d2_tools_and_checkers():
    w = d2.D2World.create(seed=1)
    d2.set_config(w, "auth-api", "pool_size", "4711")
    assert M.check_config_set(w.snapshot(), "auth-api", "pool_size", "4711")
    d2.deploy(w, "auth-api", "v9.9.9")
    assert M.check_version_deployed(w.snapshot(), "auth-api", "v9.9.9")
    d2.update_ticket(w, "auth-api", "rollback target v9.9.9 noted")
    assert M.check_ticket_updated(w.snapshot(), "auth-api", "v9.9.9")
    assert not M.check_config_set(w.snapshot(), "auth-api", "pool_size", "1")


def test_d2_bad_tool_args_return_error_json():
    w = d2.D2World.create(seed=1)
    assert "error" in d2.get_config(w, "auth-api", "no_such_key")
    assert "error" in d2.deploy(w, "no-such-svc", "v1.0.0")


# ---------------- D3 ----------------

def test_d3_world_is_deterministic():
    w1, w2 = d3.D3World.create(seed=7), d3.D3World.create(seed=7)
    assert w1.dump_json() == w2.dump_json()


def test_d3_arc_generation_and_leak_check():
    for arc in d3.generate_suite(n_arcs=5, dep_ratio=0.75):
        for ep in arc.episodes:
            if ep.task.dependent:
                assert 0 <= ep.task.plant_episode < ep.index
                assert ep.task.gold_fact_value


def test_d3_tools_and_checkers():
    w = d3.D3World.create(seed=1)
    d3.create_event(w, "dinner", "Thursday", "19:30", "Cafe Vesper")
    assert M.check_event_created(w.snapshot(), "Thursday", "19:30",
                                 "Cafe Vesper")
    d3.send_email(w, "me", "note", "your usual room is Room 47D")
    assert M.check_email_sent(w.snapshot(), "Room 47D")
    d3.set_preference(w, "coffee", "cortado")
    assert M.check_preference_set(w.snapshot(), "coffee", "cortado")


# ---------------- C4 extraction for D2/D3 fact shapes ----------------

def test_c4_extracts_d2_facts():
    m = StructuredFacts()
    m.write("e0", "update the ticket confirming the planned fix: "
                  "pool_size=4287 for payments-api.")
    assert m.facts[("payments-api", "planned_fix")][0] == "pool_size=4287"
    m.write("e1", "noting the rollback target v7.3.9 for search-api")
    assert m.facts[("search-api", "rollback_target")][0] == "v7.3.9"


def test_c4_extracts_d3_facts():
    m = StructuredFacts()
    m.write("e0", "send me an email confirming my usual room is Room 47D.")
    assert m.facts[("user", "usual_room")][0] == "Room 47D"
    m.write("e1", "confirming dinner with Sam at Cafe Vesper on Thursday "
                  "at 19:30.")
    assert m.facts[("Sam", "dinner")][0] == "Cafe Vesper on Thursday at 19:30"


