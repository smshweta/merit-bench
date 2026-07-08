import json
import random

from merit.world import World
from merit.tools import get_order, refund, update_address, get_policy, send_message
from merit.memory import (NoMemory, FullReplay, KeywordRAG, StructuredFacts,
                          Hybrid, corrupt_records)
from merit.metrics import (check_refund_issued, check_address_updated,
                           check_message_sent, memory_utilized)


def first_order(world):
    row = world.conn.execute(
        "SELECT id, customer_id, amount_cents FROM orders").fetchone()
    return row  # (order_id, customer_id, amount_cents)


def test_world_is_deterministic():
    assert World.create(seed=7).dump_json() == World.create(seed=7).dump_json()
    assert World.create(seed=7).dump_json() != World.create(seed=8).dump_json()


def test_refund_flow_and_checker():
    w = World.create(seed=1)
    oid, _, amount = first_order(w)
    out = json.loads(refund(w, oid, amount))
    assert out["ok"]
    assert check_refund_issued(w.snapshot(), oid, amount)
    # double refund must fail
    assert "error" in json.loads(refund(w, oid, amount))


def test_refund_rejects_bad_amounts():
    w = World.create(seed=1)
    oid, _, amount = first_order(w)
    assert "error" in json.loads(refund(w, oid, amount + 1))
    assert "error" in json.loads(refund(w, oid, 0))


def test_address_update_and_checker():
    w = World.create(seed=2)
    _, cid, _ = first_order(w)
    new_addr = "42 Test Blvd, Sampletown"
    assert json.loads(update_address(w, cid, new_addr))["ok"]
    assert check_address_updated(w.snapshot(), cid, new_addr)
    assert not check_address_updated(w.snapshot(), cid, "wrong")


def test_message_and_policy():
    w = World.create(seed=3)
    _, cid, _ = first_order(w)
    send_message(w, cid, "your refund of $12.00 was processed")
    assert check_message_sent(w.snapshot(), cid, "refund")
    assert json.loads(get_policy(w, "refund_window_days"))["value"] == "30"
    assert "error" in json.loads(get_policy(w, "nope"))


def test_memory_conditions_basic():
    transcript = ('[tool update_address] -> {"ok": true, "customer_id": '
                  '"CUST-1000", "address": "9 New St, Lakeview"}')
    assert NoMemory().read("anything") == ""

    fr = FullReplay(); fr.write("ep1", transcript)
    assert "CUST-1000" in fr.read("x")

    rag = KeywordRAG(); rag.write("ep1", transcript)
    assert "CUST-1000" in rag.read("what is the address of CUST-1000")

    sf = StructuredFacts(); sf.write("ep1", transcript)
    assert "9 New St, Lakeview" in sf.read("")
    # update-on-write: newer value replaces older
    sf.write("ep2", transcript.replace("9 New St, Lakeview", "1 Newer Ave"))
    read = sf.read("")
    assert "1 Newer Ave" in read and "9 New St, Lakeview" not in read

    hy = Hybrid(); hy.write("ep1", transcript)
    assert "CUST-1000" in hy.read("CUST-1000 address")


def test_corruption_injector_marks_ground_truth():
    fr = FullReplay()
    fr.write("ep1", "amount is 1234")
    n = corrupt_records(fr, rate=1.0, rng=random.Random(0), mode="stale")
    assert n == 1
    assert fr.records[0].corrupted
    assert "1234" not in fr.records[0].text  # digits mutated


def test_mur_tracer():
    calls = [{"name": "refund", "args": {"order_id": "ORD-111", "amount_cents": 500}}]
    assert memory_utilized(calls, "ORD-111")
    assert not memory_utilized(calls, "ORD-999")
