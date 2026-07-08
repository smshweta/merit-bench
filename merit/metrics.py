"""Programmatic success checkers (pure functions of world snapshot) and the
MUR value-tracer (pure function of the tool-call log). Both are unit-tested;
the tracer additionally gets a 100-episode human audit in Phase 4.
"""
from __future__ import annotations


# ---------- success checkers ----------

def check_refund_issued(snapshot: dict, order_id: str,
                        expected_cents: int) -> bool:
    return any(r["order_id"] == order_id and r["amount_cents"] == expected_cents
               for r in snapshot["refunds"])


def check_address_updated(snapshot: dict, customer_id: str,
                          expected_address: str) -> bool:
    return any(c["id"] == customer_id and c["address"] == expected_address
               for c in snapshot["customers"])


def check_message_sent(snapshot: dict, customer_id: str,
                       must_contain: str) -> bool:
    return any(m["customer_id"] == customer_id and must_contain in m["body"]
               for m in snapshot["messages"])


# ---------- MUR value-tracer ----------

def memory_utilized(tool_calls: list[dict], gold_fact_value: str) -> bool:
    """Did the gold fact's value appear in any executed tool call's arguments?
    Conservative string containment; validated by human audit in Phase 4."""
    for tc in tool_calls:
        if gold_fact_value in str(tc.get("args", {})):
            return True
    return False
