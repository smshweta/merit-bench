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


# ---------- D2 (IT ops) checkers ----------

def check_config_set(snapshot: dict, service: str, key: str,
                     expected_value: str) -> bool:
    return any(c["service"] == service and c["key"] == key
               and c["value"] == str(expected_value)
               for c in snapshot["configs"])


def check_version_deployed(snapshot: dict, service: str,
                           expected_version: str) -> bool:
    return any(s["name"] == service and s["version"] == expected_version
               for s in snapshot["services"])


def check_ticket_updated(snapshot: dict, service: str,
                         must_contain: str) -> bool:
    return any(t["service"] == service and must_contain in t["body"]
               for t in snapshot["tickets"])


# ---------- D3 (personal assistant) checkers ----------

def check_event_created(snapshot: dict, day: str, time: str,
                        expected_location: str) -> bool:
    return any(e["day"] == day and e["time"] == time
               and e["location"] == expected_location
               for e in snapshot["events"])


def check_email_sent(snapshot: dict, must_contain: str) -> bool:
    return any(must_contain in e["body"] for e in snapshot["emails"])


def check_preference_set(snapshot: dict, key: str,
                         expected_value: str) -> bool:
    return any(p["key"] == key and p["value"] == expected_value
               for p in snapshot["preferences"])


# ---------- MUR value-tracer ----------

def memory_utilized(tool_calls: list[dict], gold_fact_value: str) -> bool:
    """Did the gold fact's value appear in any executed tool call's arguments?
    Conservative string containment; validated by human audit in Phase 4."""
    for tc in tool_calls:
        if gold_fact_value in str(tc.get("args", {})):
            return True
    return False
