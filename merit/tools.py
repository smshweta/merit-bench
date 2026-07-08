"""Tools exposed to the agent for Domain D1. Every tool is a thin, audited
wrapper over the World's SQLite state. Tool *calls* (name + args) are logged
verbatim by the runner — that log is what the MUR value-tracer reads.
"""
from __future__ import annotations

import json
from .world import World


def get_order(world: World, order_id: str) -> str:
    cur = world.conn.execute(
        "SELECT id, customer_id, item, amount_cents, status, ship_address "
        "FROM orders WHERE id = ?", (order_id,))
    row = cur.fetchone()
    if not row:
        return json.dumps({"error": f"order {order_id} not found"})
    keys = ["id", "customer_id", "item", "amount_cents", "status", "ship_address"]
    return json.dumps(dict(zip(keys, row)))


def refund(world: World, order_id: str, amount_cents: int) -> str:
    cur = world.conn.execute("SELECT status, amount_cents FROM orders WHERE id = ?",
                             (order_id,))
    row = cur.fetchone()
    if not row:
        return json.dumps({"error": f"order {order_id} not found"})
    status, order_amount = row
    if status == "refunded":
        return json.dumps({"error": "order already refunded"})
    if amount_cents <= 0 or amount_cents > order_amount:
        return json.dumps({"error": "invalid refund amount"})
    world.conn.execute("INSERT INTO refunds (order_id, amount_cents) VALUES (?,?)",
                       (order_id, amount_cents))
    world.conn.execute("UPDATE orders SET status='refunded' WHERE id=?", (order_id,))
    world.conn.commit()
    return json.dumps({"ok": True, "order_id": order_id,
                       "refunded_cents": amount_cents})


def update_address(world: World, customer_id: str, new_address: str) -> str:
    cur = world.conn.execute("SELECT id FROM customers WHERE id=?", (customer_id,))
    if not cur.fetchone():
        return json.dumps({"error": f"customer {customer_id} not found"})
    world.conn.execute("UPDATE customers SET address=? WHERE id=?",
                       (new_address, customer_id))
    world.conn.commit()
    return json.dumps({"ok": True, "customer_id": customer_id,
                       "address": new_address})


def get_policy(world: World, key: str) -> str:
    cur = world.conn.execute("SELECT value FROM policies WHERE key=?", (key,))
    row = cur.fetchone()
    if not row:
        return json.dumps({"error": f"no policy named {key}",
                           "available": [r[0] for r in world.conn.execute(
                               "SELECT key FROM policies")]})
    return json.dumps({"key": key, "value": row[0]})


def send_message(world: World, customer_id: str, body: str) -> str:
    world.conn.execute("INSERT INTO messages (customer_id, body) VALUES (?,?)",
                       (customer_id, body))
    world.conn.commit()
    return json.dumps({"ok": True})


TOOL_FUNCS = {
    "get_order": get_order,
    "refund": refund,
    "update_address": update_address,
    "get_policy": get_policy,
    "send_message": send_message,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_order",
        "description": "Look up an order by its ID (format ORD-XXXXXX).",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "refund",
        "description": "Issue a refund for an order, in cents.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"},
            "amount_cents": {"type": "integer"}},
            "required": ["order_id", "amount_cents"]}}},
    {"type": "function", "function": {
        "name": "update_address",
        "description": "Update a customer's shipping address.",
        "parameters": {"type": "object", "properties": {
            "customer_id": {"type": "string"},
            "new_address": {"type": "string"}},
            "required": ["customer_id", "new_address"]}}},
    {"type": "function", "function": {
        "name": "get_policy",
        "description": "Read a store policy value by key.",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string"}}, "required": ["key"]}}},
    {"type": "function", "function": {
        "name": "send_message",
        "description": "Send a message to a customer.",
        "parameters": {"type": "object", "properties": {
            "customer_id": {"type": "string"},
            "body": {"type": "string"}},
            "required": ["customer_id", "body"]}}},
]
