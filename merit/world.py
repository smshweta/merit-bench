"""MERIT Domain D1: customer-support world backed by SQLite.

Deterministic, seeded world state. All tool effects are writes to this DB,
and all success checkers are pure reads of it.
"""
from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE customers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    address TEXT NOT NULL
);
CREATE TABLE orders (
    id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    item TEXT NOT NULL,
    amount_cents INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'delivered',  -- delivered | refunded | shipped
    ship_address TEXT NOT NULL
);
CREATE TABLE refunds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL REFERENCES orders(id),
    amount_cents INTEGER NOT NULL
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE TABLE policies (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

DEFAULT_POLICIES = {
    "refund_window_days": "30",
    "refund_requires_delivered": "true",
    "max_refund_without_approval_cents": "10000",
}

FIRST = ["Aarav", "Bianca", "Chen", "Divya", "Elena", "Farid", "Grace", "Hiro",
         "Ines", "Jamal", "Kira", "Luis", "Mei", "Noah", "Priya", "Quinn"]
LAST = ["Anders", "Brown", "Costa", "Das", "Evans", "Fischer", "Garcia", "Huang",
        "Iyer", "Jones", "Kim", "Lopez", "Mishra", "Nguyen", "Okafor", "Patel"]
ITEMS = ["wireless mouse", "desk lamp", "yoga mat", "water bottle", "backpack",
         "phone case", "notebook set", "coffee grinder", "headphones", "blanket"]
STREETS = ["Maple St", "Oak Ave", "Cedar Ln", "Pine Rd", "Elm Dr", "Birch Way"]
CITIES = ["Springfield", "Riverton", "Lakeview", "Fairfield", "Georgetown"]


@dataclass
class World:
    """Owns the SQLite connection for one episode arc."""
    conn: sqlite3.Connection
    rng: random.Random

    @classmethod
    def create(cls, seed: int, db_path: str | Path = ":memory:") -> "World":
        conn = sqlite3.connect(db_path)
        conn.executescript(SCHEMA)
        for k, v in DEFAULT_POLICIES.items():
            conn.execute("INSERT INTO policies VALUES (?, ?)", (k, v))
        rng = random.Random(seed)
        world = cls(conn=conn, rng=rng)
        world._seed_entities(n_customers=4, orders_per_customer=3)
        conn.commit()
        return world

    # ---------- seeding ----------
    def _seed_entities(self, n_customers: int, orders_per_customer: int) -> None:
        for ci in range(n_customers):
            cid = f"CUST-{1000 + ci}"
            name = f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)}"
            address = (f"{self.rng.randint(10, 999)} {self.rng.choice(STREETS)}, "
                       f"{self.rng.choice(CITIES)}")
            self.conn.execute("INSERT INTO customers VALUES (?,?,?)",
                              (cid, name, address))
            for oi in range(orders_per_customer):
                oid = f"ORD-{self.rng.randint(100000, 999999)}"
                item = self.rng.choice(ITEMS)
                amount = self.rng.randrange(500, 20000, 100)
                self.conn.execute(
                    "INSERT INTO orders (id, customer_id, item, amount_cents, "
                    "status, ship_address) VALUES (?,?,?,?, 'delivered', ?)",
                    (oid, cid, item, amount, address))

    # ---------- snapshots for checkers ----------
    def snapshot(self) -> dict:
        out: dict = {}
        for table in ("customers", "orders", "refunds", "messages", "policies"):
            cur = self.conn.execute(f"SELECT * FROM {table}")
            cols = [d[0] for d in cur.description]
            out[table] = [dict(zip(cols, row)) for row in cur.fetchall()]
        return out

    def dump_json(self) -> str:
        return json.dumps(self.snapshot(), sort_keys=True)
