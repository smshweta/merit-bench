"""MERIT Domain D2: IT operations. Self-contained — world, tools, checkers,
and arc generator in one module, mirroring D1's plant/probe design.

Dependent task types:
  T_cfg  plant: on-call reports a diagnosed fix "set {key} to {value}" but a
         change freeze forbids applying it; the agent only confirms it on the
         ticket. probe: the incident recurs and the user asks to apply the
         agreed fix. The (key, value) pair exists only in memory — get_config
         returns the OLD value, and the fix value appears nowhere in the world.
  T_ver  plant: postmortem names a last-known-good version to roll back to,
         pending approval; agent confirms the target on the ticket. probe:
         approval arrives, "roll back to the version we identified". The gold
         version is absent from the deploy history, so a memoryless agent's
         plausible guess (previous version in history) is scored as failure.
"""
from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass

from .arcs import Arc, EpisodeSpec, TaskSpec, leak_check

SCHEMA = """
CREATE TABLE services (
    name TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'running',
    version TEXT NOT NULL
);
CREATE TABLE deployments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    service TEXT NOT NULL REFERENCES services(name),
    version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'  -- active | superseded | rolled_back
);
CREATE TABLE configs (
    service TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (service, key)
);
CREATE TABLE logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    service TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE TABLE tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    service TEXT NOT NULL,
    body TEXT NOT NULL
);
"""

SERVICES = ["payments-api", "search-api", "auth-api",
            "checkout-api", "billing-api", "metrics-api"]
CONFIG_KEYS = ["timeout_ms", "retries", "pool_size"]
LOG_MESSAGES = ["connection pool exhausted", "upstream timeout",
                "healthcheck passed", "cache miss ratio elevated",
                "GC pause exceeded budget"]


@dataclass
class D2World:
    """Owns the SQLite connection for one episode arc."""
    conn: sqlite3.Connection
    rng: random.Random

    @classmethod
    def create(cls, seed: int) -> "D2World":
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA)
        rng = random.Random(seed)
        for svc in SERVICES:
            version = f"v{rng.randint(1, 4)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}"
            conn.execute("INSERT INTO services VALUES (?, 'running', ?)",
                         (svc, version))
            conn.execute("INSERT INTO deployments (service, version, status) "
                         "VALUES (?, ?, 'active')", (svc, version))
            conn.execute("INSERT INTO deployments (service, version, status) "
                         "VALUES (?, ?, 'superseded')",
                         (svc, f"v{rng.randint(1, 4)}.{rng.randint(0, 9)}"
                               f".{rng.randint(0, 9)}"))
            conn.execute("INSERT INTO configs VALUES (?, 'timeout_ms', ?)",
                         (svc, str(rng.randrange(1000, 9000, 500))))
            conn.execute("INSERT INTO configs VALUES (?, 'retries', ?)",
                         (svc, str(rng.randint(1, 5))))
            conn.execute("INSERT INTO configs VALUES (?, 'pool_size', ?)",
                         (svc, str(rng.randrange(10, 90, 10))))
            for _ in range(2):
                conn.execute("INSERT INTO logs (service, level, message) "
                             "VALUES (?, ?, ?)",
                             (svc, rng.choice(["INFO", "WARN"]),
                              rng.choice(LOG_MESSAGES)))
        conn.commit()
        return cls(conn=conn, rng=rng)

    def snapshot(self) -> dict:
        out: dict = {}
        for table in ("services", "deployments", "configs", "logs", "tickets"):
            cur = self.conn.execute(f"SELECT * FROM {table}")
            cols = [d[0] for d in cur.description]
            out[table] = [dict(zip(cols, row)) for row in cur.fetchall()]
        return out

    def dump_json(self) -> str:
        return json.dumps(self.snapshot(), sort_keys=True)


# ---------------- tools ----------------

def search_logs(world: D2World, service: str) -> str:
    rows = world.conn.execute(
        "SELECT level, message FROM logs WHERE service=?", (service,)).fetchall()
    return json.dumps({"service": service,
                       "logs": [{"level": l, "message": m} for l, m in rows]})


def get_deploy_history(world: D2World, service: str) -> str:
    rows = world.conn.execute(
        "SELECT version, status FROM deployments WHERE service=? ORDER BY id",
        (service,)).fetchall()
    if not rows:
        return json.dumps({"error": f"service {service} not found"})
    return json.dumps({"service": service,
                       "deployments": [{"version": v, "status": s}
                                       for v, s in rows]})


def get_config(world: D2World, service: str, key: str) -> str:
    row = world.conn.execute(
        "SELECT value FROM configs WHERE service=? AND key=?",
        (service, key)).fetchone()
    if not row:
        return json.dumps({"error": f"no config {key} for {service}",
                           "available": CONFIG_KEYS})
    return json.dumps({"service": service, "key": key, "value": row[0]})


def set_config(world: D2World, service: str, key: str, value) -> str:
    if not world.conn.execute("SELECT 1 FROM services WHERE name=?",
                              (service,)).fetchone():
        return json.dumps({"error": f"service {service} not found"})
    world.conn.execute(
        "INSERT INTO configs VALUES (?,?,?) "
        "ON CONFLICT(service, key) DO UPDATE SET value=excluded.value",
        (service, key, str(value)))
    world.conn.commit()
    return json.dumps({"ok": True, "service": service, "key": key,
                       "value": str(value)})


def deploy(world: D2World, service: str, version: str) -> str:
    if not world.conn.execute("SELECT 1 FROM services WHERE name=?",
                              (service,)).fetchone():
        return json.dumps({"error": f"service {service} not found"})
    world.conn.execute("UPDATE deployments SET status='superseded' "
                       "WHERE service=? AND status='active'", (service,))
    world.conn.execute("INSERT INTO deployments (service, version, status) "
                       "VALUES (?,?, 'active')", (service, version))
    world.conn.execute("UPDATE services SET version=? WHERE name=?",
                       (version, service))
    world.conn.commit()
    return json.dumps({"ok": True, "service": service, "version": version})


def update_ticket(world: D2World, service: str, body: str) -> str:
    world.conn.execute("INSERT INTO tickets (service, body) VALUES (?,?)",
                       (service, body))
    world.conn.commit()
    return json.dumps({"ok": True})


TOOL_FUNCS = {
    "search_logs": search_logs,
    "get_deploy_history": get_deploy_history,
    "get_config": get_config,
    "set_config": set_config,
    "deploy": deploy,
    "update_ticket": update_ticket,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_logs",
        "description": "Fetch recent log lines for a service.",
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string"}}, "required": ["service"]}}},
    {"type": "function", "function": {
        "name": "get_deploy_history",
        "description": "List a service's deployments (version, status).",
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string"}}, "required": ["service"]}}},
    {"type": "function", "function": {
        "name": "get_config",
        "description": "Read one config value for a service.",
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string"}, "key": {"type": "string"}},
            "required": ["service", "key"]}}},
    {"type": "function", "function": {
        "name": "set_config",
        "description": "Set one config value for a service.",
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string"}, "key": {"type": "string"},
            "value": {"type": "string"}},
            "required": ["service", "key", "value"]}}},
    {"type": "function", "function": {
        "name": "deploy",
        "description": "Deploy (or roll back to) a version of a service.",
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string"}, "version": {"type": "string"}},
            "required": ["service", "version"]}}},
    {"type": "function", "function": {
        "name": "update_ticket",
        "description": "Append a note to the service's incident ticket.",
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string"}, "body": {"type": "string"}},
            "required": ["service", "body"]}}},
]

SYSTEM_PROMPT = """You are an SRE assistant for an internal platform team.
Resolve the user's request by calling tools. Verify facts with tools when
unsure. When the task is done, reply with a short confirmation and no further
tool calls.

{memory_block}"""


# ---------------- checkers (pure functions of a snapshot) ----------------

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


# ---------------- arc generator ----------------

@dataclass
class D2Arc(Arc):
    def make_world(self) -> D2World:
        return D2World.create(seed=self.seed)


def _fix_value(rng: random.Random, world_json: str) -> str:
    """A 4-digit odd config value whose string form is absent from the world."""
    while True:
        v = rng.randrange(1001, 9999, 2)
        if str(v) not in world_json:
            return str(v)


def _good_version(rng: random.Random, world_json: str) -> str:
    """A version string absent from the world (so it is not re-derivable
    from the deploy history)."""
    while True:
        v = f"v{rng.randint(5, 9)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}"
        if v not in world_json:
            return v


def generate_arc(arc_id: str, seed: int, n_episodes: int = 5,
                 dep_ratio: float = 0.5, difficulty: str = "easy") -> D2Arc:
    """Deterministic given (arc_id, seed, n_episodes, dep_ratio, difficulty).
    Difficulty tiers mirror D1: easy = single fact; medium = one plant with
    BOTH a config fix and a rollback target, probe requires both; hard = the
    rollback target is later REVISED and the probe needs the latest value."""
    rng = random.Random((seed, arc_id, difficulty).__repr__()
                        if difficulty != "easy"
                        else (seed, arc_id).__repr__())
    world = D2World.create(seed=seed)
    world_json = world.dump_json()
    services = list(SERVICES)
    rng.shuffle(services)

    arc = D2Arc(arc_id=arc_id, seed=seed)
    taboo = world_json  # grows with each gold: values unique arc-wide
    n_dep = round(dep_ratio * (n_episodes - 1))
    first_probe = 2 if difficulty == "hard" else 1
    n_dep = min(n_dep, n_episodes - first_probe)
    probe_positions = sorted(rng.sample(range(first_probe, n_episodes), n_dep))

    pending: dict[int, TaskSpec] = {}
    svc_iter = iter(services)

    for pos in probe_positions:
        svc = next(svc_iter)
        plant_pos = rng.randrange(0, pos)
        if difficulty == "medium":
            key = rng.choice(CONFIG_KEYS)
            val = _fix_value(rng, taboo)
            taboo += "|" + val
            gv = _good_version(rng, taboo)
            taboo += "|" + gv
            plant = TaskSpec(
                task_id=f"{arc_id}-e{plant_pos}-plant-combo",
                dependent=False, kind="I_note",
                user_messages=[
                    f"Postmortem for {svc}: full remediation is two steps — "
                    f"a config change and a rollback. Do NOT apply anything "
                    f"yet, we are in a change freeze. Just update the ticket "
                    f"confirming the planned fix: {key}={val} for {svc}, "
                    f"and noting the rollback target {gv} for {svc}."],
                checker="check_ticket_updated",
                checker_args={"service": svc, "must_contain": val})
            probe = TaskSpec(
                task_id=f"{arc_id}-e{pos}-probe-combo",
                dependent=True, kind="T_combo",
                user_messages=[
                    f"The change freeze on {svc} is lifted — please apply "
                    f"the FULL remediation we agreed on the ticket: both the "
                    f"config change and the rollback."],
                checker="check_config_and_version",
                checker_args={"service": svc, "key": key,
                              "expected_value": val, "expected_version": gv},
                gold_fact_value=val, gold_fact_values=[val, gv],
                plant_episode=plant_pos)
            pending[plant_pos] = _merge(pending.get(plant_pos), plant)
            pending[pos] = _merge(pending.get(pos), probe)
            continue
        if difficulty == "hard":
            gv1 = _good_version(rng, taboo)
            taboo += "|" + gv1
            gv2 = _good_version(rng, taboo)
            taboo += "|" + gv2
            plant_pos = rng.randrange(0, pos - 1)
            upd_pos = rng.randrange(plant_pos + 1, pos)
            plant = TaskSpec(
                task_id=f"{arc_id}-e{plant_pos}-plant-ver",
                dependent=False, kind="I_note",
                user_messages=[
                    f"Postmortem update for {svc}: the current release is "
                    f"faulty. Do NOT roll back yet — approval is pending. "
                    f"Just update the ticket noting the rollback target "
                    f"{gv1} for {svc}."],
                checker="check_ticket_updated",
                checker_args={"service": svc, "must_contain": gv1})
            update = TaskSpec(
                task_id=f"{arc_id}-e{upd_pos}-update-ver",
                dependent=False, kind="I_note",
                user_messages=[
                    f"Revision on the {svc} postmortem: QA found the earlier "
                    f"target also affected. The correct rollback target "
                    f"{gv2} for {svc} replaces the previous one. Still do "
                    f"NOT roll back yet; just update the ticket noting the "
                    f"rollback target {gv2} for {svc}."],
                checker="check_ticket_updated",
                checker_args={"service": svc, "must_contain": gv2})
            probe = TaskSpec(
                task_id=f"{arc_id}-e{pos}-probe-ver-upd",
                dependent=True, kind="T_ver_upd",
                user_messages=[
                    f"Approval came through for {svc} — roll it back now to "
                    f"the CURRENT rollback target from the ticket."],
                checker="check_version_deployed",
                checker_args={"service": svc, "expected_version": gv2},
                gold_fact_value=gv2, plant_episode=plant_pos)
            pending[plant_pos] = _merge(pending.get(plant_pos), plant)
            pending[upd_pos] = _merge(pending.get(upd_pos), update)
            pending[pos] = _merge(pending.get(pos), probe)
            continue
        if rng.random() < 0.5:
            key = rng.choice(CONFIG_KEYS)
            val = _fix_value(rng, taboo)
            taboo += "|" + val
            plant = TaskSpec(
                task_id=f"{arc_id}-e{plant_pos}-plant-cfg",
                dependent=False, kind="I_note",
                user_messages=[
                    f"On-call here about {svc}: we diagnosed the recurring "
                    f"degradation — the fix is setting {key} to {val}. Do NOT "
                    f"apply it yet, we are in a change freeze. For now, just "
                    f"update the ticket confirming the planned fix: "
                    f"{key}={val} for {svc}."],
                checker="check_ticket_updated",
                checker_args={"service": svc, "must_contain": val})
            probe = TaskSpec(
                task_id=f"{arc_id}-e{pos}-probe-cfg",
                dependent=True, kind="T_cfg",
                user_messages=[
                    f"{svc} is degraded again, same symptoms as the earlier "
                    f"incident. The change freeze is over — please apply the "
                    f"fix we agreed on the ticket."],
                checker="check_config_set",
                checker_args={"service": svc, "key": key,
                              "expected_value": val},
                gold_fact_value=val, plant_episode=plant_pos)
        else:
            gv = _good_version(rng, taboo)
            taboo += "|" + gv
            plant = TaskSpec(
                task_id=f"{arc_id}-e{plant_pos}-plant-ver",
                dependent=False, kind="I_note",
                user_messages=[
                    f"Postmortem update for {svc}: the current release is "
                    f"faulty. Do NOT roll back yet — approval is pending. "
                    f"Just update the ticket noting the rollback target "
                    f"{gv} for {svc}."],
                checker="check_ticket_updated",
                checker_args={"service": svc, "must_contain": gv})
            probe = TaskSpec(
                task_id=f"{arc_id}-e{pos}-probe-ver",
                dependent=True, kind="T_ver",
                user_messages=[
                    f"Approval came through for {svc} — roll it back now to "
                    f"the version we identified on the ticket."],
                checker="check_version_deployed",
                checker_args={"service": svc, "expected_version": gv},
                gold_fact_value=gv, plant_episode=plant_pos)
        pending[plant_pos] = _merge(pending.get(plant_pos), plant)
        pending[pos] = _merge(pending.get(pos), probe)

    for i in range(n_episodes):
        if i in pending:
            arc.episodes.append(EpisodeSpec(index=i, task=pending[i]))
            continue
        svc = next(svc_iter)
        if rng.random() < 0.5:
            key = rng.choice(CONFIG_KEYS)
            val = str(rng.randrange(1000, 9000, 500))
            task = TaskSpec(
                task_id=f"{arc_id}-e{i}-indep-cfg",
                dependent=False, kind="I_cfg",
                user_messages=[
                    f"Please set {key} to {val} for {svc}."],
                checker="check_config_set",
                checker_args={"service": svc, "key": key,
                              "expected_value": val})
        else:
            v = f"v{rng.randint(1, 4)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}"
            task = TaskSpec(
                task_id=f"{arc_id}-e{i}-indep-deploy",
                dependent=False, kind="I_deploy",
                user_messages=[f"Please deploy version {v} to {svc}."],
                checker="check_version_deployed",
                checker_args={"service": svc, "expected_version": v})
        arc.episodes.append(EpisodeSpec(index=i, task=task))

    arc.episodes.sort(key=lambda e: e.index)
    leak_check(arc, world_json)
    return arc


def _merge(existing: TaskSpec | None, new: TaskSpec) -> TaskSpec:
    if existing is None:
        return new
    existing.user_messages = existing.user_messages + new.user_messages
    return existing


def generate_suite(n_arcs: int = 10, episodes_per_arc: int = 5,
                   dep_ratio: float = 0.5, base_seed: int = 0,
                   difficulty: str = "easy") -> list[D2Arc]:
    return [generate_arc(arc_id=f"d2-arc{base_seed}-{i:03d}",
                         seed=base_seed * 10_000 + i,
                         n_episodes=episodes_per_arc, dep_ratio=dep_ratio,
                         difficulty=difficulty)
            for i in range(n_arcs)]
