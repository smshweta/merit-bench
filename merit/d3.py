"""MERIT Domain D3: personal assistant. Self-contained — world, tools,
checkers, and arc generator in one module, mirroring D1's plant/probe design.

Dependent task types:
  T_room    plant: the user names their preferred meeting room and forbids
            saving it anywhere ("just email me a confirmation"). probe: "book
            X on <day> at <time> in my usual room" — the room name exists only
            in memory (the preferences store never contains it).
  T_dinner  MULTI-FACT probe: the plant mentions a dinner commitment (place,
            day, time); the probe asks to "add the dinner I told you about"
            with no details. All three facts must come from memory — this
            deliberately raises difficulty over D1/D2 single-fact probes.
"""
from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass

from .arcs import Arc, EpisodeSpec, TaskSpec, leak_check

SCHEMA = """
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    day TEXT NOT NULL,
    time TEXT NOT NULL,
    location TEXT NOT NULL
);
CREATE TABLE emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    to_addr TEXT NOT NULL,
    subject TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE TABLE preferences (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TITLES = ["team standup", "1:1 with Alex", "budget review", "gym class",
          "dentist appointment", "sprint planning", "book club"]
LOCATIONS = ["Room 2A", "Room 3C", "online", "HQ lobby", "clinic"]
DEFAULT_PREFS = {"coffee": "oat latte", "seat": "aisle",
                 "reminder_lead_minutes": "15"}


@dataclass
class D3World:
    """Owns the SQLite connection for one episode arc."""
    conn: sqlite3.Connection
    rng: random.Random

    @classmethod
    def create(cls, seed: int) -> "D3World":
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA)
        rng = random.Random(seed)
        for k, v in DEFAULT_PREFS.items():
            conn.execute("INSERT INTO preferences VALUES (?,?)", (k, v))
        for _ in range(5):
            conn.execute(
                "INSERT INTO events (title, day, time, location) "
                "VALUES (?,?,?,?)",
                (rng.choice(TITLES), rng.choice(DAYS),
                 f"{rng.randint(8, 17):02d}:00", rng.choice(LOCATIONS)))
        conn.commit()
        return cls(conn=conn, rng=rng)

    def snapshot(self) -> dict:
        out: dict = {}
        for table in ("events", "emails", "preferences"):
            cur = self.conn.execute(f"SELECT * FROM {table}")
            cols = [d[0] for d in cur.description]
            out[table] = [dict(zip(cols, row)) for row in cur.fetchall()]
        return out

    def dump_json(self) -> str:
        return json.dumps(self.snapshot(), sort_keys=True)


# ---------------- tools ----------------

def get_calendar(world: D3World, day: str) -> str:
    rows = world.conn.execute(
        "SELECT title, time, location FROM events WHERE day=? ORDER BY time",
        (day,)).fetchall()
    return json.dumps({"day": day,
                       "events": [{"title": t, "time": tm, "location": l}
                                  for t, tm, l in rows]})


def create_event(world: D3World, title: str, day: str, time: str,
                 location: str) -> str:
    world.conn.execute(
        "INSERT INTO events (title, day, time, location) VALUES (?,?,?,?)",
        (title, day, time, location))
    world.conn.commit()
    return json.dumps({"ok": True, "title": title, "day": day,
                       "time": time, "location": location})


def send_email(world: D3World, to: str, subject: str, body: str) -> str:
    world.conn.execute(
        "INSERT INTO emails (to_addr, subject, body) VALUES (?,?,?)",
        (to, subject, body))
    world.conn.commit()
    return json.dumps({"ok": True, "to": to})


def get_preference(world: D3World, key: str) -> str:
    row = world.conn.execute("SELECT value FROM preferences WHERE key=?",
                             (key,)).fetchone()
    if not row:
        return json.dumps({"error": f"no preference named {key}",
                           "available": [r[0] for r in world.conn.execute(
                               "SELECT key FROM preferences")]})
    return json.dumps({"key": key, "value": row[0]})


def set_preference(world: D3World, key: str, value: str) -> str:
    world.conn.execute(
        "INSERT INTO preferences VALUES (?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
    world.conn.commit()
    return json.dumps({"ok": True, "key": key, "value": value})


TOOL_FUNCS = {
    "get_calendar": get_calendar,
    "create_event": create_event,
    "send_email": send_email,
    "get_preference": get_preference,
    "set_preference": set_preference,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_calendar",
        "description": "List the user's events on a given day "
                       "(Monday..Friday).",
        "parameters": {"type": "object", "properties": {
            "day": {"type": "string"}}, "required": ["day"]}}},
    {"type": "function", "function": {
        "name": "create_event",
        "description": "Add an event to the user's calendar.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"}, "day": {"type": "string"},
            "time": {"type": "string"}, "location": {"type": "string"}},
            "required": ["title", "day", "time", "location"]}}},
    {"type": "function", "function": {
        "name": "send_email",
        "description": "Send an email; use to='me' for the user themself.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string"}, "subject": {"type": "string"},
            "body": {"type": "string"}},
            "required": ["to", "subject", "body"]}}},
    {"type": "function", "function": {
        "name": "get_preference",
        "description": "Read one stored user preference by key.",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string"}}, "required": ["key"]}}},
    {"type": "function", "function": {
        "name": "set_preference",
        "description": "Store one user preference.",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string"}, "value": {"type": "string"}},
            "required": ["key", "value"]}}},
]

SYSTEM_PROMPT = """You are a personal assistant managing the user's calendar,
email, and preferences. Resolve the user's request by calling tools. Verify
facts with tools when unsure. When the task is done, reply with a short
confirmation and no further tool calls.

{memory_block}"""


# ---------------- checkers (pure functions of a snapshot) ----------------

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


# ---------------- arc generator ----------------

@dataclass
class D3Arc(Arc):
    def make_world(self) -> D3World:
        return D3World.create(seed=self.seed)


ROOM_LETTERS = "DEFGHJKL"  # avoid letters used by seeded rooms (A, C)
PLACE_FIRST = ["Cafe", "Bistro", "Trattoria", "Izakaya", "Brasserie"]
PLACE_SECOND = ["Meridian", "Juniper", "Saffron", "Lumen", "Vesper",
                "Cobalt", "Marigold", "Halcyon"]
PEOPLE = ["Sam", "Dana", "Ravi", "Noor", "Felix"]


def _room(rng: random.Random, world_json: str) -> str:
    while True:
        r = f"Room {rng.randint(10, 99)}{rng.choice(ROOM_LETTERS)}"
        if r not in world_json:
            return r


def _place(rng: random.Random, world_json: str) -> str:
    while True:
        p = f"{rng.choice(PLACE_FIRST)} {rng.choice(PLACE_SECOND)}"
        if p not in world_json:
            return p


def generate_arc(arc_id: str, seed: int, n_episodes: int = 5,
                 dep_ratio: float = 0.5) -> D3Arc:
    """Deterministic given (arc_id, seed, n_episodes, dep_ratio)."""
    rng = random.Random((seed, arc_id).__repr__())
    world = D3World.create(seed=seed)
    world_json = world.dump_json()

    arc = D3Arc(arc_id=arc_id, seed=seed)
    n_dep = round(dep_ratio * (n_episodes - 1))
    probe_positions = sorted(rng.sample(range(1, n_episodes), n_dep))

    pending: dict[int, TaskSpec] = {}

    for pos in probe_positions:
        plant_pos = rng.randrange(0, pos)
        if rng.random() < 0.5:
            room = _room(rng, world_json)
            title = rng.choice(["design sync", "vendor call", "retro"])
            day = rng.choice(DAYS)
            time = f"{rng.randint(9, 16):02d}:30"
            plant = TaskSpec(
                task_id=f"{arc_id}-e{plant_pos}-plant-room",
                dependent=False, kind="I_note",
                user_messages=[
                    f"From now on, book all my meetings in {room} — it has "
                    f"the good screen. Don't save this anywhere or change "
                    f"any events yet; just send me an email confirming my "
                    f"usual room is {room}."],
                checker="check_email_sent",
                checker_args={"must_contain": room})
            probe = TaskSpec(
                task_id=f"{arc_id}-e{pos}-probe-room",
                dependent=True, kind="T_room",
                user_messages=[
                    f"Please book a {title} with Dana on {day} at {time} "
                    f"in my usual room."],
                checker="check_event_created",
                checker_args={"day": day, "time": time,
                              "expected_location": room},
                gold_fact_value=room, plant_episode=plant_pos)
        else:
            place = _place(rng, world_json)
            person = rng.choice(PEOPLE)
            day = rng.choice(DAYS)
            time = f"{rng.randint(18, 20):02d}:30"
            plant = TaskSpec(
                task_id=f"{arc_id}-e{plant_pos}-plant-dinner",
                dependent=False, kind="I_note",
                user_messages=[
                    f"I just promised {person} dinner but can't add it to "
                    f"the calendar yet — waiting on the babysitter. Don't "
                    f"create the event; just send me an email confirming "
                    f"dinner with {person} at {place} on {day} at {time}."],
                checker="check_email_sent",
                checker_args={"must_contain": place})
            probe = TaskSpec(
                task_id=f"{arc_id}-e{pos}-probe-dinner",
                dependent=True, kind="T_dinner",
                user_messages=[
                    "The babysitter is confirmed — please add the dinner I "
                    "told you about to my calendar, with the right place, "
                    "day, and time."],
                checker="check_event_created",
                checker_args={"day": day, "time": time,
                              "expected_location": place},
                gold_fact_value=place, plant_episode=plant_pos)
        pending[plant_pos] = _merge(pending.get(plant_pos), plant)
        pending[pos] = _merge(pending.get(pos), probe)

    for i in range(n_episodes):
        if i in pending:
            arc.episodes.append(EpisodeSpec(index=i, task=pending[i]))
            continue
        r = rng.random()
        if r < 0.4:
            title = rng.choice(TITLES)
            day, time = rng.choice(DAYS), f"{rng.randint(8, 17):02d}:00"
            loc = rng.choice(LOCATIONS)
            task = TaskSpec(
                task_id=f"{arc_id}-e{i}-indep-event",
                dependent=False, kind="I_event",
                user_messages=[
                    f"Please add {title} to my calendar on {day} at {time}, "
                    f"location: {loc}."],
                checker="check_event_created",
                checker_args={"day": day, "time": time,
                              "expected_location": loc})
        elif r < 0.7:
            key = rng.choice(["coffee", "seat", "reminder_lead_minutes"])
            val = rng.choice(["flat white", "window", "30", "cortado", "45"])
            task = TaskSpec(
                task_id=f"{arc_id}-e{i}-indep-pref",
                dependent=False, kind="I_pref",
                user_messages=[
                    f"Update my {key} preference to: {val}."],
                checker="check_preference_set",
                checker_args={"key": key, "expected_value": val})
        else:
            phrase = f"pick up package #{rng.randint(100, 999)}"
            task = TaskSpec(
                task_id=f"{arc_id}-e{i}-indep-email",
                dependent=False, kind="I_email",
                user_messages=[
                    f"Send me a reminder email that says: {phrase}."],
                checker="check_email_sent",
                checker_args={"must_contain": phrase})
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
                   dep_ratio: float = 0.5, base_seed: int = 0) -> list[D3Arc]:
    return [generate_arc(arc_id=f"d3-arc{base_seed}-{i:03d}",
                         seed=base_seed * 10_000 + i,
                         n_episodes=episodes_per_arc, dep_ratio=dep_ratio)
            for i in range(n_arcs)]
