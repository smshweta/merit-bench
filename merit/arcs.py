"""Arc generator for Domain D1 (Protocol Phase 1, step 2).

An ARC is a sequence of 4-6 episodes sharing one World (same customers/orders).
Each episode contains exactly one scored task, either:

  - DEPENDENT: success requires a fact PLANTED in an earlier episode of the
    same arc. The fact is absent from the current episode's inputs and cannot
    be re-derived from tools (verified by construction + the leak check).
  - INDEPENDENT: solvable entirely within-episode.

Dependent task types (D1):
  T_addr   plant: user changes address to a NEW value in episode i.
           probe: episode j asks the agent to send a message confirming the
           address on file. No tool returns the current customer address
           (get_order returns the ORDER's ship_address = the OLD one), so the
           only source of the new address is memory.
  T_refund plant: user records an agreed partial-refund amount for order O
           (a value != the order total). probe: episode j asks to "process the
           refund we agreed on" for O. The agreed amount is only in memory;
           a memoryless agent's best guess (full amount) is scored as failure.

The LEAK CHECK asserts, for every dependent task, that the gold fact value
appears in NO user message of the probe episode and NOWHERE in the initial
world state. It runs at generation time; generation fails loudly on a leak.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .world import World, STREETS, CITIES


@dataclass
class TaskSpec:
    task_id: str
    dependent: bool
    kind: str                      # T_addr | T_refund | I_refund | I_addr | I_policy
    user_messages: list[str]
    checker: str                   # name of checker in merit.metrics
    checker_args: dict
    gold_fact_value: str = ""      # the value memory must supply (dependent only)
    plant_episode: int = -1        # index of the episode where the fact was planted
    gold_fact_values: list[str] | None = None  # multi-fact probes (difficulty
    #                                            medium/hard); None = single

    def golds(self) -> list[str]:
        """All values memory must supply for this task."""
        if self.gold_fact_values:
            return self.gold_fact_values
        return [self.gold_fact_value] if self.gold_fact_value else []


@dataclass
class EpisodeSpec:
    index: int
    task: TaskSpec


@dataclass
class Arc:
    arc_id: str
    seed: int
    episodes: list[EpisodeSpec] = field(default_factory=list)

    def make_world(self) -> World:
        return World.create(seed=self.seed)


class LeakError(AssertionError):
    pass


def _new_address(rng: random.Random, world_json: str) -> str:
    """An address guaranteed not to collide with anything in the world."""
    while True:
        addr = (f"{rng.randint(10, 999)} {rng.choice(STREETS)} Apt "
                f"{rng.randint(1, 99)}, {rng.choice(CITIES)}")
        if addr not in world_json:
            return addr


def _agreed_cents(rng: random.Random, order_amount: int,
                  world_json: str) -> int:
    """A partial-refund amount that differs from the order total and from
    obvious fractions of it, and whose string form appears NOWHERE in the
    initial world state (world amounts are multiples of 100; we also reject
    substring collisions with order IDs etc. so the leak check stays exact)."""
    while True:
        v = rng.randrange(301, order_amount)
        if v % 100 == 0 or v in (order_amount, order_amount // 2):
            continue
        if str(v) not in world_json:
            return v


def generate_arc(arc_id: str, seed: int, n_episodes: int = 5,
                 dep_ratio: float = 0.5, difficulty: str = "easy") -> Arc:
    """Deterministic given (arc_id, seed, n_episodes, dep_ratio, difficulty).

    difficulty:
      easy    single-fact probes (T_addr, T_refund)
      medium  MULTI-FACT probes: one plant carries two facts (agreed refund
              amount AND new address); the probe requires both
      hard    UPDATED-FACT probes: the fact is planted, then CHANGED in a
              later episode; the probe requires the latest value. Stresses
              update-on-write vs. stores that retain both values.
    """
    rng = random.Random((seed, arc_id, difficulty).__repr__()
                        if difficulty != "easy"
                        else (seed, arc_id).__repr__())
    world = World.create(seed=seed)
    world_json = world.dump_json()
    orders = world.conn.execute(
        "SELECT id, customer_id, amount_cents FROM orders").fetchall()
    rng.shuffle(orders)

    arc = Arc(arc_id=arc_id, seed=seed)
    # taboo grows with every gold so values are unique arc-wide (a repeated
    # value in another pair's plant would leak into a probe episode)
    taboo = world_json
    n_dep = round(dep_ratio * (n_episodes - 1))  # episode 0 can't be dependent
    # hard probes need plant < update < probe, so they start at episode 2
    first_probe = 2 if difficulty == "hard" else 1
    n_dep = min(n_dep, n_episodes - first_probe)
    probe_positions = sorted(rng.sample(range(first_probe, n_episodes), n_dep))

    pending: dict[int, TaskSpec] = {}
    order_iter = iter(orders)

    for pos in probe_positions:
        oid, cid, amount = next(order_iter)
        if difficulty == "medium":
            agreed = _agreed_cents(rng, amount, taboo)
            taboo += "|" + str(agreed)
            new_addr = _new_address(rng, taboo)
            taboo += "|" + new_addr
            plant_pos = rng.randrange(0, pos)
            plant = TaskSpec(
                task_id=f"{arc_id}-e{plant_pos}-plant-combo",
                dependent=False, kind="I_note",
                user_messages=[
                    f"Hello, this is customer {cid}, about order {oid}: the "
                    f"item arrived damaged. Your colleague agreed I'll get a "
                    f"partial refund of exactly {agreed} cents. Do NOT "
                    f"process the refund yet — I still need to confirm with "
                    f"my bank. Also, I've moved: please update my address on "
                    f"file to: {new_addr}. And send me a message confirming "
                    f"this agreed amount of {agreed} cents for order {oid}."],
                checker="check_message_sent",
                checker_args={"customer_id": cid,
                              "must_contain": str(agreed)})
            probe = TaskSpec(
                task_id=f"{arc_id}-e{pos}-probe-combo",
                dependent=True, kind="T_combo",
                user_messages=[
                    f"Hi, customer {cid} about order {oid} — I'm ready now. "
                    f"Please process the partial refund we agreed on, and "
                    f"send me a message confirming the delivery address you "
                    f"have on file for the replacement. The message must "
                    f"state the full address."],
                checker="check_refund_and_message",
                checker_args={"order_id": oid, "expected_cents": agreed,
                              "customer_id": cid, "must_contain": new_addr},
                gold_fact_value=str(agreed),
                gold_fact_values=[str(agreed), new_addr],
                plant_episode=plant_pos)
            pending[plant_pos] = _merge(pending.get(plant_pos), plant)
            pending[pos] = _merge(pending.get(pos), probe)
            continue
        if difficulty == "hard":
            agreed1 = _agreed_cents(rng, amount, taboo)
            taboo += "|" + str(agreed1)
            agreed2 = _agreed_cents(rng, amount, taboo)
            taboo += "|" + str(agreed2)
            plant_pos = rng.randrange(0, pos - 1)
            upd_pos = rng.randrange(plant_pos + 1, pos)
            plant = TaskSpec(
                task_id=f"{arc_id}-e{plant_pos}-plant-refund",
                dependent=False, kind="I_note",
                user_messages=[
                    f"Hello, this is customer {cid}, about order {oid}: "
                    f"the item arrived damaged. Your colleague agreed I'll "
                    f"get a partial refund of exactly {agreed1} cents. Do "
                    f"NOT process the refund yet — I still need to confirm "
                    f"with my bank. For now, just send me a message "
                    f"confirming this agreed amount of {agreed1} cents for "
                    f"order {oid}."],
                checker="check_message_sent",
                checker_args={"customer_id": cid,
                              "must_contain": str(agreed1)})
            update = TaskSpec(
                task_id=f"{arc_id}-e{upd_pos}-update-refund",
                dependent=False, kind="I_note",
                user_messages=[
                    f"Customer {cid} again about order {oid}: we "
                    f"re-negotiated with your colleague — the agreed partial "
                    f"refund is now exactly {agreed2} cents, replacing the "
                    f"earlier figure. Still do NOT process it yet. Just send "
                    f"me a message confirming this agreed amount of "
                    f"{agreed2} cents for order {oid}."],
                checker="check_message_sent",
                checker_args={"customer_id": cid,
                              "must_contain": str(agreed2)})
            probe = TaskSpec(
                task_id=f"{arc_id}-e{pos}-probe-refund-upd",
                dependent=True, kind="T_refund_upd",
                user_messages=[
                    f"Hi, following up on order {oid} — please now process "
                    f"the partial refund we agreed on. Use the CURRENT "
                    f"agreed amount from your notes."],
                checker="check_refund_issued",
                checker_args={"order_id": oid, "expected_cents": agreed2},
                gold_fact_value=str(agreed2), plant_episode=plant_pos)
            pending[plant_pos] = _merge(pending.get(plant_pos), plant)
            pending[upd_pos] = _merge(pending.get(upd_pos), update)
            pending[pos] = _merge(pending.get(pos), probe)
            continue
        if rng.random() < 0.5:
            new_addr = _new_address(rng, taboo)
            taboo += "|" + new_addr
            plant_pos = rng.randrange(0, pos)
            plant = TaskSpec(
                task_id=f"{arc_id}-e{plant_pos}-plant-addr",
                dependent=False, kind="I_addr",
                user_messages=[
                    f"Hi, this is customer {cid}. I've moved — please update "
                    f"my address on file to: {new_addr}"],
                checker="check_address_updated",
                checker_args={"customer_id": cid, "expected_address": new_addr})
            probe = TaskSpec(
                task_id=f"{arc_id}-e{pos}-probe-addr",
                dependent=True, kind="T_addr",
                user_messages=[
                    f"Hi, customer {cid} again. Before you ship my replacement, "
                    f"send me a message confirming the delivery address you "
                    f"have on file for me. It must state the full address."],
                checker="check_message_sent",
                checker_args={"customer_id": cid, "must_contain": new_addr},
                gold_fact_value=new_addr, plant_episode=plant_pos)
        else:
            agreed = _agreed_cents(rng, amount, taboo)
            taboo += "|" + str(agreed)
            plant_pos = rng.randrange(0, pos)
            plant = TaskSpec(
                task_id=f"{arc_id}-e{plant_pos}-plant-refund",
                dependent=False, kind="I_note",
                user_messages=[
                    f"Hello, this is customer {cid}, about order {oid}: "
                    f"the item arrived damaged. Your "
                    f"colleague agreed I'll get a partial refund of exactly "
                    f"{agreed} cents. Do NOT process the refund yet — I still "
                    f"need to confirm with my bank, and I'll follow up when "
                    f"I'm ready. For now, just send me a message confirming "
                    f"this agreed amount of {agreed} cents for order {oid}."],
                checker="check_message_sent",
                checker_args={"customer_id": cid, "must_contain": str(agreed)})
            probe = TaskSpec(
                task_id=f"{arc_id}-e{pos}-probe-refund",
                dependent=True, kind="T_refund",
                user_messages=[
                    f"Hi, following up on order {oid} — please now process "
                    f"the partial refund we agreed on earlier. You should "
                    f"have the exact amount in your notes."],
                checker="check_refund_issued",
                checker_args={"order_id": oid, "expected_cents": agreed},
                gold_fact_value=str(agreed), plant_episode=plant_pos)
        pending[plant_pos] = _merge(pending.get(plant_pos), plant)
        pending[pos] = _merge(pending.get(pos), probe)

    # fill remaining episodes with independent tasks
    for i in range(n_episodes):
        if i in pending:
            arc.episodes.append(EpisodeSpec(index=i, task=pending[i]))
            continue
        oid, cid, amount = next(order_iter)
        r = rng.random()
        if r < 0.5:
            task = TaskSpec(
                task_id=f"{arc_id}-e{i}-indep-refund",
                dependent=False, kind="I_refund",
                user_messages=[f"I'd like a full refund on order {oid}, please."],
                checker="check_refund_issued",
                checker_args={"order_id": oid, "expected_cents": amount})
        else:
            addr = _new_address(rng, world_json)
            task = TaskSpec(
                task_id=f"{arc_id}-e{i}-indep-addr",
                dependent=False, kind="I_addr",
                user_messages=[
                    f"Hi, this is customer {cid}. Please update my address "
                    f"on file to: {addr}"],
                checker="check_address_updated",
                checker_args={"customer_id": cid, "expected_address": addr})
        arc.episodes.append(EpisodeSpec(index=i, task=task))

    arc.episodes.sort(key=lambda e: e.index)
    leak_check(arc, world_json)
    return arc


def _merge(existing: TaskSpec | None, new: TaskSpec) -> TaskSpec:
    """One scored task per episode; if a slot is taken, chain the user
    messages so the plant still happens but only the first task is scored."""
    if existing is None:
        return new
    existing.user_messages = existing.user_messages + new.user_messages
    return existing


def leak_check(arc: Arc, world_json: str) -> None:
    """Protocol Phase 1 step 2: every gold fact string must never appear in
    the probe episode's inputs, nor in the initial world state."""
    for ep in arc.episodes:
        t = ep.task
        if not t.dependent:
            continue
        for gold in t.golds():
            for msg in t.user_messages:
                if gold in msg:
                    raise LeakError(
                        f"LEAK {t.task_id}: gold fact {gold!r} "
                        f"appears in probe input: {msg!r}")
            if gold in world_json:
                raise LeakError(
                    f"LEAK {t.task_id}: gold fact {gold!r} "
                    f"exists in initial world state (re-derivable from tools)")
        if not (0 <= t.plant_episode < ep.index):
            raise LeakError(
                f"LEAK {t.task_id}: plant episode {t.plant_episode} is not "
                f"strictly before probe episode {ep.index}")


def generate_suite(n_arcs: int = 10, episodes_per_arc: int = 5,
                   dep_ratio: float = 0.5, base_seed: int = 0,
                   difficulty: str = "easy") -> list[Arc]:
    return [generate_arc(arc_id=f"arc{base_seed}-{i:03d}",
                         seed=base_seed * 10_000 + i,
                         n_episodes=episodes_per_arc, dep_ratio=dep_ratio,
                         difficulty=difficulty)
            for i in range(n_arcs)]
