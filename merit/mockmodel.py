"""Deterministic mock model for $0 end-to-end pipeline validation.

Speaks the same response interface as litellm.completion, so the runner is
identical for mock and real models (set model="mock").

The mock agent is a competent rule-based policy: it uses facts from the
memory block when present and falls back to naive-but-plausible behavior
when absent (full refund instead of the agreed partial amount; "cannot
verify" instead of the remembered address). It TRUSTS memory verbatim, so
corrupted memory produces confidently wrong actions.

IMPORTANT — scientific status: mock results validate the HARNESS (checkers,
leak check, metering, analysis code). They are NOT evidence about LLM agent
behavior and must never appear in the paper as such. In particular H2
(distraction cost) is structurally invisible to a rule-based policy.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field


# ---------- litellm-compatible response shells ----------

@dataclass
class _Function:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    id: str
    function: _Function
    type: str = "function"


@dataclass
class _Message:
    content: str | None = None
    tool_calls: list[_ToolCall] | None = None
    role: str = "assistant"

    def model_dump(self, exclude_none: bool = False) -> dict:
        d: dict = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": tc.type,
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in self.tool_calls]
        if exclude_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _Response:
    choices: list[_Choice]
    usage: _Usage


def _tc(name: str, **args) -> _ToolCall:
    return _ToolCall(id=uuid.uuid4().hex[:12],
                     function=_Function(name=name, arguments=json.dumps(args)))


def _resp(message: _Message, messages: list) -> _Response:
    prompt_chars = sum(len(str(m)) for m in messages)
    completion_chars = len(str(message.model_dump()))
    return _Response(choices=[_Choice(message=message)],
                     usage=_Usage(prompt_tokens=prompt_chars // 4,
                                  completion_tokens=completion_chars // 4))


# ---------- memory-block readers ----------

def find_agreed_cents(memory: str, oid: str) -> str | None:
    patterns = [
        rf"agreed amount of (\d+) cents for order {oid}",
        rf"{oid}\.agreed_refund_cents = (\d+)",
        rf"refund of exactly (\d+) cents[^\n]*\n?[^\n]*{oid}",
        rf"{oid}[^\n]*?(\d{{3,}}) cents",
    ]
    for p in patterns:
        found = re.findall(p, memory)
        if found:
            return found[-1]  # last match = latest value in ordered stores
    return None


def find_address(memory: str, cid: str) -> str | None:
    patterns = [
        rf"{cid}\.address = (.+?)  \(ep",
        rf'"customer_id": "{cid}", "address": "([^"]+)"',
        rf"{cid}[^\n]*address on file to: ([^\n\"]+)",
    ]
    for p in patterns:
        found = re.findall(p, memory)
        if found:
            return found[-1].strip().rstrip(".")
    return None


def _find_last(memory: str, patterns: list[str]):
    """Last match wins: chronological stores yield the LATEST value; RAG's
    relevance ordering makes this arbitrary — which is precisely the
    updated-fact weakness the hard tier measures."""
    for p in patterns:
        found = re.findall(p, memory)
        if found:
            return found[-1]
    return None


def find_planned_fix(memory: str, svc: str):
    hit = _find_last(memory, [rf"{svc}\.planned_fix = (\w+)=(\d+)",
                              rf"planned fix: (\w+)=(\d+) for {svc}"])
    return hit  # (key, value) or None


def find_rollback_target(memory: str, svc: str) -> str | None:
    return _find_last(memory, [rf"{svc}\.rollback_target = (v[\d.]+)",
                               rf"rollback target (v[\d.]+) for {svc}"])


def find_usual_room(memory: str) -> str | None:
    return _find_last(memory, [r"user\.usual_room = (Room \d+[A-Z])",
                               r"usual room is (Room \d+[A-Z])"])


def find_usual_time(memory: str) -> str | None:
    return _find_last(memory, [r"user\.usual_time = (\d\d:\d\d)",
                               r"usual meeting time is (\d\d:\d\d)"])


def find_dinner(memory: str, person: str):
    hit = _find_last(
        memory, [rf"{person}\.dinner = (.+?) on (\w+) at (\d\d:\d\d)",
                 rf"dinner with {person} at (.+?) on (\w+) at (\d\d:\d\d)"])
    return hit  # (place, day, time) or None


# ---------- the policy ----------

def _executed_since_last_user(messages: list) -> list[tuple[str, dict, str]]:
    """(tool_name, args, output) for tool calls after the last user message."""
    last_user = max(i for i, m in enumerate(messages)
                    if _get(m, "role") == "user")
    calls: dict[str, tuple[str, dict]] = {}
    out: list[tuple[str, dict, str]] = []
    for m in messages[last_user:]:
        for tc in (_get(m, "tool_calls") or []):
            tcid = _get(tc, "id")
            fn = _get(tc, "function")
            calls[tcid] = (_get(fn, "name"),
                           json.loads(_get(fn, "arguments") or "{}"))
        if _get(m, "role") == "tool":
            name, args = calls.get(_get(m, "tool_call_id"), ("?", {}))
            out.append((name, args, _get(m, "content") or ""))
    return out


def _get(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def completion(model: str, messages: list, tools=None,
               temperature: float = 0) -> _Response:
    system = str(_get(messages[0], "content") or "")
    user = str(next(_get(m, "content") for m in reversed(messages)
                    if _get(m, "role") == "user"))
    done = _executed_since_last_user(messages)
    done_names = [d[0] for d in done]

    oid_m = re.search(r"\b(ORD-\d+)\b", user)
    cid_m = re.search(r"\b(CUST-\d+)\b", user)
    oid = oid_m.group(1) if oid_m else None
    cid = cid_m.group(1) if cid_m else None

    def final(text: str) -> _Response:
        return _resp(_Message(content=text), messages)

    def call(*tcs: _ToolCall) -> _Response:
        return _resp(_Message(content=None, tool_calls=list(tcs)), messages)

    svc_m = re.search(r"\b([a-z]+-api)\b", user)
    svc = svc_m.group(1) if svc_m else None

    # ---- D1 difficulty=medium: combo plant (agreed amount + new address)
    m = re.search(r"agreed amount of (\d+) cents for order (ORD-\d+)", user)
    addr_m = re.search(r"address on file to: (.+?)\. And send", user)
    if m and addr_m and cid:
        if "send_message" in done_names and "update_address" in done_names:
            return final("Confirmed and updated.")
        return call(
            _tc("send_message", customer_id=cid,
                body=f"Confirmed: agreed refund amount of {m.group(1)} "
                     f"cents for order {m.group(2)}."),
            _tc("update_address", customer_id=cid,
                new_address=addr_m.group(1).strip()))

    # ---- D1 difficulty=medium: combo probe (refund + address, both memory)
    if (oid and cid and re.search(r"refund we agreed", user)
            and "confirming the delivery address" in user):
        if "refund" in done_names and "send_message" in done_names:
            return final("Refund processed and address confirmed.")
        agreed = find_agreed_cents(system, oid)
        addr = find_address(system, cid)
        if agreed and addr:
            return call(
                _tc("refund", order_id=oid, amount_cents=int(agreed)),
                _tc("send_message", customer_id=cid,
                    body=f"The delivery address on file is: {addr}"))
        return final("I could not find the agreed details in my notes; "
                     "please resend them.")

    # 1. plant: confirm agreed amount by message
    m = re.search(r"confirming this agreed amount of (\d+) cents for order "
                  r"(ORD-\d+)", user)
    if m and cid:
        if "send_message" in done_names:
            return final("Confirmed and noted.")
        return call(_tc("send_message", customer_id=cid,
                        body=f"Confirmed: agreed refund amount of {m.group(1)} "
                             f"cents for order {m.group(2)}."))

    # 2. probe: process the refund we agreed on (amount only in memory)
    if oid and re.search(r"refund we agreed", user):
        if "refund" in done_names:
            return final("Your agreed refund has been processed.")
        agreed = find_agreed_cents(system, oid)
        if agreed:
            return call(_tc("refund", order_id=oid, amount_cents=int(agreed)))
        # no memory of the agreement -> confidently wrong fallback: full refund
        for name, args, out in done:
            if name == "get_order" and args.get("order_id") == oid:
                try:
                    amount = json.loads(out)["amount_cents"]
                except (KeyError, json.JSONDecodeError):
                    return final("I could not look up that order.")
                return call(_tc("refund", order_id=oid, amount_cents=amount))
        return call(_tc("get_order", order_id=oid))

    # 3. independent: full refund
    if oid and re.search(r"full refund", user):
        if "refund" in done_names:
            return final("Your full refund has been processed.")
        for name, args, out in done:
            if name == "get_order" and args.get("order_id") == oid:
                try:
                    amount = json.loads(out)["amount_cents"]
                except (KeyError, json.JSONDecodeError):
                    return final("I could not look up that order.")
                return call(_tc("refund", order_id=oid, amount_cents=amount))
        return call(_tc("get_order", order_id=oid))

    # 4. independent: address update (address is in the user message)
    m = re.search(r"address on file to: (.+)$", user, re.MULTILINE)
    if m and cid:
        if "update_address" in done_names:
            return final("Your address has been updated.")
        return call(_tc("update_address", customer_id=cid,
                        new_address=m.group(1).strip()))

    # 5. probe: confirm address on file (address only in memory)
    if cid and re.search(r"confirming the delivery address", user):
        if "send_message" in done_names:
            return final("Sent — please check your messages.")
        addr = find_address(system, cid)
        body = (f"The delivery address we have on file for you is: {addr}"
                if addr else
                "We could not verify an address on file for your account; "
                "please provide it.")
        return call(_tc("send_message", customer_id=cid, body=body))

    # ================= D2 (IT ops) =================

    # 6. plant/update: confirm planned fix / rollback target on the ticket
    fix_m = re.search(r"planned fix: (\w+=\d+) for ([a-z]+-api)", user)
    rb_m = re.search(r"rollback target (v[\d.]+) for ([a-z]+-api)", user)
    if (fix_m or rb_m) and re.search(r"do not", user, re.IGNORECASE):
        if "update_ticket" in done_names:
            return final("Ticket updated.")
        parts = []
        if fix_m:
            parts.append(f"planned fix: {fix_m.group(1)} for {fix_m.group(2)}")
        if rb_m:
            parts.append(f"rollback target {rb_m.group(1)} for {rb_m.group(2)}")
        return call(_tc("update_ticket", service=svc,
                        body="Confirmed — " + "; ".join(parts) + "."))

    # 7. probe (medium): apply the FULL remediation (config + rollback)
    if svc and re.search(r"full remediation", user, re.IGNORECASE):
        if "set_config" in done_names and "deploy" in done_names:
            return final("Full remediation applied.")
        fix = find_planned_fix(system, svc)
        target = find_rollback_target(system, svc)
        if fix and target:
            return call(_tc("set_config", service=svc, key=fix[0],
                            value=fix[1]),
                        _tc("deploy", service=svc, version=target))
        return final("I could not find the agreed remediation in my notes.")

    # 8. probe: apply the config fix we agreed (value only in memory)
    if svc and re.search(r"apply the fix we agreed", user):
        if "set_config" in done_names:
            return final("Fix applied.")
        fix = find_planned_fix(system, svc)
        if fix:
            return call(_tc("set_config", service=svc, key=fix[0],
                            value=fix[1]))
        return final("I could not find the agreed fix in my notes; "
                     "please resend it.")

    # 9. probe: roll back to the version we identified (memory only);
    #    fallback = previous version from history (plausible but wrong)
    if svc and re.search(r"roll (it )?back", user, re.IGNORECASE):
        if "deploy" in done_names:
            return final("Rollback complete.")
        target = find_rollback_target(system, svc)
        if target:
            return call(_tc("deploy", service=svc, version=target))
        for name, args, out in done:
            if name == "get_deploy_history":
                try:
                    deps = json.loads(out)["deployments"]
                    prev = [d for d in deps if d["status"] != "active"]
                    if prev:
                        return call(_tc("deploy", service=svc,
                                        version=prev[-1]["version"]))
                except (KeyError, json.JSONDecodeError):
                    pass
                return final("No usable deploy history found.")
        return call(_tc("get_deploy_history", service=svc))

    # 10. independent: set config / deploy with values in the message
    m = re.search(r"[Ss]et (\w+) to (\S+?) for ([a-z]+-api)", user)
    if m:
        if "set_config" in done_names:
            return final("Config updated.")
        return call(_tc("set_config", service=m.group(3), key=m.group(1),
                        value=m.group(2).rstrip(".")))
    m = re.search(r"[Dd]eploy version (v[\d.]+) to ([a-z]+-api)", user)
    if m:
        if "deploy" in done_names:
            return final("Deployed.")
        return call(_tc("deploy", service=m.group(2), version=m.group(1)))

    # ================= D3 (personal assistant) =================

    # 11. independent: reminder email with a verbatim phrase
    m = re.search(r"reminder email that says: (.+)$", user)
    if m:
        if "send_email" in done_names:
            return final("Reminder sent.")
        return call(_tc("send_email", to="me", subject="Reminder",
                        body=m.group(1).strip().rstrip(".")))

    # 12. plant: email confirming usual room / usual time / dinner details
    if re.search(r"send me an email confirming", user):
        if "send_email" in done_names:
            return final("Confirmation sent.")
        parts = []
        room_m = re.search(r"usual room is (Room \d+[A-Z])", user)
        if room_m:
            parts.append(f"your usual room is {room_m.group(1)}")
        time_m = re.search(r"usual meeting time is (\d\d:\d\d)", user)
        if time_m:
            parts.append(f"your usual meeting time is {time_m.group(1)}")
        din_m = re.search(
            r"dinner with (\w+) at (.+?) on (\w+) at (\d\d:\d\d)", user)
        if din_m:
            parts.append(f"dinner with {din_m.group(1)} at {din_m.group(2)} "
                         f"on {din_m.group(3)} at {din_m.group(4)}")
        if parts:
            return call(_tc("send_email", to="me", subject="Confirmation",
                            body="Confirmed: " + "; ".join(parts) + "."))

    # 13. probe (medium): book at my usual time in my usual room
    m = re.search(r"[Bb]ook a (.+?) with Dana on (\w+) at my usual time "
                  r"in my usual room", user)
    if m:
        if "create_event" in done_names:
            return final("Booked.")
        room, time = find_usual_room(system), find_usual_time(system)
        if room and time:
            return call(_tc("create_event", title=f"{m.group(1)} with Dana",
                            day=m.group(2), time=time, location=room))
        return final("I could not find your usual room and time in my notes.")

    # 14. probe (easy/hard): book at a given time in my usual room
    m = re.search(r"[Bb]ook a (.+?) with Dana on (\w+) at (\d\d:\d\d) "
                  r"in my usual room", user)
    if m:
        if "create_event" in done_names:
            return final("Booked.")
        room = find_usual_room(system)
        if room:
            return call(_tc("create_event", title=f"{m.group(1)} with Dana",
                            day=m.group(2), time=m.group(3), location=room))
        return final("I could not verify your usual room; please tell me "
                     "which room to book.")

    # 15. probe: add my dinner with <person> (all details from memory)
    m = re.search(r"add my dinner with (\w+)", user)
    if m:
        if "create_event" in done_names:
            return final("Dinner added.")
        dinner = find_dinner(system, m.group(1))
        if dinner:
            place, day, time = dinner
            return call(_tc("create_event",
                            title=f"Dinner with {m.group(1)}", day=day,
                            time=time, location=place))
        return final("I could not find the dinner details in my notes.")

    # 16. independent: add event with all details in the message
    m = re.search(r"[Aa]dd (.+?) to my calendar on (\w+) at (\d\d:\d\d), "
                  r"location: (.+?)\.?$", user)
    if m:
        if "create_event" in done_names:
            return final("Event added.")
        return call(_tc("create_event", title=m.group(1), day=m.group(2),
                        time=m.group(3), location=m.group(4).strip()))

    # 17. independent: update a preference
    m = re.search(r"[Uu]pdate my (\w+) preference to: (.+?)\.?$", user)
    if m:
        if "set_preference" in done_names:
            return final("Preference updated.")
        return call(_tc("set_preference", key=m.group(1),
                        value=m.group(2).strip()))

    return final("How can I help you today?")
