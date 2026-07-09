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
        m = re.search(p, memory)
        if m:
            return m.group(1)
    return None


def find_address(memory: str, cid: str) -> str | None:
    patterns = [
        rf"{cid}\.address = (.+?)  \(ep",
        rf'"customer_id": "{cid}", "address": "([^"]+)"',
        rf"{cid}[^\n]*address on file to: ([^\n\"]+)",
    ]
    for p in patterns:
        m = re.search(p, memory)
        if m:
            return m.group(1).strip().rstrip(".")
    return None


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

    return final("How can I help you today?")
