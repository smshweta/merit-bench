"""Simulated user (Protocol Phase 1, step 3).

Two modes behind one interface:

  scripted (default, $0, fully deterministic): replays the arc generator's
      script verbatim. This is what tests and the offline pilot use.
  llm: role-plays from a fixed persona + the script skeleton at temperature 0,
      turning each scripted turn into natural phrasing. Turn outputs are
      cached on disk keyed by (persona, script-turn, model) so repeated runs
      are free, per the protocol's "cache user turns where possible".

IMPORTANT: the LLM mode re-runs the leak check on its own output — a
paraphrase must never smuggle the gold fact into a probe turn.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

PERSONAS = [
    "You are a polite but busy customer. You are brief and expect competence.",
    "You are a slightly annoyed customer. You had an issue and want it fixed.",
    "You are a chatty, friendly customer, but your requests are still clear.",
]

CACHE_DIR = Path(os.environ.get("MERIT_USER_CACHE", ".user_cache"))


class SimulatedUser:
    def __init__(self, script: list[str], persona_idx: int = 0,
                 mode: str = "scripted", model: str | None = None,
                 forbidden: str | list[str] = "") -> None:
        self.script = script
        self.persona = PERSONAS[persona_idx % len(PERSONAS)]
        self.mode = mode
        self.model = model
        # gold fact value(s) that must NOT be uttered by a paraphrase
        self.forbidden = ([forbidden] if isinstance(forbidden, str)
                          else list(forbidden))

    def turns(self) -> list[str]:
        if self.mode == "scripted":
            return list(self.script)
        return [self._llm_turn(t) for t in self.script]

    # ---------------- llm mode ----------------
    def _llm_turn(self, scripted: str) -> str:
        key = hashlib.sha256(
            f"{self.persona}|{scripted}|{self.model}".encode()).hexdigest()[:24]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached = CACHE_DIR / f"{key}.json"
        if cached.exists():
            return json.loads(cached.read_text())["text"]

        import litellm
        resp = litellm.completion(
            model=self.model or os.environ.get("MERIT_MODEL"),
            temperature=0,
            messages=[
                {"role": "system", "content":
                    f"{self.persona}\nRewrite the following customer-support "
                    f"request in your own voice. Keep ALL identifiers, "
                    f"addresses, amounts, and order numbers EXACTLY as "
                    f"written. Output only the rewritten message."},
                {"role": "user", "content": scripted}])
        text = resp.choices[0].message.content.strip()

        # paraphrase leak check: forbidden values must not appear unless they
        # were already in the scripted turn (plants legitimately contain them)
        for value in self.forbidden:
            if value and value not in scripted and value in text:
                raise AssertionError(
                    f"user_sim leak: paraphrase introduced forbidden value "
                    f"{value!r}")
        cached.write_text(json.dumps({"text": text}))
        return text
