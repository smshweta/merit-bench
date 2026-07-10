"""Domain registry: one place that maps a domain name to its arc generator,
tools, and system prompt, so the pilot runner and analysis stay domain-
agnostic. All checkers live in merit.metrics (resolved by name)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import arcs as _d1_arcs
from . import d2 as _d2
from . import d3 as _d3
from .runner import SYSTEM_PROMPT as _D1_PROMPT
from .tools import TOOL_FUNCS as _D1_TOOLS, TOOL_SCHEMAS as _D1_SCHEMAS


@dataclass(frozen=True)
class Domain:
    name: str
    generate_suite: Callable
    tool_funcs: dict
    tool_schemas: list
    system_prompt: str


DOMAINS = {
    "d1": Domain("d1", _d1_arcs.generate_suite,
                 _D1_TOOLS, _D1_SCHEMAS, _D1_PROMPT),
    "d2": Domain("d2", _d2.generate_suite,
                 _d2.TOOL_FUNCS, _d2.TOOL_SCHEMAS, _d2.SYSTEM_PROMPT),
    "d3": Domain("d3", _d3.generate_suite,
                 _d3.TOOL_FUNCS, _d3.TOOL_SCHEMAS, _d3.SYSTEM_PROMPT),
}
