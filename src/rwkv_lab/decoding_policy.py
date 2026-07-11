"""Allowlisted, typed decoding-policy transitions for structured generation.

Inspired by an RWKV Discord proposal for model-controlled sampling regions:
https://discord.com/channels/992359628979568762/1076020543205163118/1140560536313004084.
The proposal is intentionally narrowed here: generated tokens can only select
named policies pre-authorized by the operator, and nesting/budget are bounded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SamplingPolicy:
    name: str
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    grammar: str = ""

    def __post_init__(self):
        if self.temperature < 0 or not 0 < self.top_p <= 1 or self.top_k < 0:
            raise ValueError("invalid sampling policy")


class DecodingPolicyMachine:
    """Push/pop named sampling modes using allowlisted control-token IDs."""
    def __init__(self, policies: Mapping[str, SamplingPolicy], *, default: str,
                 enter_tokens: Mapping[int, str] = (), exit_tokens: tuple[int, ...] = (),
                 max_depth: int = 4, max_transitions: int = 32):
        self.policies = dict(policies)
        if default not in self.policies or max_depth < 1 or max_transitions < 1:
            raise ValueError("policy machine needs a valid default and positive bounds")
        self.default = default; self.enter_tokens = dict(enter_tokens)
        if set(self.enter_tokens.values()) - set(self.policies):
            raise ValueError("control token refers to an unknown policy")
        if set(self.enter_tokens) & set(exit_tokens):
            raise ValueError("a token cannot both enter and exit a policy")
        self.exit_tokens = set(exit_tokens); self.max_depth = max_depth
        self.max_transitions = max_transitions; self.reset()

    def reset(self) -> None:
        self.stack = [self.default]; self.transitions = 0

    @property
    def current(self) -> SamplingPolicy:
        return self.policies[self.stack[-1]]

    def consume(self, token: int) -> SamplingPolicy:
        token = int(token)
        if token in self.enter_tokens:
            if self.transitions >= self.max_transitions or len(self.stack) >= self.max_depth:
                raise RuntimeError("decoding policy transition budget exceeded")
            self.stack.append(self.enter_tokens[token]); self.transitions += 1
        elif token in self.exit_tokens:
            if len(self.stack) == 1:
                raise RuntimeError("decoding policy stack underflow")
            self.stack.pop(); self.transitions += 1
        return self.current
