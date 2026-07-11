"""Tool-augmented length-generalization curriculum and evaluation.

Malach et al. (2025), https://arxiv.org/abs/2510.14826. Community lead:
https://discord.com/channels/992359628979568762/1426889957221466153/1492868620831821894
Tool execution remains Adamaton's responsibility; this module handles inert tapes only.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ToolStep:
    observation: str; tool: str; arguments: dict; result: str

def make_length_curriculum(generator, lengths, examples_per_length=8):
    return [{"train_length":n, "tape":generator(n, i)} for n in lengths for i in range(examples_per_length)]

def length_generalization_report(rows):
    grouped={}
    for row in rows: grouped.setdefault(int(row["length"]), []).append(bool(row["success"]))
    curve={n:sum(v)/len(v) for n,v in sorted(grouped.items())}
    train_max=max((int(r.get("train_max",0)) for r in rows), default=0)
    beyond=[v for n,v in curve.items() if n>train_max]
    return {"schema":"rwkv-lab.tool-length-generalization.v1", "curve":curve,
            "train_max":train_max, "extrapolation_success":sum(beyond)/len(beyond) if beyond else None}

def validate_tool_tape(tape, allowed_tools, max_steps=128):
    if len(tape)>max_steps: return False
    return all(isinstance(s, ToolStep) and s.tool in allowed_tools for s in tape)
