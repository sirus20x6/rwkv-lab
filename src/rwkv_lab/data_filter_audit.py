"""Compute-aware raw-vs-filtered pretraining audit.

Mohri et al. (2026), https://arxiv.org/abs/2605.19407, and Nait Saada et al.
(2026), https://machinelearning.apple.com/research/data-quality-illusion.
Community lead: https://discord.com/channels/992359628979568762/1426889957221466153/1513070964446068769
"""
from __future__ import annotations
import math

def filter_regime_report(runs):
    """Compare matched model/data/compute cells; never crown a filter across unmatched budgets."""
    cells={}
    for r in runs:
        key=(r["parameters"],r["tokens"],r["compute"]); cells.setdefault(key,{})[r["filter"]]=r["score"]
    comparisons=[]
    for key, arms in cells.items():
        if "none" in arms:
            comparisons += [{"cell":key,"filter":name,"delta_vs_raw":score-arms["none"]}
                            for name,score in arms.items() if name!="none"]
    return {"schema":"rwkv-lab.data-filter-audit.v1", "matched_comparisons":comparisons,
            "passed":bool(comparisons) and all(math.isfinite(x["delta_vs_raw"]) for x in comparisons)}
