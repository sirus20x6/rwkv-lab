"""Full-model execution-plan qualification inspired by Megakernels and TileRT.

https://github.com/HazyResearch/Megakernels/tree/throughput and
https://github.com/tile-ai/TileRT; community lead:
https://discord.com/channels/992359628979568762/1084547464452907088/1507270209910734878
"""
from __future__ import annotations
from rwkv_lab.kernel_candidates import qualify_kernel_candidate

def qualify_execution_plan(reference, candidate, probes, *, operators_before, operators_after,
                           source="external-plan", **kwargs):
    report=qualify_kernel_candidate(reference,candidate,probes,source=source,**kwargs)
    report.update({"schema":"rwkv-lab.execution-plan.v1","operators_before":operators_before,
                   "operators_after":operators_after,
                   "launch_reduction":operators_before-operators_after})
    report["adopted"] = bool(report["adopted"] and operators_after < operators_before)
    return report
