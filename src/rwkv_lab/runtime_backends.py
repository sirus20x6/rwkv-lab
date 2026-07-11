"""Common qualification receipts for Albatross, vLLM, Tenstorrent, and ROSA-JAX.

Sources: https://github.com/BlinkDL/Albatross, https://github.com/marty1885/ttWKV7,
https://github.com/ytfh44/rosa_gpu_jax. Backends supply inert callables; this module never loads code.
"""
from __future__ import annotations
from rwkv_lab.kernel_candidates import qualify_kernel_candidate

KNOWN_BACKENDS={"albatross":"BlinkDL/Albatross","vllm":"vllm-project/vllm",
                "tenstorrent":"marty1885/ttWKV7","rosa-jax":"ytfh44/rosa_gpu_jax"}

def qualify_runtime_backend(name, reference, candidate, probes, **kwargs):
    if name not in KNOWN_BACKENDS: raise ValueError("unknown runtime backend")
    report=qualify_kernel_candidate(reference,candidate,probes,source=KNOWN_BACKENDS[name],**kwargs)
    report.update({"schema":"rwkv-lab.runtime-backend.v1","backend":name})
    return report
