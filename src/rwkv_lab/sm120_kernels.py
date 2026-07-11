"""RTX PRO 6000 / compute-capability 12.0 megakernel helpers.

The scheduler and occupancy choices follow NVIDIA's Blackwell tuning guidance
(48 resident warps and 128 KiB shared memory per SM on compute capability 12.0):
https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html

CUTLASS projection discovery uses NVIDIA's Operator API and deliberately
rejects architecture-portable SM80 fallbacks.  A kernel is called ``native``
only when its metadata advertises SM120 support:
https://docs.nvidia.com/cutlass/latest/media/docs/operators/overview.html

The optional L2 window is the CUDA access-policy mechanism documented here:
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#l2-access-properties
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = hasattr(torch.library, "triton_op")
except Exception:  # pragma: no cover - environment dependent
    triton = tl = None
    _HAVE_TRITON = False


@dataclass(frozen=True)
class SM120Profile:
    available: bool
    reason: str
    device: str = ""
    compute_capability: tuple[int, int] = (0, 0)
    sms: int = 0
    max_warps_per_sm: int = 0
    shared_memory_per_sm: int = 0
    l2_bytes: int = 0
    max_persisting_l2_bytes: int = 0


def sm120_profile(device: torch.device | str = "cuda") -> SM120Profile:
    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        return SM120Profile(False, "CUDA is unavailable")
    index = selected.index if selected.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    cc = (props.major, props.minor)
    if cc != (12, 0):
        return SM120Profile(False, f"compute capability {cc[0]}.{cc[1]} is not SM120",
                            props.name, cc, props.multi_processor_count)
    persisting = 0
    try:
        from cuda.bindings import runtime as cudart
        error, persisting = cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrMaxPersistingL2CacheSize, index)
        if int(error) != 0:
            persisting = 0
    except Exception:
        pass
    return SM120Profile(
        True, "native SM120", props.name, cc, props.multi_processor_count,
        48, int(getattr(props, "shared_memory_per_multiprocessor", 0)),
        int(getattr(props, "L2_cache_size", 0)), int(persisting),
    )


def cutlass_sm120_status() -> dict[str, Any]:
    """Report whether the installed Operator API contains native SM120 GEMMs.

    CUTLASS 0.1.0 can accept ``target_sm='120'`` while returning portable SM80
    operators.  Checking metadata prevents that useful fallback from being
    mislabeled as a Blackwell projection backend.
    """
    report: dict[str, Any] = {
        "available": False, "native_operators": 0,
        "source": "https://docs.nvidia.com/cutlass/latest/media/docs/operators/overview.html",
    }
    try:
        import cutlass
        import cutlass.operators as operators
        all_ops = operators.get_operators(target_sm="120")
        native = []
        for operator in all_ops:
            targets = getattr(operator.metadata, "supported_targets", ())
            if any(getattr(target, "cc", None) == 120 for target in targets):
                native.append(operator)
        report.update({
            "package_version": getattr(operators, "__version__", "unknown"),
            "cutlass_version": getattr(cutlass, "__version__", "unknown"),
            "native_operators": len(native),
            "available": bool(native),
            "reason": ("native SM120 operators discovered" if native else
                       "installed registry exposes no native SM120 operator"),
            "operator_names": [op.metadata.operator_name for op in native[:32]],
        })
    except Exception as exc:
        report["reason"] = f"CUTLASS Operator API unavailable: {exc!r}"
    return report


class SM120CutlassProjection:
    """Prepared native-CUTLASS projection, unavailable unless metadata says SM120.

    The adapter is intentionally outside ``torch.compile``; its compiled launch
    is suitable for eager qualification and CUDA Graph capture. Weights use the
    Operator API's contiguous ``[K,N]`` layout rather than ``nn.Linear``'s
    ``[N,K]`` layout.
    """

    def __init__(self, weight: torch.Tensor, *, bias: torch.Tensor | None = None):
        if weight.ndim != 2 or not weight.is_cuda:
            raise ValueError("CUTLASS projection weight must be a CUDA [N,K] tensor")
        if sm120_profile(weight.device).available is False:
            raise RuntimeError("native CUTLASS projection requires SM120")
        self.weight = weight.detach().t().contiguous()
        self.bias = bias
        self.operator = None
        self.artifact = None
        self.name = ""

    def _arguments(self, x: torch.Tensor, output: torch.Tensor):
        import cutlass
        import cutlass.operators as operators
        accumulator = cutlass.Float32
        return operators.GemmArguments(x.reshape(-1, x.shape[-1]), self.weight,
                                       output.reshape(-1, output.shape[-1]),
                                       accumulator)

    def prepare(self, sample: torch.Tensor) -> None:
        import cutlass.operators as operators
        output = torch.empty((*sample.shape[:-1], self.weight.shape[1]),
                             device=sample.device, dtype=sample.dtype)
        args = self._arguments(sample, output)
        candidates = operators.get_operators(args=args, target_sm="120")
        native = [op for op in candidates if any(
            getattr(target, "cc", None) == 120
            for target in getattr(op.metadata, "supported_targets", ()))]
        if not native:
            raise RuntimeError("CUTLASS registry has no native SM120 operator for this projection")
        # Operator order is deterministic. Full production qualification still
        # compares this choice with cuBLAS before setting any model adoption flag.
        self.operator = native[0]
        self.artifact = self.operator.compile(args, target_sm="120")
        self.name = self.operator.metadata.operator_name

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.operator is None:
            self.prepare(x)
        output = torch.empty((*x.shape[:-1], self.weight.shape[1]),
                             device=x.device, dtype=x.dtype)
        args = self._arguments(x, output)
        self.operator.run(args, compiled_artifact=self.artifact,
                          stream=torch.cuda.current_stream(x.device))
        if self.bias is not None:
            output.add_(self.bias)
        return output


def occupancy_plan(tasks: int, device: torch.device | str = "cuda") -> dict[str, Any]:
    """Create the fixed SM120 launch plan used by the persistent scheduler."""
    profile = sm120_profile(device)
    sms = profile.sms if profile.available else 0
    programs = min(max(int(tasks), 0), sms) if sms else 0
    return {
        "available": profile.available,
        "tasks": int(tasks), "resident_programs": programs,
        "tasks_per_program": ((int(tasks) + programs - 1) // programs if programs else 0),
        "block_v": 8, "warps_per_program": 2,
        "max_warps_per_sm": profile.max_warps_per_sm,
        "sms": sms,
        "source": "https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html",
    }


if _HAVE_TRITON:
    @triton.jit
    def _persistent_rwkv7_state_kernel(
        state_ptr, r_ptr, gk_ptr, k_ptr, v_ptr, a_ptr, b_ptr, out_ptr,
        B: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, V_BLOCKS: tl.constexpr,
        TASKS: tl.constexpr,
    ):
        """One resident CTA consumes multiple disjoint state tiles.

        Capping the grid at the physical SM count avoids a long tail of tiny
        launches on the 188-SM RTX PRO 6000. Each loop iteration owns a unique
        ``(batch_head, value_tile)`` task, so the in-place update needs no atomics.
        """
        task = tl.program_id(0)
        stride = tl.num_programs(0)
        while task < TASKS:
            bh = task // V_BLOCKS
            v_block = task - bh * V_BLOCKS
            offs_k = tl.arange(0, BLOCK_K)
            offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
            mask_k = offs_k < K
            mask_v = offs_v < V
            state_offsets = bh * K * V + offs_k[:, None] * V + offs_v[None, :]
            matrix = tl.load(state_ptr + state_offsets,
                             mask=mask_k[:, None] & mask_v[None, :],
                             other=0.0).to(tl.float32)
            vec = bh * K + offs_k
            r = tl.load(r_ptr + vec, mask=mask_k, other=0.0).to(tl.float32)
            decay = tl.load(gk_ptr + vec, mask=mask_k, other=0.0).to(tl.float32)
            key = tl.load(k_ptr + vec, mask=mask_k, other=0.0).to(tl.float32)
            remove_a = tl.load(a_ptr + vec, mask=mask_k, other=0.0).to(tl.float32)
            remove_b = tl.load(b_ptr + vec, mask=mask_k, other=0.0).to(tl.float32)
            value = tl.load(v_ptr + bh * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            removal = tl.sum(matrix * remove_a[:, None], axis=0)
            updated = matrix * tl.exp(decay)[:, None]
            updated += remove_b[:, None] * removal[None, :]
            updated += key[:, None] * value[None, :]
            result = tl.sum(updated * r[:, None], axis=0)
            tl.store(state_ptr + state_offsets, updated,
                     mask=mask_k[:, None] & mask_v[None, :])
            tl.store(out_ptr + bh * V + offs_v, result, mask=mask_v)
            task += stride

    @torch.library.triton_op(
        "rwkv_lab::sm120_persistent_rwkv7_state_", mutates_args={"state"})
    def _persistent_rwkv7_state_op(
        state: torch.Tensor, r: torch.Tensor, gk: torch.Tensor,
        key: torch.Tensor, value: torch.Tensor, remove_a: torch.Tensor,
        remove_b: torch.Tensor, resident_programs: int,
    ) -> torch.Tensor:
        bsz, _, heads, width = r.shape
        value_width = value.shape[-1]
        block_v = 8
        tasks = bsz * heads * triton.cdiv(value_width, block_v)
        output = torch.empty_like(value)
        torch.library.wrap_triton(_persistent_rwkv7_state_kernel)[(resident_programs,)](
            state, r, gk, key, value, remove_a, remove_b, output,
            B=bsz, H=heads, K=width, V=value_width,
            BLOCK_K=triton.next_power_of_2(width), BLOCK_V=block_v,
            V_BLOCKS=triton.cdiv(value_width, block_v), TASKS=tasks,
            num_warps=2,
        )
        return output


def persistent_rwkv7_state_step(
    state: torch.Tensor, r: torch.Tensor, gk: torch.Tensor, key: torch.Tensor,
    value: torch.Tensor, remove_a: torch.Tensor, remove_b: torch.Tensor,
) -> torch.Tensor:
    """Launch the 188-SM persistent state scheduler after external qualification."""
    profile = sm120_profile(r.device)
    if not (_HAVE_TRITON and profile.available):
        raise RuntimeError(profile.reason if not profile.available else "Triton unavailable")
    tasks = r.shape[0] * r.shape[2] * ((value.shape[-1] + 7) // 8)
    programs = min(tasks, profile.sms)
    return _persistent_rwkv7_state_op(
        state, r, gk, key, value, remove_a, remove_b, programs)


@torch.no_grad()
def qualify_persistent_state(
    r: torch.Tensor, gk: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    remove_a: torch.Tensor, remove_b: torch.Tensor, state: torch.Tensor, *,
    repeats: int = 20, minimum_speedup: float = 1.0,
) -> dict[str, Any]:
    """Parity- and latency-gate the SM-count-capped scheduler."""
    from rwkv_lab.megakernel import rwkv7_recurrent_step

    profile = sm120_profile(r.device)
    tasks = r.shape[0] * r.shape[2] * ((value.shape[-1] + 7) // 8)
    report: dict[str, Any] = {
        "schema": "rwkv-lab.sm120-persistent-state-qualification.v1",
        "available": bool(profile.available and _HAVE_TRITON),
        "adopted": False, "profile": asdict(profile),
        "occupancy": occupancy_plan(tasks, r.device),
        "minimum_speedup": minimum_speedup,
        "source": "https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html",
    }
    if not report["available"]:
        report["reason"] = profile.reason if not profile.available else "Triton unavailable"
        return report

    def baseline():
        return rwkv7_recurrent_step(
            r, gk, key, value, remove_a, remove_b, state, inplace=False)

    def candidate():
        candidate_state = state.clone()
        return (persistent_rwkv7_state_step(
            candidate_state, r, gk, key, value, remove_a, remove_b),
                candidate_state)

    expected_out, expected_state = baseline()
    actual_out, actual_state = candidate()
    atol = 5e-3 if r.dtype in (torch.float16, torch.bfloat16) else 2e-5
    parity = bool(torch.allclose(actual_out, expected_out, atol=atol, rtol=3e-3)
                  and torch.allclose(actual_state, expected_state, atol=atol, rtol=3e-3))

    def median_us(fn) -> float:
        for _ in range(3):
            fn()
        torch.cuda.synchronize(r.device)
        samples = []
        for _ in range(max(3, repeats)):
            begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
            begin.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) * 1000.0)
        return sorted(samples)[len(samples) // 2]

    baseline_us = median_us(baseline)
    candidate_us = median_us(candidate)
    speedup = baseline_us / max(candidate_us, 1e-12)
    report.update({"parity": parity, "baseline_median_us": baseline_us,
                   "candidate_median_us": candidate_us, "speedup": speedup,
                   "adopted": bool(parity and speedup >= minimum_speedup)})
    return report


class L2PersistenceController:
    """Own one CUDA stream access-policy window and restore it on clear()."""

    def __init__(self, device: torch.device | str = "cuda"):
        selected = torch.device(device)
        if selected.type == "cuda" and selected.index is None and torch.cuda.is_available():
            selected = torch.device("cuda", torch.cuda.current_device())
        self.device = selected
        self.profile = sm120_profile(self.device)
        self.active = False
        self.window_bytes = 0
        self.tensor_name = ""

    @staticmethod
    def _check(result, operation: str) -> None:
        error = result[0] if isinstance(result, tuple) else result
        if int(error) != 0:
            raise RuntimeError(f"{operation} failed with CUDA error {int(error)}")

    def apply(self, tensor: torch.Tensor, *, name: str = "hot_weights",
              stream: torch.cuda.Stream | None = None) -> dict[str, Any]:
        if not self.profile.available:
            return {"available": False, "adopted": False, "reason": self.profile.reason}
        if not tensor.is_cuda or tensor.device != self.device:
            raise ValueError("L2 persistence tensor must be on the controller CUDA device")
        limit = min(self.profile.max_persisting_l2_bytes,
                    self.profile.l2_bytes * 3 // 4)
        window_bytes = min(tensor.numel() * tensor.element_size(), limit)
        if window_bytes <= 0:
            return {"available": False, "adopted": False,
                    "reason": "cuda.bindings unavailable or device reports no persisting-L2 capacity"}
        from cuda.bindings import runtime as cudart
        self._check(cudart.cudaDeviceSetLimit(
            cudart.cudaLimit.cudaLimitPersistingL2CacheSize, limit),
            "cudaDeviceSetLimit")
        window = cudart.cudaAccessPolicyWindow()
        window.base_ptr = tensor.data_ptr()
        window.num_bytes = window_bytes
        window.hitRatio = min(1.0, limit / window_bytes)
        window.hitProp = cudart.cudaAccessProperty.cudaAccessPropertyPersisting
        window.missProp = cudart.cudaAccessProperty.cudaAccessPropertyStreaming
        value = cudart.cudaStreamAttrValue()
        value.accessPolicyWindow = window
        stream = stream or torch.cuda.current_stream(self.device)
        self._check(cudart.cudaStreamSetAttribute(
            stream.cuda_stream,
            cudart.cudaStreamAttrID.cudaLaunchAttributeAccessPolicyWindow,
            value), "cudaStreamSetAttribute")
        self.active, self.window_bytes, self.tensor_name = True, window_bytes, name
        self._tensor = tensor
        return {"available": True, "adopted": True, "tensor": name,
                "window_bytes": window_bytes, "set_aside_bytes": limit,
                "hit_ratio": window.hitRatio}

    def apply_to_stream(self, stream: torch.cuda.Stream) -> None:
        """Reapply an adopted window to a plan's private capture stream."""
        if not self.active:
            return
        tensor = getattr(self, "_tensor", None)
        if tensor is not None:
            self.apply(tensor, name=self.tensor_name, stream=stream)

    def clear(self) -> None:
        if not self.active:
            return
        from cuda.bindings import runtime as cudart
        window = cudart.cudaAccessPolicyWindow()
        window.base_ptr, window.num_bytes = 0, 0
        window.hitRatio = 0.0
        window.hitProp = cudart.cudaAccessProperty.cudaAccessPropertyNormal
        window.missProp = cudart.cudaAccessProperty.cudaAccessPropertyNormal
        value = cudart.cudaStreamAttrValue()
        value.accessPolicyWindow = window
        cudart.cudaStreamSetAttribute(
            torch.cuda.current_stream(self.device).cuda_stream,
            cudart.cudaStreamAttrID.cudaLaunchAttributeAccessPolicyWindow, value)
        cudart.cudaCtxResetPersistingL2Cache()
        self.active = False
        self._tensor = None

    def receipt(self) -> dict[str, Any]:
        return {"schema": "rwkv-lab.sm120-l2-persistence.v1",
                "active": self.active, "tensor": self.tensor_name,
                "window_bytes": self.window_bytes,
                "profile": asdict(self.profile)}
