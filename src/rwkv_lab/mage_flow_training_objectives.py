"""Training-only objectives for the terminal Mage-Flow expert architecture.

The helpers in this module deliberately have no Mage-Flow package dependency so
their numerical contracts can be tested on CPU.  VAE-REPA aligns an
intermediate transformer state with the clean, frozen Mage VAE representation.
Immiscible Diffusion preserves the Gaussian noise marginal by permuting already
sampled noise tensors across compatible examples in an effective batch.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from rwkv_lab.mage_flow_pretrain import rectified_flow_path

FLOW_LOSS_WEIGHTINGS = frozenset({"uniform", "min_snr", "soft_min_snr"})


class VAERepresentationAlignment(nn.Module):
    """Project a Mage-Flow hidden state into its frozen VAE latent space."""

    def __init__(
        self,
        model_dim: int,
        vae_dim: int,
        *,
        hidden_dim: int | None = None,
        smooth_l1_beta: float = 1.0,
        student_normalization: str = "token_rms",
        per_example_loss_cap: float | None = 5.0,
        normalization_epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if model_dim < 1 or vae_dim < 1:
            raise ValueError("REPA model and VAE dimensions must be positive")
        if hidden_dim is not None and hidden_dim < 1:
            raise ValueError("REPA projection hidden dimension must be positive")
        if smooth_l1_beta <= 0:
            raise ValueError("REPA smooth-L1 beta must be positive")
        if student_normalization not in {"none", "token_rms"}:
            raise ValueError("unsupported REPA student normalization")
        if per_example_loss_cap is not None and per_example_loss_cap <= 0:
            raise ValueError("REPA per-example loss cap must be positive")
        if normalization_epsilon <= 0:
            raise ValueError("REPA normalization epsilon must be positive")
        projection_dim = model_dim if hidden_dim is None else hidden_dim
        self.model_dim = int(model_dim)
        self.vae_dim = int(vae_dim)
        self.hidden_dim = int(projection_dim)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.student_normalization = student_normalization
        self.per_example_loss_cap = (
            float(per_example_loss_cap)
            if per_example_loss_cap is not None
            else None
        )
        self.normalization_epsilon = float(normalization_epsilon)
        self.last_metrics: dict[str, torch.Tensor] = {}
        self.projection = nn.Sequential(
            nn.Linear(self.model_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.vae_dim),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        clean_vae_features: torch.Tensor,
        image_lens: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if hidden.ndim != 3 or clean_vae_features.ndim != 3:
            raise ValueError("REPA features must be packed [batch, tokens, channels]")
        if hidden.shape[:2] != clean_vae_features.shape[:2]:
            raise ValueError(
                "REPA hidden and VAE features must have matching packed tokens"
            )
        if hidden.shape[-1] != self.model_dim:
            raise ValueError(
                f"REPA hidden width {hidden.shape[-1]} != {self.model_dim}"
            )
        if clean_vae_features.shape[-1] != self.vae_dim:
            raise ValueError(
                f"REPA VAE width {clean_vae_features.shape[-1]} != {self.vae_dim}"
            )
        if not torch.isfinite(hidden).all():
            raise FloatingPointError("REPA hidden features contain non-finite values")
        if not torch.isfinite(clean_vae_features).all():
            raise FloatingPointError("REPA VAE features contain non-finite values")

        hidden_float = hidden.float()
        token_rms = (
            hidden_float.square().mean(dim=-1, keepdim=True)
            + self.normalization_epsilon
        ).sqrt()
        if self.student_normalization == "token_rms":
            projected_input = hidden_float / token_rms
        else:
            projected_input = hidden_float
        projected = self.projection(projected_input.to(dtype=hidden.dtype)).float()
        target = clean_vae_features.detach().float()
        element_loss = F.smooth_l1_loss(
            projected.float(),
            target,
            beta=self.smooth_l1_beta,
            reduction="none",
        ).mean(dim=-1)

        total_tokens = int(hidden.shape[1])
        if image_lens is None:
            lengths = (total_tokens,)
        else:
            lengths = tuple(int(length) for length in image_lens)
            if not lengths or any(length < 1 for length in lengths):
                raise ValueError("REPA image lengths must be positive")
            if sum(lengths) != total_tokens:
                raise ValueError(
                    "REPA image lengths do not match the packed token count"
                )
        per_example = torch.stack(
            [
                values.mean()
                for values in torch.split(element_loss.reshape(-1), lengths)
            ]
        )
        raw_loss = per_example.mean()
        if self.per_example_loss_cap is None:
            bounded_per_example = per_example
        else:
            cap = self.per_example_loss_cap
            bounded_per_example = cap * torch.tanh(per_example / cap)
        loss = bounded_per_example.mean()
        self.last_metrics = {
            "loss": loss.detach(),
            "raw_loss_mean": raw_loss.detach(),
            "raw_loss_max": per_example.max().detach(),
            "hidden_token_rms_mean": token_rms.mean().detach(),
            "hidden_token_rms_max": token_rms.max().detach(),
            "hidden_abs_max": hidden_float.abs().max().detach(),
            "projected_rms": projected.square().mean().sqrt().detach(),
            "projected_abs_max": projected.abs().max().detach(),
            "target_rms": target.square().mean().sqrt().detach(),
            "target_abs_max": target.abs().max().detach(),
        }
        return loss

    def report(self) -> dict[str, int | float | str]:
        return {
            "method": "vae-repa",
            "model_dim": self.model_dim,
            "vae_dim": self.vae_dim,
            "projection_hidden_dim": self.hidden_dim,
            "smooth_l1_beta": self.smooth_l1_beta,
            "student_normalization": self.student_normalization,
            "per_example_loss_cap": self.per_example_loss_cap,
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in self.parameters()
            ),
        }


def save_repa_projection(module: VAERepresentationAlignment, path: Path) -> Path:
    """Save the training-only projection without polluting inference weights."""
    from safetensors.torch import save_file

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: value.detach().cpu().contiguous()
        for name, value in module.state_dict().items()
    }
    save_file(tensors, str(path))
    return path


def load_repa_projection(module: VAERepresentationAlignment, path: Path) -> int:
    from safetensors.torch import load_file

    path = path.expanduser().resolve()
    tensors = load_file(str(path), device="cpu")
    expected = module.state_dict()
    missing = sorted(set(expected) - set(tensors))
    unexpected = sorted(set(tensors) - set(expected))
    mismatched = sorted(
        name
        for name in set(expected) & set(tensors)
        if expected[name].shape != tensors[name].shape
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "REPA projection checkpoint mismatch: "
            f"missing={missing}, unexpected={unexpected}, shapes={mismatched}"
        )
    module.load_state_dict(tensors, strict=True)
    return len(tensors)


def minimum_cost_assignment(cost: Sequence[Sequence[float]]) -> list[int]:
    """Return row-to-column Hungarian assignment for a square cost matrix."""
    size = len(cost)
    if size == 0:
        return []
    if any(len(row) != size for row in cost):
        raise ValueError("assignment cost matrix must be nonempty and square")

    # Successive shortest augmenting path form of the Hungarian algorithm.
    # Index zero is the algorithm's sentinel column.
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    matched_row = [0] * (size + 1)
    previous_column = [0] * (size + 1)
    for row_index in range(1, size + 1):
        matched_row[0] = row_index
        minimum = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        column = 0
        while True:
            used[column] = True
            active_row = matched_row[column]
            delta = float("inf")
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = (
                    float(cost[active_row - 1][candidate - 1])
                    - u[active_row]
                    - v[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    previous_column[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    u[matched_row[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            prior = previous_column[column]
            matched_row[column] = matched_row[prior]
            column = prior
            if column == 0:
                break

    assignment = [-1] * size
    for column in range(1, size + 1):
        assignment[matched_row[column] - 1] = column - 1
    if sorted(assignment) != list(range(size)):
        raise RuntimeError("Hungarian assignment did not produce a permutation")
    return assignment


def _packed_segments(
    tensor: torch.Tensor,
    lengths: Sequence[int],
) -> list[torch.Tensor]:
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError("packed flow tensors must have shape [1, tokens, channels]")
    if sum(int(length) for length in lengths) != tensor.shape[1]:
        raise ValueError("packed flow lengths do not cover the tensor")
    return list(torch.split(tensor.squeeze(0), [int(length) for length in lengths]))


def _pairwise_squared_distance(
    clean: Sequence[torch.Tensor],
    noise: Sequence[torch.Tensor],
) -> torch.Tensor:
    clean_flat = torch.stack([value.float().flatten() for value in clean])
    noise_flat = torch.stack([value.float().flatten() for value in noise])
    dimension = float(clean_flat.shape[1])
    return (
        clean_flat.square().sum(dim=1, keepdim=True)
        + noise_flat.square().sum(dim=1).unsqueeze(0)
        - 2.0 * clean_flat @ noise_flat.transpose(0, 1)
    ) / dimension


def apply_immiscible_noise_assignment(
    flows: Sequence[dict[str, Any]],
) -> dict[str, int | float]:
    """Assign sampled noises across an accumulation window in-place.

    Assignment groups examples by packed latent shape.  This guarantees that
    each selected noise is a pure permutation of identically distributed
    Gaussian samples and therefore leaves the noise marginal unchanged.
    """
    if not flows:
        raise ValueError("immiscible assignment requires at least one flow")
    records: list[dict[str, Any]] = []
    for flow_index, flow in enumerate(flows):
        clean_segments = _packed_segments(flow["clean"], flow["image_lens"])
        noise_segments = _packed_segments(flow["noise"], flow["image_lens"])
        for record_index, (clean, noise) in enumerate(
            zip(clean_segments, noise_segments, strict=True)
        ):
            records.append(
                {
                    "flow_index": flow_index,
                    "record_index": record_index,
                    "clean": clean,
                    "noise": noise,
                    "shape": tuple(clean.shape),
                }
            )

    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[record["shape"]].append(index)

    selected_noise: list[torch.Tensor | None] = [None] * len(records)
    assignment_cost = 0.0
    identity_cost = 0.0
    reassigned = 0
    assigned_groups = 0
    for indices in groups.values():
        if len(indices) == 1:
            selected_noise[indices[0]] = records[indices[0]]["noise"]
            continue
        clean = [records[index]["clean"] for index in indices]
        noise = [records[index]["noise"] for index in indices]
        costs = _pairwise_squared_distance(clean, noise)
        assignment = minimum_cost_assignment(costs.detach().cpu().tolist())
        assigned_groups += 1
        identity_cost += float(costs.diagonal().sum().item())
        assignment_cost += float(
            sum(costs[row, column].item() for row, column in enumerate(assignment))
        )
        for local_row, local_column in enumerate(assignment):
            selected_noise[indices[local_row]] = noise[local_column]
            reassigned += int(local_row != local_column)

    per_flow: list[list[torch.Tensor | None]] = [
        [None] * len(flow["image_lens"]) for flow in flows
    ]
    for record, noise in zip(records, selected_noise, strict=True):
        if noise is None:
            raise RuntimeError("immiscible assignment left a record unassigned")
        per_flow[record["flow_index"]][record["record_index"]] = noise

    for flow, pieces in zip(flows, per_flow, strict=True):
        if any(piece is None for piece in pieces):
            raise RuntimeError("immiscible flow reconstruction is incomplete")
        assigned = torch.cat(
            [piece for piece in pieces if piece is not None], dim=0
        ).unsqueeze(0)
        flow["noise"] = assigned
        token_timesteps = torch.repeat_interleave(
            flow["timesteps"],
            torch.tensor(flow["image_lens"], device=assigned.device),
        ).view(1, -1, 1)
        noised, velocity = rectified_flow_path(
            flow["clean"], assigned, token_timesteps
        )
        flow["img"] = noised.to(dtype=flow["clean"].dtype)
        flow["velocity"] = velocity

    return {
        "examples": len(records),
        "shape_groups": len(groups),
        "assigned_groups": assigned_groups,
        "reassigned_examples": reassigned,
        "identity_cost": identity_cost,
        "assignment_cost": assignment_cost,
        "cost_reduction": identity_cost - assignment_cost,
    }


def flow_min_snr_weights(
    timesteps: torch.Tensor,
    *,
    weighting: str,
    gamma: float,
) -> torch.Tensor:
    """Return velocity-space weights for MageFlow's linear rectified path.

    MageFlow uses ``z_t = (1 - t) * clean + t * noise`` and predicts
    ``noise - clean``. Translating Min-SNR from clean/noise prediction into
    this velocity parameterization gives
    ``min((1 - t)^2, gamma * t^2)``. The soft form replaces the hard minimum
    with its harmonic interpolation.
    """
    if weighting not in FLOW_LOSS_WEIGHTINGS:
        raise ValueError(f"unsupported flow loss weighting {weighting!r}")
    if gamma <= 0:
        raise ValueError("flow Min-SNR gamma must be positive")
    values = timesteps.float()
    if values.ndim != 1:
        raise ValueError("flow timesteps must be a one-dimensional tensor")
    if weighting == "uniform":
        return torch.ones_like(values)
    signal = (1.0 - values).square()
    noise = values.square()
    if weighting == "min_snr":
        return torch.minimum(signal, gamma * noise)
    return gamma * signal * noise / (signal + gamma * noise).clamp_min(1.0e-12)


def effective_flow_loss_weights(
    timestep_batches: Sequence[torch.Tensor],
    *,
    weighting: str,
    gamma: float,
    normalize: bool,
) -> tuple[list[torch.Tensor], dict[str, float | str]]:
    """Build per-example weights normalized across an optimizer update."""
    if not timestep_batches:
        raise ValueError("flow weighting requires at least one timestep batch")
    raw_batches = [
        flow_min_snr_weights(batch, weighting=weighting, gamma=gamma)
        for batch in timestep_batches
    ]
    raw = torch.cat(raw_batches)
    raw_mean = raw.mean()
    if normalize:
        if not torch.isfinite(raw_mean) or raw_mean <= 0:
            raise RuntimeError("flow loss weights have an invalid mean")
        scale = raw_mean
    else:
        scale = torch.ones((), device=raw.device, dtype=raw.dtype)
    normalized_batches = [batch / scale for batch in raw_batches]
    normalized = torch.cat(normalized_batches)
    return normalized_batches, {
        "weighting": weighting,
        "gamma": float(gamma),
        "raw_mean": float(raw_mean.detach().item()),
        "raw_min": float(raw.detach().min().item()),
        "raw_max": float(raw.detach().max().item()),
        "normalized_mean": float(normalized.detach().mean().item()),
        "normalized_min": float(normalized.detach().min().item()),
        "normalized_max": float(normalized.detach().max().item()),
    }


def rectified_flow_loss_per_example(
    prediction: torch.Tensor,
    target: torch.Tensor,
    image_lens: Sequence[int],
) -> torch.Tensor:
    """Return one velocity MSE per packed image, independent of resolution."""
    if prediction.shape != target.shape:
        raise ValueError("flow prediction and target shapes must match")
    prediction_segments = _packed_segments(prediction, image_lens)
    target_segments = _packed_segments(target, image_lens)
    return torch.stack(
        [
            (predicted.float() - expected.float()).square().mean()
            for predicted, expected in zip(
                prediction_segments, target_segments, strict=True
            )
        ]
    )


def velocity_direction_loss_per_example(
    prediction: torch.Tensor,
    target: torch.Tensor,
    image_lens: Sequence[int],
    *,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Return channel-direction cosine error per packed native-resolution image.

    LightningDiT computes cosine similarity along the latent-channel dimension
    and averages over spatial positions.  MageFlow stores those positions as a
    packed token sequence, so each image segment is reduced independently.
    """

    if prediction.shape != target.shape:
        raise ValueError("velocity prediction and target shapes must match")
    if epsilon <= 0:
        raise ValueError("velocity direction epsilon must be positive")
    prediction_segments = _packed_segments(prediction, image_lens)
    target_segments = _packed_segments(target, image_lens)
    return torch.stack(
        [
            (
                1.0
                - F.cosine_similarity(
                    predicted.float(),
                    expected.float(),
                    dim=-1,
                    eps=epsilon,
                )
            )
            .clamp(min=0.0, max=2.0)
            .mean()
            for predicted, expected in zip(
                prediction_segments, target_segments, strict=True
            )
        ]
    )


def weighted_velocity_direction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    image_lens: Sequence[int],
    example_weights: torch.Tensor,
    *,
    effective_example_count: int,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weight directional flow error with the effective-batch flow policy."""

    if effective_example_count < 1:
        raise ValueError("effective example count must be positive")
    per_example = velocity_direction_loss_per_example(
        prediction,
        target,
        image_lens,
        epsilon=epsilon,
    )
    weights = example_weights.to(device=per_example.device, dtype=per_example.dtype)
    if weights.shape != per_example.shape:
        raise ValueError(
            "direction weights must contain one value per packed image"
        )
    contribution = (per_example * weights).sum() / float(effective_example_count)
    return contribution, per_example.mean().detach()


def weighted_rectified_flow_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    image_lens: Sequence[int],
    example_weights: torch.Tensor,
    *,
    effective_example_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return this microbatch's contribution to an effective-batch objective."""
    if effective_example_count < 1:
        raise ValueError("effective example count must be positive")
    per_example = rectified_flow_loss_per_example(
        prediction, target, image_lens
    )
    weights = example_weights.to(device=per_example.device, dtype=per_example.dtype)
    if weights.shape != per_example.shape:
        raise ValueError("flow weights must contain one value per packed image")
    contribution = (per_example * weights).sum() / float(effective_example_count)
    return contribution, per_example.mean().detach()


def repa_optimizer_group(
    module: VAERepresentationAlignment,
    *,
    learning_rate: float,
) -> dict[str, Any]:
    if learning_rate <= 0:
        raise ValueError("REPA learning rate must be positive")
    parameters = [
        parameter for parameter in module.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("REPA projection contains no trainable parameters")
    return {
        "params": parameters,
        "lr": learning_rate,
        "initial_lr": learning_rate,
        "group_name": "vae_repa_projection",
    }
