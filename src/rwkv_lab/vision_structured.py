"""Spatial detection and mask supervision for compressed vision prefixes.

The language objective is excellent at teaching output syntax but weak at
continuous geometry.  This module reconstructs a small spatial lattice from
the aligned visual prefix, decodes a fixed set of object queries, and applies
set-matched box/mask losses.  It is deliberately independent of the frozen
vision and language foundations.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel


BOX_RE = re.compile(r"box=\[(\d+),(\d+),(\d+),(\d+)\]")
MASK_RE = re.compile(r"mask16=([^\n]+)")


@dataclass(frozen=True)
class StructuredTarget:
    boxes_xyxy: Tensor
    masks: Tensor

    @property
    def instances(self) -> int:
        return int(self.boxes_xyxy.shape[0])


@dataclass(frozen=True)
class StructuredPrediction:
    object_logits: Tensor
    boxes_cxcywh: Tensor
    mask_logits: Tensor


def parse_mask16(value: str, *, grid: int = 16) -> Tensor:
    mask = torch.zeros((grid, grid), dtype=torch.float32)
    if not value:
        return mask
    for span in value.split("|"):
        try:
            row_text, columns = span.split(":", 1)
            start_text, end_text = columns.split("-", 1)
            row, start, end = int(row_text), int(start_text), int(end_text)
        except (TypeError, ValueError):
            continue
        if 0 <= row < grid:
            start, end = max(0, start), min(grid - 1, end)
            if start <= end:
                mask[row, start:end + 1] = 1
    return mask


def parse_structured_target(text: str, *, device: torch.device | str | None = None,
                            grid: int = 16) -> StructuredTarget:
    boxes, masks = [], []
    for line in str(text).splitlines():
        box_match = BOX_RE.search(line)
        mask_match = MASK_RE.search(line)
        if box_match is None or mask_match is None:
            continue
        values = [max(0, min(999, int(value))) / 999.0
                  for value in box_match.groups()]
        x1, y1, x2, y2 = values
        boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        masks.append(parse_mask16(mask_match.group(1), grid=grid))
    if boxes:
        box_tensor = torch.tensor(boxes, dtype=torch.float32, device=device)
        mask_tensor = torch.stack(masks).to(device=device)
    else:
        box_tensor = torch.empty((0, 4), dtype=torch.float32, device=device)
        mask_tensor = torch.empty((0, grid, grid), dtype=torch.float32,
                                  device=device)
    return StructuredTarget(box_tensor, mask_tensor)


def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    center = (boxes[..., :2] + boxes[..., 2:]) / 2
    size = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0)
    return torch.cat((center, size), dim=-1)


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    half = boxes[..., 2:] / 2
    return torch.cat((boxes[..., :2] - half, boxes[..., :2] + half), dim=-1)


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise generalized IoU for normalized xyxy boxes."""
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(dim=-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(dim=-1)
    union = area1[:, None] + area2[None, :] - intersection
    iou = intersection / union.clamp_min(1e-7)
    enclosing_left_top = torch.minimum(
        boxes1[:, None, :2], boxes2[None, :, :2])
    enclosing_right_bottom = torch.maximum(
        boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing = (enclosing_right_bottom - enclosing_left_top).clamp_min(0).prod(-1)
    return iou - (enclosing - union) / enclosing.clamp_min(1e-7)


def exact_set_match(cost: Tensor) -> Tensor:
    """Exact query/target assignment in O(queries * targets * 2**targets).

    The training corpus contains up to six targets. Enumerating query
    permutations would require 16P6 candidates (5.8 million) per geometry and
    a large persistent tensor. Matching is discrete, so a tiny CPU dynamic
    program gives the exact Hungarian objective without retaining an autograd
    graph or creating a factorial GPU allocation.
    """
    if cost.ndim != 2:
        raise ValueError("matching cost must be [queries,targets]")
    queries, targets = cost.shape
    if targets > queries:
        raise ValueError("target count must fit the query set")
    if targets == 0:
        return torch.empty(0, dtype=torch.long, device=cost.device)
    values = cost.detach().float().cpu().tolist()
    empty = tuple(-1 for _ in range(targets))
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, empty)}
    for query in range(queries):
        updated = dict(states)
        for mask, (total, assignment) in states.items():
            for target in range(targets):
                bit = 1 << target
                if mask & bit:
                    continue
                candidate = total + values[query][target]
                new_mask = mask | bit
                previous = updated.get(new_mask)
                if previous is None or candidate < previous[0]:
                    assigned = list(assignment)
                    assigned[target] = query
                    updated[new_mask] = (candidate, tuple(assigned))
        states = updated
    assignment = states[(1 << targets) - 1][1]
    return torch.tensor(assignment, dtype=torch.long, device=cost.device)


class StructuredSpatialHead(nn.Module):
    """Spatialize compressed tokens, then decode object boxes and 16x16 masks."""

    def __init__(self, input_width: int = 2560, width: int = 256,
                 object_queries: int = 16, grid: int = 16,
                 spatial_layers: int = 2, object_layers: int = 2,
                 heads: int = 8):
        super().__init__()
        if (input_width < 1 or width < 1 or object_queries < 1 or grid < 1
                or spatial_layers < 1 or object_layers < 1 or heads < 1
                or width % heads):
            raise ValueError("invalid structured-head geometry")
        self.input_width = input_width
        self.width = width
        self.object_queries = object_queries
        self.grid = grid
        self.memory_norm = nn.LayerNorm(input_width)
        self.memory_projection = nn.Linear(input_width, width)
        self.task_projection = nn.Linear(input_width, width)
        self.spatial_queries = nn.Parameter(torch.empty(grid * grid, width))
        self.object_query = nn.Parameter(torch.empty(object_queries, width))
        spatial_layer = nn.TransformerDecoderLayer(
            width, heads, dim_feedforward=width * 4, dropout=0.0,
            batch_first=True, norm_first=True, activation="gelu")
        object_layer = nn.TransformerDecoderLayer(
            width, heads, dim_feedforward=width * 4, dropout=0.0,
            batch_first=True, norm_first=True, activation="gelu")
        self.spatial_decoder = nn.TransformerDecoder(
            spatial_layer, spatial_layers, norm=nn.LayerNorm(width))
        self.object_decoder = nn.TransformerDecoder(
            object_layer, object_layers, norm=nn.LayerNorm(width))
        self.objectness = nn.Linear(width, 1)
        self.box = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 4))
        self.mask_embedding = nn.Linear(width, width)
        self.spatial_mask_embedding = nn.Linear(width, width)
        nn.init.normal_(self.spatial_queries, std=0.02)
        nn.init.normal_(self.object_query, std=0.02)
        nn.init.constant_(self.objectness.bias, -2.0)

    def forward(self, memory: Tensor, task_context: Tensor) -> StructuredPrediction:
        if (memory.ndim != 3 or memory.shape[-1] != self.input_width
                or task_context.shape != (memory.shape[0], self.input_width)):
            raise ValueError("structured head input geometry is invalid")
        batch = memory.shape[0]
        memory = self.memory_projection(self.memory_norm(memory))
        context = self.task_projection(task_context).unsqueeze(1)
        spatial = self.spatial_queries.unsqueeze(0).expand(batch, -1, -1)
        # The sequences here are tiny. PyTorch's inference-only fused SDPA
        # kernel has proved unstable for this uncommon 256-query cross-attn
        # shape on the training host, while the backward-selected kernel is
        # fine. The math backend costs little at this size and keeps train/eval
        # behavior identical instead of failing asynchronously at eval.
        with sdpa_kernel(SDPBackend.MATH):
            spatial = self.spatial_decoder(spatial + context, memory)
            objects = self.object_query.unsqueeze(0).expand(batch, -1, -1)
            objects = self.object_decoder(objects + context, spatial)
        object_logits = self.objectness(objects).squeeze(-1)
        boxes = self.box(objects).sigmoid()
        object_masks = self.mask_embedding(objects)
        spatial_masks = self.spatial_mask_embedding(spatial)
        mask_logits = torch.einsum(
            "bqd,bpd->bqp", object_masks, spatial_masks) / math.sqrt(self.width)
        return StructuredPrediction(
            object_logits, boxes,
            mask_logits.reshape(batch, self.object_queries, self.grid, self.grid))


def structured_detection_loss(
        prediction: StructuredPrediction, targets: Sequence[StructuredTarget], *,
        object_weight: float = 2.0, box_l1_weight: float = 5.0,
        giou_weight: float = 2.0, mask_dice_weight: float = 5.0,
        mask_bce_weight: float = 2.0, no_object_weight: float = 0.1,
        ) -> tuple[Tensor, dict[str, Tensor]]:
    batch, queries = prediction.object_logits.shape
    if len(targets) != batch:
        raise ValueError("one structured target is required per prediction")
    object_losses, box_losses, giou_losses = [], [], []
    bce_losses, dice_losses, matched_ious, matched_dice = [], [], [], []
    positive_instances = 0
    for row, target in enumerate(targets):
        count = target.instances
        if count > queries:
            raise ValueError(f"{count} targets exceed {queries} object queries")
        logits = prediction.object_logits[row]
        object_target = torch.zeros_like(logits)
        if count:
            predicted_xyxy = cxcywh_to_xyxy(
                prediction.boxes_cxcywh[row]).clamp(0, 1)
            target_cxcywh = xyxy_to_cxcywh(target.boxes_xyxy)
            pair_l1 = torch.cdist(
                prediction.boxes_cxcywh[row], target_cxcywh, p=1)
            pair_giou = generalized_box_iou(predicted_xyxy, target.boxes_xyxy)
            pred_flat = prediction.mask_logits[row].flatten(1)
            target_flat = target.masks.flatten(1)
            probability = pred_flat.sigmoid()
            pair_dice = 1 - (
                2 * probability @ target_flat.T + 1
            ) / (
                probability.sum(1, keepdim=True)
                + target_flat.sum(1).unsqueeze(0) + 1
            )
            pair_bce = (
                F.softplus(pred_flat).mean(1, keepdim=True)
                - pred_flat @ target_flat.T / pred_flat.shape[1])
            cost = (box_l1_weight * pair_l1
                    + giou_weight * (1 - pair_giou)
                    + mask_dice_weight * pair_dice
                    + mask_bce_weight * pair_bce
                    - object_weight * F.logsigmoid(logits).unsqueeze(1))
            matched = exact_set_match(cost)
            columns = torch.arange(count, device=logits.device)
            object_target[matched] = 1
            box_losses.append(F.l1_loss(
                prediction.boxes_cxcywh[row, matched], target_cxcywh))
            selected_giou = pair_giou[matched, columns]
            giou_losses.append((1 - selected_giou).mean())
            selected_logits = prediction.mask_logits[row, matched]
            bce_losses.append(F.binary_cross_entropy_with_logits(
                selected_logits, target.masks))
            selected_probability = selected_logits.sigmoid().flatten(1)
            selected_target = target.masks.flatten(1)
            dice = (2 * (selected_probability * selected_target).sum(1) + 1) / (
                selected_probability.sum(1) + selected_target.sum(1) + 1)
            dice_losses.append(1 - dice.mean())
            matched_ious.append(selected_giou.clamp(0, 1).mean())
            matched_dice.append(dice.mean())
            positive_instances += count
        weights = torch.full_like(logits, no_object_weight)
        weights[object_target.bool()] = 1
        object_losses.append(F.binary_cross_entropy_with_logits(
            logits, object_target, weight=weights))

    zero = prediction.object_logits.sum() * 0
    def mean(values: list[Tensor]) -> Tensor:
        return torch.stack(values).mean() if values else zero
    object_loss = mean(object_losses)
    box_l1 = mean(box_losses)
    giou = mean(giou_losses)
    mask_bce = mean(bce_losses)
    mask_dice = mean(dice_losses)
    total = (object_weight * object_loss + box_l1_weight * box_l1
             + giou_weight * giou + mask_dice_weight * mask_dice
             + mask_bce_weight * mask_bce)
    metrics = {
        "structured_loss": total.detach(),
        "structured_object_loss": object_loss.detach(),
        "structured_box_l1": box_l1.detach(),
        "structured_giou_loss": giou.detach(),
        "structured_mask_bce": mask_bce.detach(),
        "structured_mask_dice_loss": mask_dice.detach(),
        "structured_box_iou": mean(matched_ious).detach(),
        "structured_mask_dice": mean(matched_dice).detach(),
        "structured_examples": total.new_tensor(batch),
        "structured_positive_instances": total.new_tensor(positive_instances),
    }
    return total, metrics
