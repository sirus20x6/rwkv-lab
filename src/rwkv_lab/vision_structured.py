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
from PIL import Image, ImageDraw
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel


BOX_RE = re.compile(r"box=\[(\d+),(\d+),(\d+),(\d+)\]")
MASK_RE = re.compile(r"mask16=([^\n]+)")
# ``scripts/build_captioning_first_mix.py`` labels COCO instances with their
# real category name but labels LVIS instances positionally ("instance 1" ..
# "instance 6", ordered by descending area).  A positional label carries no
# category information, so it must not partition the matching groups.
POSITIONAL_LABEL_RE = re.compile(r"instance\s*\d+")
POSITIONAL_LABEL = "\x00positional"
# Upper bound on the smaller side of one matching group.  ``exact_set_match``
# is exponential in that dimension; beyond this the metric falls back to greedy
# IoU matching, which is the usual detection-metric assignment anyway.
EXACT_MATCH_LIMIT = 12


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
    mask_shapes: tuple[tuple[int, int], ...] | None = None


def parse_mask16(value: str, *, grid: int = 16, strict: bool = False) -> Tensor:
    """Rasterize ``row:start-end`` spans, tolerating junk unless ``strict``.

    ``MASK_RE`` runs to end of line, so a systematically mangled suffix would
    otherwise be dropped span by span and score as a merely partial mask.
    ``strict=True`` raises instead, letting the caller reject the whole line.
    """
    mask = torch.zeros((grid, grid), dtype=torch.float32)
    if not value:
        if strict:
            raise ValueError("mask16 span list is empty")
        return mask
    for span in value.split("|"):
        try:
            row_text, columns = span.split(":", 1)
            start_text, end_text = columns.split("-", 1)
            row, start, end = int(row_text), int(start_text), int(end_text)
        except (TypeError, ValueError):
            if strict:
                raise ValueError(f"unparseable mask16 span {span!r}") from None
            continue
        if 0 <= row < grid:
            start, end = max(0, start), min(grid - 1, end)
            if start <= end:
                mask[row, start:end + 1] = 1
        elif strict:
            raise ValueError(f"mask16 row {row} is outside the {grid}x{grid} grid")
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


def structured_target_from_row(
        row: dict, *, mask_shape: tuple[int, int],
        device: torch.device | str | None = None) -> StructuredTarget:
    """Rasterize preserved polygons at the visual feature grid.

    New manifests retain normalized COCO/LVIS polygons in
    ``structured_instances``. The textual ``mask16`` target remains the
    conversational output contract, while this path builds dense supervision
    at whatever native grid the visual prefix actually uses. Older manifests
    fall back to nearest-neighbor expansion of their 16x16 masks.
    """
    height, width = map(int, mask_shape)
    if height < 1 or width < 1:
        raise ValueError("structured mask shape must be positive")
    instances = row.get("structured_instances")
    if instances is None:
        target = parse_structured_target(str(row.get("text") or ""))
        masks = target.masks
        if masks.shape[-2:] != (height, width):
            masks = F.interpolate(
                masks.unsqueeze(1), size=(height, width),
                mode="nearest").squeeze(1)
        return StructuredTarget(
            target.boxes_xyxy.to(device=device),
            masks.to(device=device))
    if not isinstance(instances, list):
        raise ValueError("structured_instances must be a list")
    orientation = int(row.get("exif_orientation", 1) or 1)
    if orientation not in range(1, 9):
        orientation = 1

    def orient(x: float, y: float) -> tuple[float, float]:
        transforms = {
            1: (x, y), 2: (1 - x, y), 3: (1 - x, 1 - y),
            4: (x, 1 - y), 5: (y, x), 6: (1 - y, x),
            7: (1 - y, 1 - x), 8: (y, 1 - x),
        }
        return transforms[orientation]

    boxes: list[list[float]] = []
    masks: list[Tensor] = []
    for instance in instances:
        if not isinstance(instance, dict):
            raise ValueError("each structured instance must be an object")
        values = instance.get("box_xyxy")
        polygons = instance.get("polygons")
        if (not isinstance(values, list) or len(values) != 4
                or not isinstance(polygons, list)):
            raise ValueError("structured instance box/polygons are malformed")
        box = [max(0.0, min(1.0, float(value))) for value in values]
        x1, y1, x2, y2 = box
        corners = [
            orient(x1, y1), orient(x2, y1),
            orient(x2, y2), orient(x1, y2)]
        xs, ys = zip(*corners)
        boxes.append([min(xs), min(ys), max(xs), max(ys)])
        canvas = Image.new("1", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        for polygon in polygons:
            if not isinstance(polygon, list) or len(polygon) < 6:
                continue
            if len(polygon) % 2:
                raise ValueError("structured polygon coordinate count is odd")
            points = []
            for index in range(0, len(polygon), 2):
                x = max(0.0, min(1.0, float(polygon[index])))
                y = max(0.0, min(1.0, float(polygon[index + 1])))
                x, y = orient(x, y)
                points.append((x * width, y * height))
            draw.polygon(points, fill=1)
        masks.append(torch.tensor(
            list(canvas.get_flattened_data()), dtype=torch.float32
        ).reshape(height, width))
    if boxes:
        box_tensor = torch.tensor(boxes, dtype=torch.float32, device=device)
        mask_tensor = torch.stack(masks).to(device=device)
    else:
        box_tensor = torch.empty((0, 4), dtype=torch.float32, device=device)
        mask_tensor = torch.empty(
            (0, height, width), dtype=torch.float32, device=device)
    return StructuredTarget(box_tensor, mask_tensor)


def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    center = (boxes[..., :2] + boxes[..., 2:]) / 2
    size = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0)
    return torch.cat((center, size), dim=-1)


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    half = boxes[..., 2:] / 2
    return torch.cat((boxes[..., :2] - half, boxes[..., :2] + half), dim=-1)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise standard IoU for normalized xyxy boxes."""
    if (boxes1.ndim != 2 or boxes2.ndim != 2
            or boxes1.shape[-1] != 4 or boxes2.shape[-1] != 4):
        raise ValueError("box IoU inputs must be [instances,4]")
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(dim=-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(dim=-1)
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp_min(1e-7)


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


def _parsed_structured_instances(text: str, *, grid: int = 16,
                                 strict_boxes: bool = False,
                                 strict_masks: bool = False,
                                 ) -> list[tuple[str, Tensor, Tensor]]:
    """Parse category, box, and mask from each complete generated line."""
    output = []
    for line in str(text).splitlines():
        box_match = BOX_RE.search(line)
        mask_match = MASK_RE.search(line)
        if box_match is None or mask_match is None:
            continue
        label = line[:box_match.start()].strip().removesuffix(";").strip().casefold()
        values = [int(value) for value in box_match.groups()]
        # ``BOX_RE`` cannot match a sign, so a negative coordinate never reaches
        # here: the line simply fails the regex and is skipped above.
        if strict_boxes and (
                any(value > 999 for value in values)
                or values[2] <= values[0] or values[3] <= values[1]):
            continue
        x1, y1, x2, y2 = [max(0, min(999, value)) / 999.0
                          for value in values]
        box = torch.tensor(
            (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
            dtype=torch.float32)
        try:
            mask = parse_mask16(mask_match.group(1), grid=grid,
                                strict=strict_masks)
        except ValueError:
            continue
        output.append((label, box, mask))
    return output


def canonical_instance_label(label: str) -> str:
    """Collapse positional placeholder labels onto one matching group.

    The LVIS half of the corpus numbers its instances by descending area, so
    every line would otherwise form its own 1x1 group and set matching would
    degenerate into rank-order matching: two area-swapped but geometrically
    correct instances would both score zero.  Real category names (the COCO
    half) keep partitioning the groups.
    """
    return POSITIONAL_LABEL if POSITIONAL_LABEL_RE.fullmatch(label) else label


def structured_generation_metrics(
        pairs: Sequence[tuple[str, str]], *, grid: int = 16,
        iou_threshold: float = 0.5) -> dict[str, float]:
    """Score parsed free-running box/mask text with category-aware matching.

    IoU and Dice use the number of target instances as their denominator, so a
    missing or wrong-category prediction contributes zero. Precision/recall
    require both a category match and standard IoU at ``iou_threshold``.
    Positional placeholder labels carry no category and share one group, so the
    COCO (named) and LVIS (numbered) halves of the corpus stay comparable.
    Precision counts malformed lines in its denominator: a syntax collapse must
    lower P@.5 rather than quietly shrink what it is measured against.
    """
    if not 0 <= iou_threshold <= 1:
        raise ValueError("IoU threshold must lie in [0,1]")
    examples = target_count = prediction_count = invalid_count = 0
    matched_count = true_positives = greedy_groups = 0
    iou_sum = dice_sum = 0.0
    for reference, prediction in pairs:
        examples += 1
        targets = _parsed_structured_instances(reference, grid=grid)
        predicted = _parsed_structured_instances(
            prediction, grid=grid, strict_boxes=True, strict_masks=True)
        target_count += len(targets)
        prediction_count += len(predicted)
        candidates = sum(
            "box=" in line.casefold() or "mask16=" in line.casefold()
            for line in str(prediction).splitlines())
        invalid_count += max(0, candidates - len(predicted))
        target_labels = [canonical_instance_label(item[0]) for item in targets]
        predicted_labels = [
            canonical_instance_label(item[0]) for item in predicted]
        for label in set(target_labels) | set(predicted_labels):
            target_group = [item for item, name in zip(targets, target_labels)
                            if name == label]
            prediction_group = [
                item for item, name in zip(predicted, predicted_labels)
                if name == label]
            if not target_group or not prediction_group:
                continue
            target_boxes = torch.stack([item[1] for item in target_group])
            predicted_boxes = torch.stack([item[1] for item in prediction_group])
            pair_iou = box_iou(predicted_boxes, target_boxes)
            exact = min(len(target_group),
                        len(prediction_group)) <= EXACT_MATCH_LIMIT
            greedy_groups += not exact
            match = exact_set_match if exact else greedy_set_match
            if len(prediction_group) >= len(target_group):
                predicted_indices = match(1 - pair_iou)
                target_indices = torch.arange(len(target_group))
            else:
                target_indices = match(1 - pair_iou.T)
                predicted_indices = torch.arange(len(prediction_group))
            selected_iou = pair_iou[predicted_indices, target_indices]
            matched_count += int(selected_iou.numel())
            true_positives += int((selected_iou >= iou_threshold).sum().item())
            iou_sum += float(selected_iou.sum().item())
            for predicted_index, target_index in zip(
                    predicted_indices.tolist(), target_indices.tolist()):
                predicted_mask = prediction_group[predicted_index][2]
                target_mask = target_group[target_index][2]
                dice_sum += float((
                    (2 * (predicted_mask * target_mask).sum() + 1)
                    / (predicted_mask.sum() + target_mask.sum() + 1)
                ).item())
    scored_predictions = prediction_count + invalid_count
    result = {
        "examples": float(examples),
        "target_instances": float(target_count),
        "predicted_instances": float(prediction_count),
        "matched_instances": float(matched_count),
        "invalid_predictions": float(invalid_count),
        "greedy_matched_groups": float(greedy_groups),
        "true_positives_at_50": float(true_positives),
        "precision_at_50": (
            true_positives / scored_predictions if scored_predictions else 0.0),
        "recall_at_50": true_positives / target_count if target_count else 0.0,
    }
    if target_count:
        result["box_iou"] = iou_sum / target_count
        result["mask_dice"] = dice_sum / target_count
    return result


def exact_set_match(cost: Tensor) -> Tensor:
    """Exact query/target assignment in O(queries * targets * 2**targets).

    The training corpus contains up to six targets. Enumerating query
    permutations would require 16P6 candidates (5.8 million) per geometry and
    a large persistent tensor. Matching is discrete, so a tiny CPU dynamic
    program gives the exact Hungarian objective without retaining an autograd
    graph or creating a factorial GPU allocation.

    The cost is exponential in ``targets``; callers whose target count is not
    bounded by the corpus must keep it at or below :data:`EXACT_MATCH_LIMIT`
    and fall back to :func:`greedy_set_match` beyond it.  The loss path is
    bounded by its own ``count > queries`` check.
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


def greedy_set_match(cost: Tensor) -> Tensor:
    """Bounded stand-in for :func:`exact_set_match` on oversized groups.

    Repeatedly commits the cheapest remaining query/target pair. This is the
    assignment ordinary detection metrics use, and unlike the exact program it
    is O(queries * targets) in time with no exponential state set.
    """
    if cost.ndim != 2:
        raise ValueError("matching cost must be [queries,targets]")
    queries, targets = cost.shape
    if targets > queries:
        raise ValueError("target count must fit the query set")
    assignment = [-1] * targets
    if targets == 0:
        return torch.empty(0, dtype=torch.long, device=cost.device)
    remaining = cost.detach().float().cpu().clone()
    for _ in range(targets):
        flat = int(torch.argmin(remaining).item())
        query, target = divmod(flat, targets)
        assignment[target] = query
        remaining[query, :] = float("inf")
        remaining[:, target] = float("inf")
    return torch.tensor(assignment, dtype=torch.long, device=cost.device)


class StructuredSpatialHead(nn.Module):
    """Decode objects and masks without collapsing native visual cells.

    The learned 16x16 spatial queries remain the object-reasoning canvas. When
    ``mask_shapes`` is supplied, however, masks are dotted directly against the
    projected visual memory and retain every input feature cell. Omitting
    ``mask_shapes`` preserves the legacy 16x16 path for old callers.
    """

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

    def forward(
            self, memory: Tensor, task_context: Tensor, *,
            mask_shapes: Sequence[tuple[int, int]] | None = None,
            ) -> StructuredPrediction:
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
        if mask_shapes is None:
            mask_memory = self.spatial_mask_embedding(spatial)
            output_shapes = None
        else:
            output_shapes = tuple(
                (int(height), int(width)) for height, width in mask_shapes)
            if (len(output_shapes) != batch
                    or any(height < 1 or width < 1
                           or height * width != memory.shape[1]
                           for height, width in output_shapes)):
                raise ValueError(
                    "each native mask shape must match the visual memory")
            # Reuse the checkpoint-compatible mask projection. No new
            # parameters are needed: only the canvas changes from 256 learned
            # slots to every native visual feature cell.
            mask_memory = self.spatial_mask_embedding(memory)
        mask_logits = torch.einsum(
            "bqd,bpd->bqp", object_masks, mask_memory) / math.sqrt(self.width)
        if output_shapes is None:
            mask_logits = mask_logits.reshape(
                batch, self.object_queries, self.grid, self.grid)
        return StructuredPrediction(
            object_logits, boxes, mask_logits, output_shapes)


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
    bce_losses, dice_losses = [], []
    matched_ious, matched_gious, matched_dice = [], [], []
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
            pair_iou = box_iou(predicted_xyxy, target.boxes_xyxy)
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
                selected_logits.flatten(1), target.masks.flatten(1)))
            selected_probability = selected_logits.sigmoid().flatten(1)
            selected_target = target.masks.flatten(1)
            dice = (2 * (selected_probability * selected_target).sum(1) + 1) / (
                selected_probability.sum(1) + selected_target.sum(1) + 1)
            dice_losses.append(1 - dice.mean())
            matched_ious.append(pair_iou[matched, columns])
            matched_gious.append(selected_giou)
            matched_dice.append(dice)
            positive_instances += count
        weights = torch.full_like(logits, no_object_weight)
        weights[object_target.bool()] = 1
        object_losses.append(F.binary_cross_entropy_with_logits(
            logits, object_target, weight=weights))

    zero = prediction.object_logits.sum() * 0
    def mean(values: list[Tensor]) -> Tensor:
        return torch.stack(values).mean() if values else zero
    def instance_mean(values: list[Tensor]) -> Tensor:
        return torch.cat(values).mean() if values else zero
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
        # Renamed from ``structured_box_iou``: that key carried a per-example
        # mean of clamped GIoU. This is a per-instance mean of true IoU, which
        # is >= the old value, so reusing the name would silently rebaseline a
        # continuous dashboard series at the commit that changed it.
        "structured_box_iou_instance": instance_mean(matched_ious).detach(),
        "structured_box_giou": instance_mean(matched_gious).detach(),
        "structured_mask_dice": instance_mean(matched_dice).detach(),
        "structured_examples": total.new_tensor(batch),
        "structured_positive_instances": total.new_tensor(positive_instances),
    }
    return total, metrics


def mask_row_spans(mask: Tensor) -> str:
    """Serialize a binary mask at its native shape for JSON/inference output."""
    if mask.ndim != 2:
        raise ValueError("mask row spans require a 2-D mask")
    rows = []
    binary = mask.detach().to(device="cpu", dtype=torch.bool)
    for y in range(binary.shape[0]):
        x = 0
        while x < binary.shape[1]:
            while x < binary.shape[1] and not bool(binary[y, x]):
                x += 1
            if x == binary.shape[1]:
                break
            start = x
            while x + 1 < binary.shape[1] and bool(binary[y, x + 1]):
                x += 1
            rows.append(f"{y}:{start}-{x}")
            x += 1
    return "|".join(rows)


def structured_prediction_instances(
        prediction: StructuredPrediction, *, row: int = 0,
        object_threshold: float = 0.5,
        mask_threshold: float = 0.5) -> list[dict]:
    """Convert one parallel structured-head result into usable native masks."""
    if not 0 <= row < prediction.object_logits.shape[0]:
        raise IndexError("structured prediction row is out of range")
    if not 0 <= object_threshold <= 1 or not 0 <= mask_threshold <= 1:
        raise ValueError("structured prediction thresholds must be in [0,1]")
    probabilities = prediction.object_logits[row].sigmoid()
    selected = torch.nonzero(
        probabilities >= object_threshold, as_tuple=False).flatten()
    selected = selected[torch.argsort(
        probabilities.index_select(0, selected), descending=True)]
    boxes = cxcywh_to_xyxy(
        prediction.boxes_cxcywh[row]).clamp(0, 1)
    logits = prediction.mask_logits[row]
    if logits.ndim == 2:
        if prediction.mask_shapes is None:
            raise ValueError("flat native mask logits have no recorded shape")
        height, width = prediction.mask_shapes[row]
        logits = logits.reshape(logits.shape[0], height, width)
    elif logits.ndim != 3:
        raise ValueError("structured mask logits have invalid rank")
    output = []
    for query in selected.tolist():
        mask = logits[query].sigmoid() >= mask_threshold
        output.append({
            "query": int(query),
            "objectness": float(probabilities[query].item()),
            "box_xyxy": [float(value) for value in boxes[query].tolist()],
            "mask_shape": [int(mask.shape[0]), int(mask.shape[1])],
            "mask_spans": mask_row_spans(mask),
        })
    return output
