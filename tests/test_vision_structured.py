import pytest
import torch

from rwkv_lab.vision_structured import (
    StructuredPrediction,
    StructuredSpatialHead,
    box_iou,
    exact_set_match,
    generalized_box_iou,
    greedy_set_match,
    parse_structured_target,
    structured_detection_loss,
    structured_generation_metrics,
    structured_prediction_instances,
    structured_target_from_row,
)
from rwkv_lab.vision_train import (
    _load_optimizer_with_appended_group,
    apply_coordinate_token_weights,
    invalid_boundary_token_loss,
    stratified_eval_indices,
    structured_coordinate_token_offsets,
)
from rwkv_lab.generate import WorldVocab


def test_parse_structured_target_and_negative():
    target = parse_structured_target(
        "person; box=[100,200,700,900]; "
        "mask16=2:3-5|3:4-4|3:7-8")
    assert target.boxes_xyxy.shape == (1, 4)
    assert target.masks.shape == (1, 16, 16)
    assert target.masks[0, 2, 3:6].sum() == 3
    assert target.masks[0, 3, 4] == 1
    assert target.masks[0, 3, 7:9].sum() == 2
    negative = parse_structured_target("none")
    assert negative.boxes_xyxy.shape == (0, 4)
    assert negative.masks.shape == (0, 16, 16)


def test_exact_set_match_uses_distinct_queries():
    cost = torch.tensor([[1.0, 1.0], [0.0, 9.0], [9.0, 0.0]])
    assert exact_set_match(cost).tolist() == [1, 2]


def test_standard_iou_is_not_clamped_generalized_iou():
    boxes = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
    targets = torch.tensor([[0.25, 0.25, 0.75, 0.75]])
    torch.testing.assert_close(box_iou(boxes, targets), torch.tensor([[1 / 7]]))


def test_generalized_iou_goes_negative_on_disjoint_boxes():
    boxes = torch.tensor([[0.0, 0.0, 0.2, 0.2]])
    targets = torch.tensor([[0.6, 0.6, 1.0, 1.0]])
    # Areas 0.04 and 0.16, no overlap, enclosing hull is the unit square:
    # GIoU = 0 - (1 - 0.2) / 1.
    torch.testing.assert_close(
        generalized_box_iou(boxes, targets), torch.tensor([[-0.8]]))
    torch.testing.assert_close(box_iou(boxes, targets), torch.tensor([[0.0]]))


def test_generated_structured_metrics_penalize_missing_wrong_and_extra_boxes():
    reference = "\n".join((
        "chair; box=[000,000,499,499]; mask16=0:0-7",
        "chair; box=[500,500,999,999]; mask16=8:8-15",
        "person; box=[100,100,300,300]; mask16=2:2-4",
    ))
    prediction = "\n".join((
        "chair; box=[000,000,499,499]; mask16=0:0-7",
        "chair; box=[700,700,900,900]; mask16=10:10-12",
        "dog; box=[100,100,300,300]; mask16=2:2-4",
        "chair; box=[010,010,020,020]; mask16=0:0-0",
        "broken; box=[999,999,000,000]",
    ))
    metrics = structured_generation_metrics([(reference, prediction)])
    assert metrics["target_instances"] == 3
    assert metrics["predicted_instances"] == 4
    assert metrics["matched_instances"] == 2
    assert metrics["invalid_predictions"] == 1
    assert metrics["true_positives_at_50"] == 1
    # The malformed fifth line stays in the precision denominator.
    assert metrics["precision_at_50"] == 1 / 5
    assert metrics["recall_at_50"] == 1 / 3
    # Exact geometry: the matched pairs are (identical box, IoU 1) and the
    # 200/999-wide box nested in the 499/999-wide target, over three targets.
    assert metrics["box_iou"] == pytest.approx((1 + 200 ** 2 / 499 ** 2) / 3)
    # Dice pairs are 8/8 identical cells and 3 vs 8 disjoint cells.
    assert metrics["mask_dice"] == pytest.approx((1 + 1 / 12) / 3)


def test_positional_instance_labels_do_not_force_rank_order_matching():
    """The LVIS half numbers instances by area; the COCO half names them."""
    reference = "\n".join((
        "instance 1; box=[000,000,499,499]; mask16=0:0-7",
        "instance 2; box=[500,500,999,999]; mask16=8:8-15",
    ))
    # The same two objects, emitted in the other area order.
    swapped = "\n".join((
        "instance 1; box=[500,500,999,999]; mask16=8:8-15",
        "instance 2; box=[000,000,499,499]; mask16=0:0-7",
    ))
    metrics = structured_generation_metrics([(reference, swapped)])
    assert metrics["matched_instances"] == 2
    assert metrics["box_iou"] == pytest.approx(1.0)
    assert metrics["mask_dice"] == pytest.approx(1.0)
    assert metrics["recall_at_50"] == 1.0
    assert metrics["precision_at_50"] == 1.0

    # A prediction numbered past the reference's instance count still matches.
    renumbered = "instance 5; box=[000,000,499,499]; mask16=0:0-7"
    extended = structured_generation_metrics([(reference, renumbered)])
    assert extended["matched_instances"] == 1
    assert extended["true_positives_at_50"] == 1

    # Real category names keep partitioning the groups, so a swap scores zero.
    named_reference = "\n".join((
        "chair; box=[000,000,499,499]; mask16=0:0-7",
        "person; box=[500,500,999,999]; mask16=8:8-15",
    ))
    named_swap = "\n".join((
        "chair; box=[500,500,999,999]; mask16=8:8-15",
        "person; box=[000,000,499,499]; mask16=0:0-7",
    ))
    named = structured_generation_metrics([(named_reference, named_swap)])
    assert named["matched_instances"] == 2
    assert named["true_positives_at_50"] == 0
    assert named["box_iou"] == 0.0


def test_mangled_mask_suffix_is_invalid_not_partially_credited():
    reference = "chair; box=[000,000,499,499]; mask16=0:0-7"
    mangled = "chair; box=[000,000,499,499]; mask16=0:0-7|3:4?9|junk"
    metrics = structured_generation_metrics([(reference, mangled)])
    assert metrics["predicted_instances"] == 0
    assert metrics["invalid_predictions"] == 1
    assert metrics["matched_instances"] == 0
    assert metrics["precision_at_50"] == 0.0
    # A mask row outside the 16x16 grid is mangled too, not a silent no-op.
    off_grid = "chair; box=[000,000,499,499]; mask16=0:0-7|31:1-2"
    assert structured_generation_metrics(
        [(reference, off_grid)])["invalid_predictions"] == 1


def test_generation_matching_stays_bounded_on_oversized_instance_groups():
    def corpus(offset: int) -> str:
        return "\n".join(
            f"instance {index + 1}; box=[{index * 10 + offset:03d},000,"
            f"{index * 10 + offset + 5:03d},005]; mask16=0:{index % 16}-{index % 16}"
            for index in range(14))

    metrics = structured_generation_metrics([(corpus(0), corpus(0))])
    assert metrics["greedy_matched_groups"] == 1
    assert metrics["matched_instances"] == 14
    assert metrics["box_iou"] == pytest.approx(1.0)


def test_greedy_set_match_agrees_with_the_exact_program_when_bounded():
    torch.manual_seed(5)
    cost = torch.tensor([[0.1, 0.9, 0.5], [0.8, 0.2, 0.7],
                         [0.6, 0.4, 0.05], [0.3, 0.3, 0.3]])
    assert greedy_set_match(cost).tolist() == exact_set_match(cost).tolist()


def test_structured_coordinate_offsets_cover_only_box_numbers():
    vocab = WorldVocab()
    text = "café; box=[000,259,159,999]; mask16=0:1-2"
    tokens = vocab.encode(text)
    covered, starts = structured_coordinate_token_offsets(text, tokens, vocab)
    assert len(starts) == 4
    assert vocab.decode([tokens[index] for index in covered]) == "000259159999"
    assert [vocab.decode([tokens[index]]) for index in starts] == [
        "00", "25", "15", "99"]


def test_invalid_boundary_loss_allows_true_edge_and_penalizes_false_edge():
    logits = torch.zeros(2, 1024, requires_grad=True)
    targets = torch.tensor([620, 645])  # true 00-prefix, then a non-edge 25-prefix
    starts = torch.tensor([True, True])
    loss = invalid_boundary_token_loss(
        logits, targets, starts, (620, 719), margin=1.0)
    loss.backward()
    # A real 00-prefix is not treated as its own negative.
    assert logits.grad[0, 620] < 0
    # Both collapse paths are explicitly suppressed for a non-edge target.
    assert logits.grad[1, 620] > 0
    assert logits.grad[1, 719] > 0


def test_coordinate_weights_initialize_when_grounding_weights_are_absent():
    mask = torch.tensor([False, True, False, True])
    weights = apply_coordinate_token_weights(None, mask, 4.0)
    assert weights.tolist() == [1.0, 4.0, 1.0, 4.0]


def test_structured_head_loss_backpropagates():
    torch.manual_seed(1)
    head = StructuredSpatialHead(
        input_width=32, width=32, object_queries=4, grid=16,
        spatial_layers=1, object_layers=1, heads=4)
    memory = torch.randn(2, 12, 32, requires_grad=True)
    context = torch.randn(2, 32)
    targets = [
        parse_structured_target(
            "object; box=[100,200,700,900]; mask16=2:3-5|3:4-8"),
        parse_structured_target("none"),
    ]
    prediction = head(memory, context)
    assert prediction.object_logits.shape == (2, 4)
    assert prediction.boxes_cxcywh.shape == (2, 4, 4)
    assert prediction.mask_logits.shape == (2, 4, 16, 16)
    loss, metrics = structured_detection_loss(prediction, targets)
    assert torch.isfinite(loss)
    assert metrics["structured_positive_instances"] == 1
    # The instance-weighted true-IoU key. The old per-example clamped-GIoU
    # ``structured_box_iou`` is retired rather than silently redefined.
    assert "structured_box_iou" not in metrics
    assert 0 <= metrics["structured_box_iou_instance"] <= 1
    assert metrics["structured_box_giou"] <= metrics["structured_box_iou_instance"]
    loss.backward()
    assert memory.grad is not None
    assert torch.isfinite(memory.grad).all()


def test_native_polygon_target_and_memory_grid_mask_backpropagate():
    torch.manual_seed(7)
    row = {
        "text": "object; box=[250,250,749,749]; mask16=4:4-12",
        "structured_instances": [{
            "box_xyxy": [0.25, 0.25, 0.75, 0.75],
            "polygons": [[0.25, 0.25, 0.75, 0.25,
                          0.75, 0.75, 0.25, 0.75]],
        }],
    }
    target = structured_target_from_row(row, mask_shape=(6, 10))
    assert target.masks.shape == (1, 6, 10)
    assert target.masks.sum() > 0

    head = StructuredSpatialHead(
        input_width=16, width=16, object_queries=3, grid=4,
        spatial_layers=1, object_layers=1, heads=4)
    memory = torch.randn(1, 60, 16, requires_grad=True)
    prediction = head(
        memory, torch.randn(1, 16), mask_shapes=[(6, 10)])
    assert prediction.mask_logits.shape == (1, 3, 60)
    assert prediction.mask_shapes == ((6, 10),)
    loss, _metrics = structured_detection_loss(prediction, [target])
    loss.backward()
    assert memory.grad is not None
    assert torch.isfinite(memory.grad).all()
    # Native masks must supervise the memory canvas, not only the learned
    # spatial-query canvas.
    assert head.spatial_mask_embedding.weight.grad is not None


def test_old_mask16_rows_expand_for_native_head_compatibility():
    target = structured_target_from_row(
        {"text": "object; box=[000,000,999,999]; mask16=0:0-0"},
        mask_shape=(30, 40))
    assert target.masks.shape == (1, 30, 40)
    assert target.masks.sum() > 0


def test_native_structured_prediction_is_exposed_as_parallel_mask_output():
    prediction = StructuredPrediction(
        object_logits=torch.tensor([[5.0, -5.0]]),
        boxes_cxcywh=torch.tensor([[
            [0.5, 0.5, 0.4, 0.2],
            [0.2, 0.2, 0.1, 0.1],
        ]]),
        mask_logits=torch.tensor([[
            [8.0, 8.0, -8.0, -8.0, -8.0, -8.0],
            [-8.0, -8.0, -8.0, -8.0, -8.0, -8.0],
        ]]),
        mask_shapes=((2, 3),),
    )
    instances = structured_prediction_instances(prediction)
    assert len(instances) == 1
    assert instances[0]["mask_shape"] == [2, 3]
    assert instances[0]["mask_spans"] == "0:0-1"
    assert instances[0]["box_xyxy"] == pytest.approx([0.3, 0.4, 0.7, 0.6])


def test_eval_selection_round_robins_caption_ocr_and_structured():
    rows = [
        {"task": "caption"}, {"task": "caption"}, {"task": "caption"},
        {"task": "ocr"}, {"task": "ocr"},
        {"task": "sam_mask"}, {"task": "sam_mask"},
    ]
    selected = stratified_eval_indices(rows, list(range(len(rows))), 6)
    tasks = [rows[index]["task"] for index in selected]
    assert tasks == ["caption", "ocr", "sam_mask"] * 2


def test_optimizer_migration_preserves_old_moments_and_appends_head():
    old_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    old_optimizer = torch.optim.AdamW([
        {"params": [old_parameter], "lr": 1e-3, "name": "projector"},
    ])
    old_parameter.grad = torch.tensor([2.0])
    old_optimizer.step()
    saved = old_optimizer.state_dict()

    restored_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    new_parameter = torch.nn.Parameter(torch.tensor([3.0]))
    optimizer = torch.optim.AdamW([
        {"params": [restored_parameter], "lr": 1e-3, "name": "projector"},
        {"params": [new_parameter], "lr": 2e-3, "name": "structured_head"},
    ])
    _load_optimizer_with_appended_group(
        optimizer, saved, group_name="structured_head")
    state = optimizer.state_dict()
    assert len(state["param_groups"]) == 2
    assert state["param_groups"][1]["name"] == "structured_head"
    assert state["param_groups"][1]["lr"] == 2e-3
    assert state["param_groups"][1]["params"][0] not in state["state"]
    assert torch.equal(
        optimizer.state[restored_parameter]["exp_avg"],
        old_optimizer.state[old_parameter]["exp_avg"],
    )
