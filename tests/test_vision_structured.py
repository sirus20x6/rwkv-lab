import torch

from rwkv_lab.vision_structured import (
    StructuredSpatialHead,
    exact_set_match,
    parse_structured_target,
    structured_detection_loss,
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
    loss.backward()
    assert memory.grad is not None
    assert torch.isfinite(memory.grad).all()


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
