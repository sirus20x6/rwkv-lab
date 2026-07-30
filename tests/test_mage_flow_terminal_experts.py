from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from rwkv_lab.mage_flow_terminal_experts import (
    TERMINAL_EXPERT_DEPTH,
    configure_terminal_training_scope,
    initialize_from_residual_expert,
    install_terminal_expert,
    load_terminal_expert,
    route_prompt,
    save_terminal_expert,
    terminal_architecture_report,
    terminal_optimizer_parameter_groups,
)


class TinyFeedForward(nn.Module):
    def __init__(self, dim: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Sequential(nn.Linear(dim, width), nn.GELU()),
            nn.Linear(width, dim),
        )

    def forward(self, inputs):
        return self.net(inputs)


class TinyBlock(nn.Module):
    def __init__(self, dim: int = 4, mlp_width: int = 6):
        super().__init__()
        self.img_mlp = TinyFeedForward(dim, mlp_width)
        self.txt_mlp = TinyFeedForward(dim, mlp_width)
        self.attention_marker = nn.Parameter(torch.zeros(dim, dim))
        self.calls = 0

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        image_rotary_emb,
        txt_cu_lens,
        img_cu_lens,
        joint_attention_kwargs=None,
    ):
        del temb, image_rotary_emb, txt_cu_lens, img_cu_lens
        del joint_attention_kwargs
        self.calls += 1
        return encoder_hidden_states + 1, hidden_states + 1


class TinyPosition(nn.Module):
    def forward(self, _shapes, *, device):
        return torch.zeros(1, device=device)


class TinyTimestep(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps, image):
        del timesteps
        return torch.zeros(image.shape[0], self.dim, device=image.device)


class TinyNormOut(nn.Module):
    def forward(self, image, temb, *, cu_seqlens):
        del temb, cu_seqlens
        return image


class TinyMageFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_dim = 4
        self.pos_embed = TinyPosition()
        self.img_in = nn.Linear(2, 4)
        self.txt_norm = nn.Identity()
        self.txt_in = nn.Linear(3, 4)
        self.time_text_embed = TinyTimestep(4)
        self.transformer_blocks = nn.ModuleList(TinyBlock() for _ in range(12))
        self.norm_out = TinyNormOut()
        self.proj_out = nn.Linear(4, 2)
        self.checkpoint = False
        self.checkpoint_block_indices = None
        self.checkpoint_context_fn = None

    def forward(self, *args, **kwargs):
        raise AssertionError("the released forward must be replaced")


def _forward(model):
    return model(
        img=torch.zeros(1, 5, 2),
        txt=torch.zeros(1, 3, 3),
        timesteps=torch.zeros(1),
        img_shapes=None,
        img_cu_seqlens=None,
        txt_cu_seqlens=None,
    )


def test_prompt_router_always_selects_exactly_one_domain():
    assert route_prompt("35mm photograph of a harbor").domain == "photo"
    assert route_prompt("cel-shaded anime character sheet").domain == "animation"
    assert route_prompt("a red fox in a forest").domain == "photo"
    assert (
        route_prompt("a photograph redrawn as an illustration").domain == "animation"
    )
    assert route_prompt("anime rendered as a realistic photo").domain == "photo"
    assert route_prompt("anything", override="animation").reason == "explicit_override"
    with pytest.raises(ValueError, match="unsupported explicit"):
        route_prompt("anything", override="general")


def test_complete_backbone_executes_before_one_terminal_expert():
    model = TinyMageFlow()
    base_count = sum(parameter.numel() for parameter in model.parameters())
    expected_expert = sum(
        parameter.numel() for block in model.transformer_blocks[-3:]
        for parameter in block.parameters()
    )
    controller = install_terminal_expert(model, "photo")

    assert controller.config.depth == TERMINAL_EXPERT_DEPTH
    assert controller.config.base_parameter_count == base_count
    assert controller.parameter_count() == expected_expert
    assert controller.config.active_parameter_count == base_count + expected_expert
    assert (
        controller.config.total_checkpoint_parameter_count
        == base_count + 2 * expected_expert
    )
    assert len(model.transformer_blocks) == 12
    assert len(model.terminal_expert_blocks) == 3
    assert not any(
        isinstance(module, nn.ModuleDict)
        for module in model.terminal_expert_blocks.modules()
    )

    output = _forward(model)
    assert output.shape == (1, 5, 2)
    assert all(block.calls == 1 for block in model.transformer_blocks)
    assert all(block.calls == 1 for block in model.terminal_expert_blocks)
    assert terminal_architecture_report(controller)["passed"]

    with controller.route("photo"):
        pass
    with (
        pytest.raises(RuntimeError, match="while photo is resident"),
        controller.route("animation"),
    ):
        pass


def test_terminal_checkpoints_swap_one_resident_module(tmp_path):
    photo_model = TinyMageFlow()
    photo = install_terminal_expert(photo_model, "photo")
    with torch.no_grad():
        for parameter in photo.parameters():
            parameter.fill_(1)
    photo_path = tmp_path / "photo.safetensors"
    save_terminal_expert(photo, photo_path)

    animation_model = TinyMageFlow()
    animation = install_terminal_expert(animation_model, "animation")
    with torch.no_grad():
        for parameter in animation.parameters():
            parameter.fill_(2)
    animation_path = tmp_path / "animation.safetensors"
    save_terminal_expert(animation, animation_path)

    original_module_id = id(photo.blocks)
    result = load_terminal_expert(photo, "animation", animation_path)
    assert result["compatible"]
    assert photo.resident_domain == "animation"
    assert id(photo.blocks) == original_module_id
    assert all(
        torch.count_nonzero(parameter != 2).item() == 0
        for parameter in photo.parameters()
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_terminal_expert(photo, "animation", photo_path)


def _write_residual(path: Path, domain: str, *, source_width: int = 8) -> None:
    from safetensors.torch import save_file

    tensors = {}
    for block in (9, 10, 11):
        prefix = f"blocks.{block}.expert."
        # Increasing neuron magnitudes make the selected top-k deterministic.
        neuron = torch.arange(1, source_width + 1, dtype=torch.float32)
        tensors[prefix + "fc1.weight"] = neuron[:, None].repeat(1, 4)
        tensors[prefix + "fc1.bias"] = neuron.clone()
        tensors[prefix + "fc2.weight"] = neuron[None, :].repeat(4, 1)
        tensors[prefix + "fc2.bias"] = torch.ones(4)
        tensors[f"blocks.{block}.scale"] = torch.tensor(0.5)
    save_file(tensors, str(path), metadata={"domain": domain})


def test_residual_migration_seeds_late_image_mlps_only(tmp_path):
    model = TinyMageFlow()
    controller = install_terminal_expert(model, "animation")
    attention_before = [
        block.attention_marker.detach().clone() for block in controller.blocks
    ]
    residual = tmp_path / "residual.safetensors"
    _write_residual(residual, "animation")

    report = initialize_from_residual_expert(controller, residual)
    assert report["domain"] == "animation"
    assert [item["source_residual_block"] for item in report["blocks"]] == [9, 10, 11]
    assert all(item["retained_fraction"] == pytest.approx(0.75) for item in report["blocks"])
    assert all(
        torch.equal(before, block.attention_marker)
        for before, block in zip(attention_before, controller.blocks, strict=True)
    )
    # Six of eight neurons survive: magnitudes 3..8. The output scale is folded
    # into the migrated second projection.
    first_in, first_out = [
        module
        for module in controller.blocks[0].img_mlp.modules()
        if isinstance(module, nn.Linear)
    ]
    assert first_in.weight[:, 0].tolist() == [3, 4, 5, 6, 7, 8]
    assert first_out.weight[0].tolist() == [1.5, 2, 2.5, 3, 3.5, 4]
    assert first_out.bias.tolist() == [0.5] * 4


def test_training_scope_uses_one_expert_and_half_rate_backbone_tail():
    model = TinyMageFlow()
    controller = install_terminal_expert(model, "photo")
    scope = configure_terminal_training_scope(
        model,
        controller,
        train_backbone_final_fraction=1 / 3,
    )
    assert scope["backbone_block_indices"] == [8, 9, 10, 11]
    assert scope["expert_trainable_parameter_count"] == controller.parameter_count()
    assert scope["backbone_trainable_parameter_count"] == sum(
        parameter.numel()
        for block in model.transformer_blocks[8:]
        for parameter in block.parameters()
    )
    assert all(
        not parameter.requires_grad
        for block in model.transformer_blocks[:8]
        for parameter in block.parameters()
    )
    groups = terminal_optimizer_parameter_groups(
        model,
        controller,
        expert_learning_rate=2e-5,
        backbone_learning_rate_multiplier=0.5,
    )
    assert [group["group_name"] for group in groups] == [
        "terminal_expert",
        "shared_backbone",
    ]
    assert groups[0]["lr"] == pytest.approx(2e-5)
    assert groups[1]["lr"] == pytest.approx(1e-5)
