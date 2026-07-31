from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from rwkv_lab.mage_flow_terminal_experts import (
    TERMINAL_EXPERT_DEPTH,
    LightningRMSNorm,
    LightningSwiGLU,
    _factored_mage_block_forward,
    _factored_swiglu_sum,
    configure_terminal_training_scope,
    convert_terminal_path_to_lightning_blocks,
    initialize_from_residual_expert,
    install_terminal_expert,
    load_terminal_expert,
    load_terminal_shared_backbone,
    route_prompt,
    save_terminal_expert,
    terminal_architecture_report,
    terminal_optimizer_parameter_groups,
)
from rwkv_lab.mage_flow_tread_looping import (
    TreadLoopConfig,
    install_tread_factored_looping,
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
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False)
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
    def __init__(self, dim: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, image, temb, *, cu_seqlens):
        del temb, cu_seqlens
        return self.norm(image)


class TinyMageFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_dim = 4
        self.num_attention_heads = 2
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


def test_lightning_block_conversion_covers_backbone_expert_and_output_norm(
    tmp_path,
):
    model = TinyMageFlow()
    controller = install_terminal_expert(model, "animation")
    original_active = controller.config.active_parameter_count

    report = convert_terminal_path_to_lightning_blocks(
        model,
        controller,
        use_swiglu=True,
        use_rmsnorm=True,
    )

    assert report["converted_blocks"] == 15
    assert report["swiglu_hidden_features"] == 4
    assert report["rmsnorm_final_output"]
    assert controller.config.active_parameter_count > original_active
    assert all(
        isinstance(block.img_mlp, LightningSwiGLU)
        and isinstance(block.txt_mlp, LightningSwiGLU)
        and isinstance(block.img_norm1, LightningRMSNorm)
        and isinstance(block.txt_norm2, LightningRMSNorm)
        for block in (*model.transformer_blocks, *controller.blocks)
    )
    assert isinstance(model.norm_out.norm, LightningRMSNorm)
    assert terminal_architecture_report(controller)["passed"]
    assert _forward(model).shape == (1, 5, 2)

    checkpoint = tmp_path / "converted-animation.safetensors"
    save_terminal_expert(controller, checkpoint)
    destination = TinyMageFlow()
    destination_controller = install_terminal_expert(
        destination, "animation"
    )
    convert_terminal_path_to_lightning_blocks(
        destination,
        destination_controller,
        use_swiglu=True,
        use_rmsnorm=True,
    )
    result = load_terminal_expert(
        destination_controller,
        "animation",
        checkpoint,
    )
    assert result["compatible"]


def test_lightning_swiglu_dense_and_factored_execution_are_equivalent():
    torch.manual_seed(4)
    module = LightningSwiGLU(8, 8, 8)
    inputs = torch.randn(2, 3, 8)

    dense = module(inputs)
    factored = _factored_swiglu_sum(
        inputs,
        module,
        factor_count=4,
        active_factors=(0, 1, 2, 3),
    )

    assert torch.allclose(factored, dense, atol=1.0e-6, rtol=1.0e-6)


def test_lightning_swiglu_strictly_loads_legacy_mage_mlp_keys():
    torch.manual_seed(7)
    module = LightningSwiGLU(4, 4, 4)
    legacy = {
        "net.0.proj.weight": torch.randn(6, 4),
        "net.0.proj.bias": torch.randn(6),
        "net.2.weight": torch.randn(4, 6),
        "net.2.bias": torch.randn(4),
    }

    result = module.load_state_dict(legacy, strict=True)

    assert not result.missing_keys
    assert not result.unexpected_keys
    assert torch.equal(module.w12.weight[4:], torch.zeros(4, 4))
    assert torch.equal(module.w12.bias[4:], torch.ones(4))


def test_lightning_rmsnorm_matches_fp32_reference():
    norm = LightningRMSNorm(3, eps=1.0e-6, dtype=torch.float32)
    inputs = torch.tensor([[[1.0, 2.0, 3.0], [-2.0, 0.0, 2.0]]])

    expected = inputs * torch.rsqrt(
        inputs.square().mean(dim=-1, keepdim=True) + 1.0e-6
    )

    assert torch.allclose(norm(inputs), expected)


def test_learned_loops_keep_all_backbone_blocks_and_start_exact(tmp_path):
    baseline = TinyMageFlow()
    install_terminal_expert(baseline, "photo")
    looped = TinyMageFlow()
    controller = install_terminal_expert(looped, "photo")
    looped.load_state_dict(baseline.state_dict(), strict=False)
    loop_config = TreadLoopConfig.combined_training_preset().to_dict()
    loop_config["tread"]["route_fraction"] = 0.0
    loop_controller = install_tread_factored_looping(
        looped,
        TreadLoopConfig.from_dict(loop_config),
    )
    baseline.eval()
    looped.eval()
    assert torch.equal(_forward(baseline), _forward(looped))
    assert all(block.calls == 1 for block in looped.transformer_blocks)
    assert all(block.calls == 1 for block in looped.terminal_expert_blocks)
    looped.train()
    _forward(looped)
    assert all(block.calls == 4 for block in looped.transformer_blocks)
    assert all(block.calls == 4 for block in looped.terminal_expert_blocks)
    report = terminal_architecture_report(controller)
    assert report["base_blocks_execute_before_expert"]
    assert report["independent_backbone_blocks_replaced_by_recurrent_core"] == []
    assert report["independent_backbone_blocks_executed"] == list(range(12))
    configure_terminal_training_scope(
        looped,
        controller,
        train_backbone_final_fraction=1 / 3,
    )
    groups = terminal_optimizer_parameter_groups(
        looped,
        controller,
        expert_learning_rate=2e-5,
        backbone_learning_rate_multiplier=0.5,
    )
    expert_group_ids = {id(parameter) for parameter in groups[0]["params"]}
    backbone_group_ids = {id(parameter) for parameter in groups[1]["params"]}
    assert all(
        id(parameter) in expert_group_ids
        for parameter in loop_controller.expert_parameters()
    )
    assert all(
        id(parameter) in backbone_group_ids
        for parameter in loop_controller.backbone_adapters.parameters()
    )

    with torch.no_grad():
        loop_controller.expert_adapters[0].residual_weight.fill_(0.125)
    checkpoint = tmp_path / "looped-photo.safetensors"
    save_terminal_expert(controller, checkpoint)
    destination = TinyMageFlow()
    destination_controller = install_terminal_expert(destination, "photo")
    destination_loop = install_tread_factored_looping(
        destination,
        TreadLoopConfig.from_dict(loop_config),
    )
    result = load_terminal_expert(
        destination_controller,
        "photo",
        checkpoint,
    )
    assert result["compatible"]
    assert torch.equal(
        destination_loop.expert_adapters[0].residual_weight,
        loop_controller.expert_adapters[0].residual_weight,
    )
    with (
        pytest.raises(RuntimeError, match="while photo is resident"),
        controller.route("animation"),
    ):
        pass


def test_terminal_forward_can_return_a_configured_path_representation():
    model = TinyMageFlow()
    install_terminal_expert(model, "animation")
    model.train()
    model.checkpoint = True
    prediction, hidden = model(
        img=torch.zeros(1, 5, 2),
        txt=torch.zeros(1, 3, 3),
        timesteps=torch.zeros(1),
        img_shapes=None,
        img_cu_seqlens=None,
        txt_cu_seqlens=None,
        # First expert block after all 12 original backbone blocks.
        return_hidden_layer=12,
    )
    assert prediction.shape == (1, 5, 2)
    assert hidden.shape == (1, 5, 4)
    (prediction.float().mean() + hidden.float().mean()).backward()
    assert model.img_in.weight.grad is not None
    with pytest.raises(ValueError, match="15 path blocks|\\[0, 14\\]"):
        model(
            img=torch.zeros(1, 5, 2),
            txt=torch.zeros(1, 3, 3),
            timesteps=torch.zeros(1),
            return_hidden_layer=15,
        )


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


def test_load_shared_backbone_translates_legacy_shared_ffn_keys(tmp_path):
    from safetensors.torch import save_file

    model = TinyMageFlow()
    target_name = "transformer_blocks.8.img_mlp.net.1.weight"
    target = dict(model.named_parameters())[target_name]
    replacement = torch.full_like(target, 0.375)
    checkpoint = tmp_path / "legacy-shared.safetensors"
    save_file(
        {
            target_name.replace(
                ".img_mlp.", ".img_mlp.shared_ffn.", 1
            ): replacement
        },
        str(checkpoint),
    )

    assert load_terminal_shared_backbone(model, checkpoint) == 1
    assert torch.equal(target, replacement)


def test_all_executable_factors_reconstruct_dense_mage_block():
    mage_layers = pytest.importorskip(
        "mage_flow.models.modules.mage_layers",
        reason="isolated Mage-Flow environment is not active",
    )
    backend = pytest.importorskip("mage_flow.models.modules._attn_backend")
    backend.set_attn_backend("sdpa")
    torch.manual_seed(17)
    block = mage_layers.MageFlowTransformerBlock(
        dim=32,
        num_attention_heads=4,
        attention_head_dim=8,
    ).eval()
    image = torch.randn(1, 5, 32)
    text = torch.randn(1, 3, 32)
    temb = torch.randn(1, 32)
    rope = torch.ones(5, 4, dtype=torch.complex64)
    image_lens = torch.tensor([0, 5], dtype=torch.int32)
    text_lens = torch.tensor([0, 3], dtype=torch.int32)

    dense_text, dense_image = block(
        hidden_states=image,
        encoder_hidden_states=text,
        temb=temb,
        image_rotary_emb=rope,
        txt_cu_lens=text_lens,
        img_cu_lens=image_lens,
    )
    factored_text, factored_image = _factored_mage_block_forward(
        block,
        image,
        text,
        temb,
        rope,
        text_lens,
        image_lens,
        factor_count=4,
        active_factors=(0, 1, 2, 3),
    )

    assert torch.allclose(factored_image, dense_image, atol=2e-5, rtol=2e-5)
    assert torch.allclose(factored_text, dense_text, atol=2e-5, rtol=2e-5)
