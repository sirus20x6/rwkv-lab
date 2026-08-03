import json

import pytest
import torch
from torch import nn

from rwkv_lab.mage_flow_adaptation import (
    CFG_NULL_CONDITION,
    MAGE_FLOW_BASE_REVISION,
    UNCAPTIONED_IMAGE_CONDITION,
    assert_homogeneous_batch,
    audit_domain_rows,
    benchmark_suite,
    canonical_domain_row,
    cfg_null_condition,
    checkpoint_key_report,
    edge_reconstruction_error,
    freeze_for_expert_training,
    homogeneous_domain_batches,
    inject_appearance_experts,
    inspect_safetensors_layout,
    load_appearance_expert,
    load_domain_manifest,
    normalized_ocr_accuracy,
    prepare_domain_manifest,
    reconstruction_psnr,
    reconstruction_ssim,
    save_appearance_expert,
    select_caption,
    temporal_reconstruction_jitter,
    vae_reconstruction_report,
    write_benchmark_suite,
)


def test_fixed_benchmark_covers_generation_editing_and_vae_contract(tmp_path):
    first = benchmark_suite()
    second = benchmark_suite()
    assert first == second
    assert len(first) == 26
    assert len({case.case_id for case in first}) == len(first)
    assert {case.mode for case in first} == {"generation", "editing"}
    categories = {case.category for case in first}
    assert {
        "photorealistic_people",
        "anime_characters",
        "western_animation",
        "typography",
        "spatial_relationships",
        "reference_consistency",
    } <= categories

    output = tmp_path / "benchmark.json"
    payload = write_benchmark_suite(output)
    assert payload["base_revision"] == MAGE_FLOW_BASE_REVISION
    assert "temporal_reconstruction_jitter" in payload["vae_metrics"]
    assert json.loads(output.read_text()) == payload


def test_checkpoint_layout_and_key_report_are_read_only(tmp_path):
    from safetensors.torch import save_file

    checkpoint = tmp_path / "tiny.safetensors"
    save_file(
        {
            "linear.weight": torch.zeros(3, 2),
            "linear.bias": torch.ones(3, dtype=torch.bfloat16),
        },
        str(checkpoint),
    )
    layout = inspect_safetensors_layout([checkpoint])
    assert layout["tensor_count"] == 2
    assert layout["tensors"]["linear.weight"]["shape"] == [3, 2]
    assert layout["tensors"]["linear.bias"]["dtype"] == "BF16"

    report = checkpoint_key_report(
        {"linear.weight", "linear.bias", "norm.weight"},
        layout["tensors"],
    )
    assert report["missing_keys"] == ["norm.weight"]
    assert not report["compatible"]


def test_dependency_free_vae_metrics_cover_fidelity_ocr_and_temporal_jitter():
    reference = torch.zeros(3, 3, 16, 16)
    reconstructed = reference.clone()
    assert reconstruction_psnr(reference, reconstructed) == float("inf")
    assert reconstruction_ssim(reference, reconstructed) == pytest.approx(1.0)
    assert edge_reconstruction_error(reference, reconstructed) == 0
    assert temporal_reconstruction_jitter(reference, reconstructed) == 0
    assert normalized_ocr_accuracy(["North  Star"], ["north star"]) == 1

    reconstructed[1, :, 4:12, 4:12] = 0.5
    report = vae_reconstruction_report(
        reference,
        reconstructed,
        reference_texts=["NORTH STAR", "AUGUST 14", ""],
        reconstructed_texts=["NORTH STAP", "AUGUST 14", ""],
        temporal=True,
        lpips_fn=lambda left, right: (
            (left - right).square().mean(dim=(1, 2, 3), keepdim=True)
        ),
    )
    assert report["psnr"] < float("inf")
    assert 0 < report["ssim"] < 1
    assert report["edge_reconstruction_error"] > 0
    assert report["temporal_reconstruction_jitter"] > 0
    assert 0 < report["ocr_accuracy"] < 1
    assert report["lpips"] > 0


def test_caption_hierarchy_and_cfg_null_are_semantically_distinct():
    caption, kind, is_captioned = select_caption(
        {
            "short_caption": "short",
            "dense_caption": "dense",
            "human_caption": "human",
        }
    )
    assert (caption, kind, is_captioned) == ("human", "human", True)
    caption, kind, is_captioned = select_caption(
        {"tags": ["red hair", "portrait"], "entities": ["person"]}
    )
    assert caption == "red hair, portrait, person"
    assert kind == "tags_entities" and is_captioned

    caption, kind, is_captioned = select_caption({})
    assert caption == UNCAPTIONED_IMAGE_CONDITION
    assert kind == "uncaptioned_image" and not is_captioned
    cfg = cfg_null_condition()
    assert cfg["conditioning_text"] == CFG_NULL_CONDITION
    assert cfg["conditioning_text"] != caption
    assert cfg["conditioning_kind"] == "cfg_null"


def test_domain_canonicalization_restricts_uncaptioned_updates(tmp_path):
    from PIL import Image

    image = tmp_path / "sample.png"
    Image.new("RGB", (640, 480), "blue").save(image)
    photo, reason = canonical_domain_row(
        {"image": image.name, "domain": "photo", "source": "camera-a"},
        data_root=tmp_path,
    )
    assert reason is None
    assert photo["conditioning_text"] == UNCAPTIONED_IMAGE_CONDITION
    assert photo["training_scope"] == "expert_only"
    assert photo["train_width"] % 16 == photo["train_height"] % 16 == 0

    general, reason = canonical_domain_row(
        {"image": image.name, "domain": "general", "source": "anchor"},
        data_root=tmp_path,
    )
    assert general is None and reason == "uncaptioned_general_has_no_expert"

    invalid, reason = canonical_domain_row(
        {"image": image.name, "caption": "An image."},
        data_root=tmp_path,
    )
    assert invalid is None and reason == "invalid_or_missing_domain"


def test_prepare_domain_manifest_deduplicates_content_and_loads_contract(tmp_path):
    from PIL import Image

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (64, 64), "red").save(first)
    second.write_bytes(first.read_bytes())
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "image": first.name,
                        "domain": "photo",
                        "caption": "A red square.",
                        "source": "one",
                    }
                ),
                json.dumps(
                    {
                        "image": second.name,
                        "domain": "animation",
                        "caption": "A red square.",
                        "source": "two",
                    }
                ),
            ]
        )
        + "\n"
    )
    output = tmp_path / "prepared.jsonl"
    report = prepare_domain_manifest(source, output, data_root=tmp_path)
    assert report["counts"] == {
        "cross_domain_duplicate": 1,
        "input": 2,
        "output": 1,
    }
    assert not report["audit"]["passed"]
    assert report["audit"]["input_cross_domain_duplicate_count"] == 1
    rows = load_domain_manifest(output)
    assert len(rows) == 1 and rows[0]["domain"] == "photo"
    assert output.with_suffix(".jsonl.report.json").is_file()


def test_domain_audit_detects_leakage_and_uncaptioned_limit():
    rows = [
        {
            "image_id": "same",
            "domain": "photo",
            "source": "photos",
            "caption": "A subject",
            "conditioning_kind": "human",
            "is_captioned": True,
        },
        {
            "image_id": "same",
            "domain": "animation",
            "source": "drawings",
            "caption": "A subject",
            "conditioning_kind": "uncaptioned_image",
            "is_captioned": False,
        },
    ]
    audit = audit_domain_rows(rows, max_uncaptioned_fraction=0.15)
    assert not audit["passed"]
    assert audit["cross_domain_image_count"] == 1
    assert audit["uncaptioned_fraction"] == 0.5
    assert audit["cross_domain_normalized_caption_count"] == 1


def test_balanced_batches_are_deterministic_homogeneous_and_source_balanced():
    rows = []
    for domain, sources in (
        ("general", ("anchor",)),
        ("photo", ("camera-a", "camera-b")),
        ("animation", ("anime", "western")),
    ):
        for source in sources:
            for item in range(3):
                rows.append({"domain": domain, "source": source, "item": item})

    batches = homogeneous_domain_batches(
        rows, batch_size=2, batch_count=20, seed=7, epoch=2
    )
    assert batches == homogeneous_domain_batches(
        rows, batch_size=2, batch_count=20, seed=7, epoch=2
    )
    domains = [assert_homogeneous_batch(rows, batch) for batch in batches]
    assert domains.count("general") == 2
    assert domains.count("photo") == 9
    assert domains.count("animation") == 9

    for domain in ("photo", "animation"):
        selected_sources = [
            rows[index]["source"]
            for batch, routed_domain in zip(batches, domains, strict=True)
            if routed_domain == domain
            for index in batch
        ]
        counts = {
            source: selected_sources.count(source) for source in set(selected_sources)
        }
        assert max(counts.values()) - min(counts.values()) <= 1


class _FakeBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.img_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, inputs):
        return inputs + self.img_mlp(inputs)


class _FakeTransformer(nn.Module):
    def __init__(self, dim=8, depth=4):
        super().__init__()
        self.inner_dim = dim
        self.transformer_blocks = nn.ModuleList([_FakeBlock(dim) for _ in range(depth)])

    def forward(self, inputs):
        for block in self.transformer_blocks:
            inputs = block(inputs)
        return inputs


def test_expert_injection_preserves_neutral_and_zero_initialized_routes():
    torch.manual_seed(4)
    model = _FakeTransformer()
    inputs = torch.randn(2, 3, 8)
    original = model(inputs)
    controller = inject_appearance_experts(
        model,
        final_block_fraction=0.5,
        expert_parameter_fraction=None,
        expert_width_fraction=0.25,
    )
    assert controller.config.block_indices == (2, 3)
    assert controller.config.expert_hidden_dim == 8

    neutral = model(inputs)
    with controller.route("photo"):
        zero_photo = model(inputs)
    with controller.route("animation"):
        zero_animation = model(inputs)
    assert torch.equal(original, neutral)
    assert torch.equal(original, zero_photo)
    assert torch.equal(original, zero_animation)

    for wrapper in controller.wrappers.values():
        wrapper.experts["photo"].fc2.bias.data.fill_(0.25)
    with controller.route("photo"):
        specialized = model(inputs)
    with controller.route("animation"):
        untouched = model(inputs)
    assert not torch.equal(original, specialized)
    assert torch.equal(original, untouched)
    assert all(
        wrapper.active_domain == "general" for wrapper in controller.wrappers.values()
    )


def test_default_twelve_block_layout_targets_fifteen_percent_per_domain():
    model = _FakeTransformer(depth=12)
    controller = inject_appearance_experts(model, expert_hidden_alignment=1)
    assert controller.config.block_indices == tuple(range(4, 12))
    assert controller.config.requested_parameter_fraction == 0.15
    assert controller.config.actual_parameter_fraction == pytest.approx(0.15, abs=0.01)
    assert controller.config.combined_shared_fraction == pytest.approx(
        1 / 1.3, abs=0.02
    )


def test_released_backbone_target_derives_exact_final_expert_geometry():
    released_parameters = 4_115_745_408
    with torch.device("meta"):
        model = _FakeTransformer(dim=3072, depth=12)
        represented = sum(parameter.numel() for parameter in model.parameters())
        model.parameter_count_padding = nn.Parameter(
            torch.empty(released_parameters - represented)
        )
        controller = inject_appearance_experts(model, expert_dtype=torch.float32)
    assert controller.config.block_indices == tuple(range(4, 12))
    assert controller.config.expert_hidden_dim == 12_544
    assert controller.config.expert_parameter_count == 616_687_624
    assert controller.config.actual_parameter_fraction == pytest.approx(
        0.14983619317203403
    )
    assert controller.config.combined_shared_fraction == pytest.approx(
        0.7694246723306666
    )


def test_freeze_and_modular_expert_checkpoint_round_trip(tmp_path):
    model = _FakeTransformer()
    controller = inject_appearance_experts(
        model,
        final_block_fraction=0.5,
        expert_parameter_fraction=None,
        expert_width_fraction=0.25,
    )
    freeze_for_expert_training(model, controller)
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(".experts." in name or ".scales." in name for name in trainable)
    assert controller.parameter_count("photo") > 0

    for wrapper in controller.wrappers.values():
        wrapper.experts["photo"].fc2.bias.data.fill_(0.75)
        wrapper.scales["photo"].data.fill_(1.25)
    checkpoint = tmp_path / "mageflow-photo-expert.safetensors"
    manifest = save_appearance_expert(controller, "photo", checkpoint)
    assert manifest["domain"] == "photo"
    layout = inspect_safetensors_layout([checkpoint])
    assert layout["tensor_count"] == 10
    assert all(".shared_ffn." not in key for key in layout["tensors"])
    assert all("animation" not in key for key in layout["tensors"])

    for wrapper in controller.wrappers.values():
        wrapper.experts["photo"].fc2.bias.data.zero_()
        wrapper.scales["photo"].data.zero_()
    report = load_appearance_expert(controller, "photo", checkpoint)
    assert report["compatible"]
    assert all(
        torch.allclose(wrapper.experts["photo"].fc2.bias, torch.full((8,), 0.75))
        for wrapper in controller.wrappers.values()
    )
    assert all(
        wrapper.scales["photo"].item() == pytest.approx(1.25)
        for wrapper in controller.wrappers.values()
    )

    with pytest.raises(ValueError, match="domain"):
        load_appearance_expert(controller, "animation", checkpoint)
