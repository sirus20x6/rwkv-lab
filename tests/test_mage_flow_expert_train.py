import hashlib
import json
import random
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

import rwkv_lab.mage_flow_expert_train as expert_train
from rwkv_lab.mage_flow_adaptation import (
    CFG_NULL_CONDITION,
    UNCAPTIONED_IMAGE_CONDITION,
    assert_homogeneous_batch,
    freeze_for_expert_training,
    inject_appearance_experts,
    inspect_safetensors_layout,
)
from rwkv_lab.mage_flow_expert_train import (
    MageFlowExpertTrainConfig,
    _evaluation_batches,
    configure_training_scope,
    effective_conditioning_prompts,
    epoch_training_batches,
    expert_preflight_report,
    export_final_experts,
    generate_eval_gallery,
    latest_compatible_checkpoint,
    load_training_checkpoint,
    optimizer_parameter_groups,
    optimizer_parameter_routing,
    prepare_fixed_expert_cache,
    prepare_run,
    resolved_worker_component_contract,
    save_training_checkpoint,
    training_contract_fingerprint,
    training_scope_preflight_report,
    unified_evaluation_is_complete,
)
from rwkv_lab.trainvm_adapters import WorkerTrainingComponents
from rwkv_lab.trainvm_worker import load_resolved_training_composition


def _worker_components(config: MageFlowExpertTrainConfig) -> WorkerTrainingComponents:
    root = Path(__file__).resolve().parents[1]
    registry = json.loads(
        (root / "docs/experiment-vm/examples/training-components.v1.json").read_text()
    )
    requested = {
        "optimizer": (
            "fp32_master_adamw_no_decay",
            {
                "learning_rate": config.learning_rate,
                "beta1": config.adam_beta1,
                "beta2": config.adam_beta2,
                "epsilon": config.adam_epsilon,
                "foreach": True,
                "fused": False,
            },
        ),
        "weight_decay": (
            "constant",
            {"weight_decay": config.weight_decay},
        ),
        "parameter_router": (
            "mageflow_appearance_expert",
            {
                "shared_backbone_multiplier": config.shared_learning_rate_multiplier
            },
        ),
        "learning_rate": (
            "linear_warmup_cosine",
            {
                "warmup_steps": config.warmup_steps,
                "max_steps": config.learning_rate_schedule_steps or config.max_steps,
                "minimum_ratio": config.min_learning_rate_ratio,
            },
        ),
        "gradient_clipping": (
            "global_norm",
            {
                "max_norm": config.max_grad_norm,
                "norm_type": 2.0,
                "error_if_nonfinite": False,
            },
        ),
    }
    components = {}
    for slot, (name, configuration) in requested.items():
        descriptor = next(
            item for item in registry["components"] if item["key"]["name"] == name
        )
        descriptor_bytes = json.dumps(
            descriptor, separators=(",", ":"), sort_keys=True
        ).encode()
        components[slot] = {
            "configuration": configuration,
            "descriptor": descriptor,
            "descriptor_digest": "sha256:"
            + hashlib.sha256(descriptor_bytes).hexdigest(),
        }
    body = {
        "api_version": "trainvm.resolved-training-composition/v1",
        "components": components,
        "model_family": "mageflow",
        "registry_digest": "sha256:" + "a" * 64,
    }
    body_bytes = json.dumps(body, separators=(",", ":"), sort_keys=True).encode()
    resolved = load_resolved_training_composition(
        {
            **body,
            "composition_digest": "sha256:" + hashlib.sha256(body_bytes).hexdigest(),
        }
    )
    return WorkerTrainingComponents(resolved, "mageflow")


def _row(index, domain, *, source="source", captioned=True):
    return {
        "image": f"/images/{index}.png",
        "image_id": f"id-{index}",
        "domain": domain,
        "source": source,
        "caption": f"Caption {index}" if captioned else UNCAPTIONED_IMAGE_CONDITION,
        "conditioning_text": (
            f"Caption {index}" if captioned else UNCAPTIONED_IMAGE_CONDITION
        ),
        "conditioning_kind": "human" if captioned else "uncaptioned_image",
        "is_captioned": captioned,
        "training_scope": (
            "expert_and_selected_shared" if captioned else "expert_only"
        ),
        "train_width": 512,
        "train_height": 512,
        "latent_tokens": 1024,
    }


def _write_manifest(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_config_audits_domains_and_initial_stage_has_no_general_updates(tmp_path):
    manifest = tmp_path / "train.jsonl"
    rows = [
        *[_row(index, "photo", source="camera") for index in range(4)],
        *[_row(index + 10, "animation", source="drawings") for index in range(4)],
        _row(30, "general", source="anchor"),
    ]
    _write_manifest(manifest, rows)
    config = MageFlowExpertTrainConfig(
        train_manifest=str(manifest),
        output_dir=str(tmp_path / "output"),
        max_steps=2,
        microbatch_size=2,
    )
    config.validate()
    assert config.domain_weights == {
        "general": 0.0,
        "photo": 0.5,
        "animation": 0.5,
    }
    assert config.expert_parameter_fraction == 0.15
    assert config.final_block_fraction == pytest.approx(2 / 3)
    assert config.expert_width_fraction is None
    batches = epoch_training_batches(rows, config, epoch=0)
    domains = [assert_homogeneous_batch(rows, batch) for batch in batches]
    assert "general" not in domains
    assert set(domains) == {"photo", "animation"}


def test_animation_only_continuation_uses_only_animation_batches(tmp_path):
    manifest = tmp_path / "train.jsonl"
    rows = [_row(index, "animation", source="anima") for index in range(10)]
    _write_manifest(manifest, rows)
    parent = tmp_path / "checkpoint-00013000"
    config = MageFlowExpertTrainConfig(
        train_manifest=str(manifest),
        output_dir=str(tmp_path / "output"),
        max_steps=16_000,
        microbatch_size=2,
        photo_weight=0.0,
        animation_weight=1.0,
        continuation_from=str(parent),
    )
    config.validate()
    domains = [
        assert_homogeneous_batch(rows, batch)
        for batch in epoch_training_batches(rows, config, epoch=0)
    ]
    assert domains == ["animation"] * 5


def test_continuation_path_is_lineage_not_contract_and_modes_are_exclusive(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, [_row(0, "animation")])
    base = MageFlowExpertTrainConfig(
        train_manifest=str(manifest),
        output_dir=str(tmp_path / "output"),
        photo_weight=0,
        animation_weight=1,
        continuation_from="/checkpoint/a",
    )
    alternate = MageFlowExpertTrainConfig(
        **{**base.__dict__, "continuation_from": "/checkpoint/b"}
    )
    assert training_contract_fingerprint(base) == training_contract_fingerprint(
        alternate
    )
    assert training_contract_fingerprint(base) != training_contract_fingerprint(
        base, component_composition_digest="sha256:" + "a" * 64
    )
    invalid = MageFlowExpertTrainConfig(
        **{**base.__dict__, "resume_from": "/checkpoint/exact"}
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        invalid.validate()

    reset = MageFlowExpertTrainConfig(
        **{**base.__dict__, "continuation_reset_scheduler": True}
    )
    assert training_contract_fingerprint(base) == training_contract_fingerprint(reset)
    invalid_reset = MageFlowExpertTrainConfig(
        **{
            **base.__dict__,
            "continuation_from": None,
            "continuation_reset_scheduler": True,
        }
    )
    with pytest.raises(ValueError, match="requires continuation_from"):
        invalid_reset.validate()

    invalid_optimizer_reset = MageFlowExpertTrainConfig(
        **{
            **base.__dict__,
            "continuation_reset_optimizer": True,
            "continuation_reset_scheduler": False,
        }
    )
    with pytest.raises(ValueError, match="also requires resetting the scheduler"):
        invalid_optimizer_reset.validate()


def test_config_rejects_excess_uncaptioned_data(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(
        manifest,
        [
            _row(0, "photo", captioned=False),
            _row(1, "animation", captioned=False),
        ],
    )
    config = MageFlowExpertTrainConfig(
        train_manifest=str(manifest),
        output_dir=str(tmp_path / "output"),
    )
    with pytest.raises(ValueError, match="failed audit"):
        config.validate()


def test_config_honors_failed_preparation_input_audit(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(
        manifest,
        [_row(0, "photo"), _row(1, "animation")],
    )
    manifest.with_suffix(".jsonl.report.json").write_text(
        json.dumps({"audit": {"passed": False}})
    )
    config = MageFlowExpertTrainConfig(
        train_manifest=str(manifest),
        output_dir=str(tmp_path / "output"),
    )
    with pytest.raises(ValueError, match="preparation report failed"):
        config.validate()


def test_unified_eval_exhausts_every_domain_and_rejects_train_leakage(tmp_path):
    train_manifest = tmp_path / "train.jsonl"
    eval_manifest = tmp_path / "eval.jsonl"
    _write_manifest(
        train_manifest,
        [
            *[_row(index, "photo") for index in range(4)],
            *[_row(index + 10, "animation") for index in range(4)],
        ],
    )
    eval_rows = [
        *[_row(index + 100, "photo") for index in range(5)],
        *[_row(index + 200, "animation") for index in range(3)],
    ]
    _write_manifest(eval_manifest, eval_rows)
    config = MageFlowExpertTrainConfig(
        train_manifest=str(train_manifest),
        eval_manifest=str(eval_manifest),
        output_dir=str(tmp_path / "output"),
        microbatch_size=2,
    )
    config.validate()
    assert config.eval_batches_per_domain is None
    photo_batches = _evaluation_batches(
        eval_rows,
        domain="photo",
        batch_size=config.microbatch_size,
        count=config.eval_batches_per_domain,
    )
    animation_batches = _evaluation_batches(
        eval_rows,
        domain="animation",
        batch_size=config.microbatch_size,
        count=config.eval_batches_per_domain,
    )
    assert sum(map(len, photo_batches)) == 5
    assert sum(map(len, animation_batches)) == 3

    receipt = prepare_run(config, tmp_path / "plan")
    assert receipt["evaluation_contract"] == {
        "unified_phase": True,
        "all_manifest_examples": True,
        "eval_examples": 8,
        "eval_domain_counts": {"animation": 3, "photo": 5},
        "routes": ["general", "photo", "animation"],
        "routing_contract": "general_or_one_expert",
        "gallery_samples_per_domain": 4,
        "gallery_generation_steps": 30,
        "gallery_cfg": 5.0,
    }

    leaked = _row(300, "photo")
    leaked["image_id"] = "id-0"
    _write_manifest(
        eval_manifest,
        [leaked, _row(301, "animation")],
    )
    with pytest.raises(ValueError, match="train/evaluation image leakage"):
        config.validate()


def test_evaluate_routes_logs_complete_unified_coverage(tmp_path, monkeypatch):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(
        manifest,
        [_row(0, "photo"), _row(1, "animation")],
    )
    rows = [
        *[_row(index + 100, "photo") for index in range(5)],
        *[_row(index + 200, "animation") for index in range(3)],
    ]
    config = MageFlowExpertTrainConfig(
        train_manifest=str(manifest),
        output_dir=str(tmp_path / "output"),
        microbatch_size=2,
    )

    class Transformer:
        training = True

        def eval(self):
            self.training = False

        def train(self, mode):
            self.training = mode

    class Controller:
        def __init__(self):
            self.active = None
            self.calls = []

        @contextmanager
        def route(self, domain):
            prior, self.active = self.active, domain
            try:
                yield
            finally:
                self.active = prior

    controller = Controller()
    transformer = Transformer()
    monkeypatch.setattr(expert_train, "_load_image_tensor", lambda row: row)
    monkeypatch.setattr(
        expert_train,
        "encode_domain_batch",
        lambda _model, batch_rows, _images, _config, _device, caption_dropout: {
            "velocity": torch.zeros(1),
            "batch_size": len(batch_rows),
        },
    )

    def forward(_transformer, flow):
        controller.calls.append((controller.active, flow["batch_size"]))
        return torch.zeros(1)

    monkeypatch.setattr(expert_train, "_forward_transformer", forward)
    monkeypatch.setattr(
        expert_train,
        "rectified_flow_loss",
        lambda _prediction, _velocity: (torch.zeros(1), torch.ones(1)),
    )
    metrics = expert_train.evaluate_routes(
        transformer,
        controller,
        object(),
        rows,
        config,
        torch.device("cpu"),
    )
    assert metrics["eval/photo_examples"] == 5
    assert metrics["eval/animation_examples"] == 3
    assert metrics["eval/examples"] == 8
    assert metrics["eval/routes_per_example"] == 3
    assert len(controller.calls) == 15
    assert Counter(route for route, _batch_size in controller.calls) == {
        "general": 5,
        "photo": 5,
        "animation": 5,
    }
    assert transformer.training is True

    controller.calls.clear()
    config.mandatory_expert_routing = True
    metrics = expert_train.evaluate_routes(
        transformer,
        controller,
        object(),
        rows,
        config,
        torch.device("cpu"),
    )
    assert metrics["eval/routes_per_example"] == 2
    assert not any("_via_general_" in key for key in metrics)
    assert len(controller.calls) == 10
    assert Counter(route for route, _batch_size in controller.calls) == {
        "photo": 5,
        "animation": 5,
    }


def test_eval_gallery_is_balanced_routed_and_dashboard_compatible(tmp_path):
    rows = [
        *[_row(index + 100, "photo") for index in range(5)],
        *[_row(index + 200, "animation") for index in range(5)],
    ]
    for index, row in enumerate(rows):
        target = tmp_path / f"target-{index}.png"
        Image.new("RGB", (16, 16), (index, 0, 0)).save(target)
        row["image"] = str(target)

    class TextEncoder:
        def screen_text(self, _prompt):
            raise AssertionError("private eval prompts should bypass refusal images")

    class Model:
        txt_enc = TextEncoder()

    class Controller:
        def __init__(self):
            self.active = None
            self.generated_routes = []

        @contextmanager
        def route(self, domain):
            prior, self.active = self.active, domain
            try:
                yield
            finally:
                self.active = prior

    controller = Controller()

    class Pipeline:
        model = Model()

        def generate(self, prompts, **_kwargs):
            controller.generated_routes.append(controller.active)
            return [Image.new("RGB", (16, 16), "blue") for _ in prompts]

    class Transformer:
        training = True

        def eval(self):
            self.training = False

        def train(self, mode):
            self.training = mode

    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, [_row(0, "photo"), _row(1, "animation")])
    config = MageFlowExpertTrainConfig(
        train_manifest=str(manifest),
        output_dir=str(tmp_path / "output"),
    )
    artifact = generate_eval_gallery(
        Pipeline(),
        Transformer(),
        controller,
        rows,
        config,
        torch.device("cpu"),
        tmp_path / "output",
        step=500,
    )
    payload = json.loads(artifact.read_text())
    assert payload["eval_kind"] == "image_generation"
    assert payload["complete"] is True
    assert payload["samples_per_domain"] == 4
    assert Counter(item["domain"] for item in payload["items"]) == {
        "photo": 4,
        "animation": 4,
    }
    assert controller.generated_routes == ["photo", "animation"]
    assert all(Path(item["image"]).is_file() for item in payload["items"])
    assert all(Path(item["target_image"]).is_file() for item in payload["items"])
    generate_eval_gallery(
        Pipeline(),
        Transformer(),
        controller,
        rows,
        config,
        torch.device("cpu"),
        tmp_path / "output",
        step=500,
    )
    assert controller.generated_routes == [
        "photo",
        "animation",
        "photo",
        "animation",
    ]
    assert not unified_evaluation_is_complete(tmp_path / "output", 500)
    (tmp_path / "output" / "train.jsonl").write_text(
        json.dumps({"kind": "eval", "step": 500}) + "\n"
    )
    assert unified_evaluation_is_complete(tmp_path / "output", 500)


def test_general_rows_cannot_dilute_expert_uncaptioned_limit(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(
        manifest,
        [
            *[_row(index, "general") for index in range(10)],
            _row(20, "photo", captioned=False),
            _row(21, "animation"),
        ],
    )
    config = MageFlowExpertTrainConfig(
        train_manifest=str(manifest),
        output_dir=str(tmp_path / "output"),
    )
    with pytest.raises(ValueError, match="expert-domain training rows failed"):
        config.validate()


def test_conditioning_dispatch_never_collapses_uncaptioned_into_cfg_null():
    rows = [
        _row(0, "photo"),
        _row(1, "photo", captioned=False),
        {
            **_row(2, "photo"),
            "conditioning_text": CFG_NULL_CONDITION,
            "conditioning_kind": "cfg_null",
            "is_captioned": False,
        },
    ]

    class FixedRng:
        @staticmethod
        def random():
            return 0.1

    prompts, kinds = effective_conditioning_prompts(
        rows, caption_dropout=0.5, rng=FixedRng()
    )
    assert prompts == [" ", UNCAPTIONED_IMAGE_CONDITION, " "]
    assert kinds == ["cfg_dropout", "uncaptioned_image", "cfg_null"]


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.img_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, value):
        return value + self.img_mlp(value)


class _Transformer(nn.Module):
    def __init__(self, dim=8, depth=4):
        super().__init__()
        self.inner_dim = dim
        self.transformer_blocks = nn.ModuleList([_Block(dim) for _ in range(depth)])

    def forward(self, value):
        for block in self.transformer_blocks:
            value = block(value)
        return value


def _controller_and_optimizer():
    model = _Transformer()
    controller = inject_appearance_experts(
        model,
        final_block_fraction=0.5,
        expert_parameter_fraction=None,
        expert_width_fraction=0.25,
        expert_dtype=torch.float32,
    )
    freeze_for_expert_training(model, controller)
    parameters = list(controller.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in parameters)
    optimizer = torch.optim.AdamW(parameters, lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    return model, controller, optimizer, scheduler


def test_fixed_experts_and_only_original_final_third_are_trainable():
    model = _Transformer(depth=6)
    controller = inject_appearance_experts(
        model,
        final_block_fraction=0.5,
        expert_parameter_fraction=None,
        expert_width_fraction=0.25,
        expert_dtype=torch.float32,
    )
    scope = configure_training_scope(
        model,
        controller,
        train_experts=False,
        shared_final_fraction=1 / 3,
    )
    assert scope["shared_block_indices"] == [4, 5]
    assert scope["expert_trainable_parameter_count"] == 0
    assert scope["shared_trainable_parameter_count"] > 0
    assert all(not parameter.requires_grad for parameter in controller.parameters())
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert all(
        name.startswith(("transformer_blocks.4.", "transformer_blocks.5."))
        for name in trainable_names
    )
    assert all(
        ".experts." not in name and ".scales." not in name for name in trainable_names
    )
    assert training_scope_preflight_report(model, controller, scope)["passed"]


def test_unfrozen_experts_use_twice_the_shared_tail_learning_rate():
    model = _Transformer(depth=6)
    controller = inject_appearance_experts(
        model,
        final_block_fraction=0.5,
        expert_parameter_fraction=None,
        expert_width_fraction=0.25,
        expert_dtype=torch.float32,
    )
    configure_training_scope(
        model,
        controller,
        train_experts=True,
        shared_final_fraction=1 / 3,
    )
    groups = optimizer_parameter_groups(
        model,
        controller,
        learning_rate=2.0e-6,
        shared_learning_rate_multiplier=0.5,
    )
    rates = {group["group_name"]: group["lr"] for group in groups}
    assert rates == {"experts": 2.0e-6, "shared_backbone": 1.0e-6}
    expert_ids = {id(parameter) for parameter in controller.parameters()}
    assert all(id(parameter) in expert_ids for parameter in groups[0]["params"])
    assert all(id(parameter) not in expert_ids for parameter in groups[1]["params"])
    routing = optimizer_parameter_routing(
        model,
        controller,
        learning_rate=2.0e-6,
        shared_learning_rate_multiplier=0.5,
    )
    assert routing.report["passed"]
    assert routing.report["trainable_tensor_count"] == sum(
        route["trainable_tensor_count"] for route in routing.report["routes"]
    )


def test_worker_composition_drives_expert_optimizer_contract_and_routing(tmp_path):
    config = MageFlowExpertTrainConfig(
        train_manifest=str(tmp_path / "train.jsonl"),
        output_dir=str(tmp_path / "run"),
        max_steps=20,
        warmup_steps=2,
        learning_rate=2.0e-5,
        shared_learning_rate_multiplier=0.5,
    )
    components = _worker_components(config)
    learning_rate, evidence, composition_digest = resolved_worker_component_contract(
        config, components
    )
    assert learning_rate == pytest.approx(2.0e-5)
    assert evidence["parameter_router"]["category"] == "parameter_router"
    assert composition_digest == components.composition.composition_digest

    model = _Transformer(depth=6)
    controller = inject_appearance_experts(
        model,
        final_block_fraction=0.5,
        expert_parameter_fraction=None,
        expert_width_fraction=0.25,
        expert_dtype=torch.float32,
    )
    configure_training_scope(
        model,
        controller,
        train_experts=True,
        shared_final_fraction=1 / 3,
    )
    routing = optimizer_parameter_routing(
        model,
        controller,
        learning_rate=learning_rate,
        shared_learning_rate_multiplier=config.shared_learning_rate_multiplier,
        worker_components=components,
    )
    assert [group["lr"] for group in routing.groups] == pytest.approx(
        [2.0e-5, 1.0e-5]
    )

    mismatched = MageFlowExpertTrainConfig(
        **{**config.__dict__, "weight_decay": 0.02}
    )
    with pytest.raises(ValueError, match="weight-decay composition disagrees"):
        resolved_worker_component_contract(mismatched, components)


def test_fp32_master_experts_run_under_bf16_autocast():
    model = _Transformer().to(dtype=torch.bfloat16)
    controller = inject_appearance_experts(
        model,
        final_block_fraction=0.5,
        expert_parameter_fraction=None,
        expert_width_fraction=0.25,
        expert_dtype=torch.float32,
    )
    freeze_for_expert_training(model, controller)
    inputs = torch.randn(2, 8, dtype=torch.bfloat16)
    with (
        controller.route("photo"),
        torch.autocast(device_type="cpu", dtype=torch.bfloat16),
    ):
        loss = model(inputs).float().square().mean()
    loss.backward()
    parameters = list(controller.parameters("photo"))
    assert all(parameter.dtype == torch.float32 for parameter in parameters)
    assert all(
        parameter.grad is None or parameter.grad.dtype == torch.float32
        for parameter in parameters
    )
    assert controller.wrappers[2].experts["photo"].fc2.weight.grad is not None


def test_preflight_rejects_shared_gradients_and_nonzero_fresh_outputs():
    model, controller, _optimizer, _scheduler = _controller_and_optimizer()
    assert expert_preflight_report(model, controller, require_zero_output=True)[
        "passed"
    ]
    controller.wrappers[2].shared_ffn[0].weight.requires_grad_(True)
    controller.wrappers[2].experts["photo"].fc2.bias.data.fill_(1)
    report = expert_preflight_report(model, controller, require_zero_output=True)
    assert not report["passed"]
    assert report["foreign_trainable_parameters"]
    assert report["nonzero_output_experts"] == ["2:photo"]


def test_exact_checkpoint_round_trip_and_bf16_final_exports(tmp_path):
    model, controller, optimizer, scheduler = _controller_and_optimizer()
    inputs = torch.randn(2, 8)
    with controller.route("photo"):
        loss = model(inputs).square().mean()
        loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    expected = {
        name: value.detach().clone()
        for name, value in controller.wrappers[2].experts["photo"].state_dict().items()
    }
    random.seed(17)
    torch.manual_seed(19)
    checkpoint = save_training_checkpoint(
        controller,
        optimizer,
        scheduler,
        tmp_path,
        global_step=3,
        epoch=2,
        batch_index=7,
        keep_last_n=2,
        contract_fingerprint="contract-a",
    )

    for parameter in controller.parameters():
        parameter.data.zero_()
    random.seed(100)
    torch.manual_seed(100)
    with pytest.raises(ValueError, match="contract does not match"):
        load_training_checkpoint(
            controller,
            optimizer,
            scheduler,
            checkpoint,
            expected_contract_fingerprint="contract-b",
        )
    restored = load_training_checkpoint(
        controller,
        optimizer,
        scheduler,
        checkpoint,
        expected_contract_fingerprint="contract-a",
    )
    assert restored == {"global_step": 3, "epoch": 2, "batch_index": 7}
    for name, value in controller.wrappers[2].experts["photo"].state_dict().items():
        assert torch.equal(value, expected[name])

    exports = export_final_experts(controller, tmp_path / "final")
    for path in exports.values():
        layout = inspect_safetensors_layout([Path(path)])
        assert {item["dtype"] for item in layout["tensors"].values()} == {"BF16"}


def test_shared_tail_checkpoint_round_trip_with_fresh_optimizer(tmp_path):
    model = _Transformer(depth=6)
    controller = inject_appearance_experts(
        model,
        final_block_fraction=0.5,
        expert_parameter_fraction=None,
        expert_width_fraction=0.25,
        expert_dtype=torch.float32,
    )
    configure_training_scope(
        model,
        controller,
        train_experts=False,
        shared_final_fraction=1 / 3,
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    expected_name, expected_parameter = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    expected_parameter.data.fill_(0.25)
    checkpoint = save_training_checkpoint(
        controller,
        optimizer,
        scheduler,
        tmp_path,
        global_step=9,
        epoch=1,
        batch_index=12,
        keep_last_n=2,
        contract_fingerprint="scope-contract",
        transformer=model,
    )
    expected_parameter.data.zero_()

    fresh_optimizer = torch.optim.AdamW(parameters, lr=2.0e-4)
    fresh_scheduler = torch.optim.lr_scheduler.LambdaLR(
        fresh_optimizer, lambda _step: 1.0
    )
    restored = load_training_checkpoint(
        controller,
        fresh_optimizer,
        fresh_scheduler,
        checkpoint,
        restore_optimizer=False,
        restore_scheduler=False,
        transformer=model,
    )
    assert restored == {"global_step": 9, "epoch": 1, "batch_index": 12}
    restored_parameter = dict(model.named_parameters())[expected_name]
    assert torch.all(restored_parameter == 0.25)
    assert fresh_optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-4)

    fixed_cache = prepare_fixed_expert_cache(checkpoint, tmp_path)
    next_checkpoint = save_training_checkpoint(
        controller,
        fresh_optimizer,
        fresh_scheduler,
        tmp_path,
        global_step=10,
        epoch=1,
        batch_index=13,
        keep_last_n=2,
        contract_fingerprint="scope-contract",
        transformer=model,
        fixed_expert_source_dir=fixed_cache,
    )
    source_expert = fixed_cache / "mageflow-photo-expert.safetensors"
    linked_expert = next_checkpoint / "mageflow-photo-expert.safetensors"
    assert source_expert.stat().st_ino == linked_expert.stat().st_ino
    assert (
        latest_compatible_checkpoint(
            tmp_path,
            checkpoint,
            contract_fingerprint="scope-contract",
        )
        == next_checkpoint
    )
    assert (
        latest_compatible_checkpoint(
            tmp_path,
            None,
            contract_fingerprint="scope-contract",
        )
        == next_checkpoint
    )


def test_plan_writes_pinned_single_gpu_launcher(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(
        manifest,
        [_row(0, "photo"), _row(1, "animation")],
    )
    config = MageFlowExpertTrainConfig(
        train_manifest=str(manifest),
        output_dir=str(tmp_path / "output"),
        max_steps=2,
        microbatch_size=1,
    )
    receipt = prepare_run(config, tmp_path / "plan")
    assert receipt["expert_only"]
    assert receipt["general_training_weight"] == 0
    launcher = tmp_path / "plan" / "launch.sh"
    assert launcher.stat().st_mode & 0o111
    assert "rwkv_lab.mage_flow_expert_train train" in launcher.read_text()
    planned = json.loads((tmp_path / "plan" / "train_config.json").read_text())
    assert planned["expert_parameter_fraction"] == 0.15
    assert planned["expert_width_fraction"] is None

    fa4_config = MageFlowExpertTrainConfig(
        train_manifest=str(manifest),
        output_dir=str(tmp_path / "fa4-output"),
        max_steps=2,
        microbatch_size=1,
        attention_backend="flash4",
    )
    prepare_run(fa4_config, tmp_path / "fa4-plan")
    fa4_launcher = (tmp_path / "fa4-plan" / "launch.sh").read_text()
    assert ".venv-mage-flow-fa4" in fa4_launcher
