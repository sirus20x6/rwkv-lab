import json
from pathlib import Path

import pytest
import torch

from rwkv_lab.mage_flow_pretrain import (
    DEFAULT_MODEL_REVISION,
    MageFlowTrainConfig,
    _generate_eval_snapshot,
    _load_image_tensor,
    _png_has_only_ancillary_crc_errors,
    canonical_caption_row,
    epoch_packs,
    latent_tokens,
    native_size,
    prepare_manifest,
    prepare_reddit_split,
    prepare_run,
    rectified_flow_loss,
    rectified_flow_path,
)


def _png_with_corrupt_chunk_crc(tmp_path, chunk_type, *, corrupt_data=False):
    from PIL import Image, PngImagePlugin

    path = tmp_path / f"corrupt-{chunk_type.decode()}.png"
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("comment", "hello")
    Image.new("RGB", (8, 6), "red").save(path, pnginfo=png_info)
    payload = bytearray(path.read_bytes())
    offset = 8
    while offset < len(payload):
        length = int.from_bytes(payload[offset : offset + 4], "big")
        current_type = bytes(payload[offset + 4 : offset + 8])
        if current_type == chunk_type:
            if corrupt_data:
                payload[offset + 8 + length // 2] ^= 0xFF
            else:
                crc_offset = offset + 8 + length
                payload[crc_offset] ^= 1
            path.write_bytes(payload)
            return path
        offset += 12 + length
    raise AssertionError(f"PNG chunk not found: {chunk_type!r}")


def test_image_loader_accepts_only_ancillary_png_crc_corruption(tmp_path):
    ancillary = _png_with_corrupt_chunk_crc(tmp_path, b"tEXt")
    assert _png_has_only_ancillary_crc_errors(ancillary)
    tensor = _load_image_tensor(
        {"image": str(ancillary), "train_width": 16, "train_height": 16}
    )
    assert tensor.shape == (3, 16, 16)

    critical = _png_with_corrupt_chunk_crc(tmp_path, b"IDAT", corrupt_data=True)
    assert not _png_has_only_ancillary_crc_errors(critical)
    with pytest.raises(OSError):
        _load_image_tensor(
            {"image": str(critical), "train_width": 16, "train_height": 16}
        )


def test_native_size_preserves_aspect_and_mage_vae_geometry():
    width, height = native_size(1600, 900)
    assert width % 16 == height % 16 == 0
    assert abs(width / height - 1600 / 900) < 0.03
    assert abs(width * height - 1024 * 1024) < 1024 * 32
    assert latent_tokens(width, height) == (width // 16) * (height // 16)


def test_native_size_rejects_extreme_aspect():
    with pytest.raises(ValueError, match="aspect ratio"):
        native_size(5000, 500, max_aspect_ratio=4.0)


def test_canonical_rows_use_caption_not_captioner_prompt(tmp_path):
    from PIL import Image

    image = tmp_path / "sample.png"
    Image.new("RGB", (640, 480), "red").save(image)
    row, reason = canonical_caption_row(
        {
            "image": image.name,
            "text": "Visible red field.",
            "prompt": "Describe this image:",
            "task": "caption",
        },
        data_root=tmp_path,
        pixel_budget=512 * 512,
        max_side=2048,
        max_aspect_ratio=4.0,
        verify_image=True,
    )
    assert reason is None
    assert row["caption"] == "Visible red field."
    assert row["task"] == "generation"


def test_canonical_rows_exclude_structured_tasks(tmp_path):
    row, reason = canonical_caption_row(
        {"image": "missing.jpg", "text": "box=[1,2,3,4]", "task": "sam_mask"},
        data_root=tmp_path,
        pixel_budget=1024 * 1024,
        max_side=2048,
        max_aspect_ratio=4.0,
        verify_image=False,
    )
    assert row is None and reason == "non_caption_task"


def test_parallel_manifest_preparation_deduplicates(tmp_path):
    from PIL import Image

    image = tmp_path / "sample.png"
    Image.new("RGB", (640, 480), "blue").save(image)
    source = tmp_path / "source.jsonl"
    row = {
        "image": image.name,
        "text": "A blue field.",
        "task": "caption",
        "image_sha256": "stable-id",
    }
    source.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    output = tmp_path / "canonical.jsonl"
    report = prepare_manifest(
        source,
        output,
        data_root=tmp_path,
        pixel_budget=512 * 512,
        workers=2,
    )
    assert report["counts"] == {"duplicate": 1, "input": 2, "output": 1}
    assert len(output.read_text().splitlines()) == 1


def test_reddit_split_freezes_valid_prefix_and_holds_out_eval(tmp_path):
    from PIL import Image

    rows = []
    for index in range(6):
        relative = f"subreddit/image-{index}.png"
        image = tmp_path / relative
        image.parent.mkdir(exist_ok=True)
        Image.new("RGB", (640, 480), (index, 0, 0)).save(image)
        rows.append(
            {
                "relative_path": relative,
                "caption": f"Caption {index}.",
                "finish_reason": "length" if index == 1 else "stop",
                "model": "caption-model",
            }
        )
    source = tmp_path / "captions.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    artifacts = tmp_path / "artifacts.jsonl"
    artifacts.write_text(
        json.dumps(
            {
                "relative_path": "subreddit/image-0.png",
                "watermarks": ["mark"],
                "censorship": [],
            }
        )
        + "\n"
    )
    train = tmp_path / "prepared" / "train.jsonl"
    eval_ = tmp_path / "prepared" / "eval.jsonl"

    report = prepare_reddit_split(
        source,
        tmp_path,
        train,
        eval_,
        train_count=3,
        eval_count=2,
        artifact_path=artifacts,
        workers=2,
        seed=9,
    )
    first_train = train.read_text()
    first_eval = eval_.read_text()
    train_rows = [json.loads(line) for line in first_train.splitlines()]
    eval_rows = [json.loads(line) for line in first_eval.splitlines()]
    assert report["train_count"] == 3
    assert report["eval_count"] == 2
    assert report["counts"]["truncated_caption"] == 1
    assert {row["image_id"] for row in train_rows}.isdisjoint(
        row["image_id"] for row in eval_rows
    )
    assert all(row["source"] == "reddit/subreddit" for row in train_rows + eval_rows)

    extra = tmp_path / "subreddit" / "later.png"
    Image.new("RGB", (640, 480), "blue").save(extra)
    with source.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "relative_path": "subreddit/later.png",
                    "caption": "A later append.",
                    "finish_reason": "stop",
                }
            )
            + "\n"
        )
    prepare_reddit_split(
        source,
        tmp_path,
        train,
        eval_,
        train_count=3,
        eval_count=2,
        artifact_path=artifacts,
        workers=2,
        seed=9,
    )
    assert train.read_text() == first_train
    assert eval_.read_text() == first_eval


def test_reddit_split_strict_artifacts_excludes_flags_and_rounds_train(tmp_path):
    from PIL import Image

    source_rows = []
    artifact_rows = []
    for index in range(10):
        relative = f"clean-test/image-{index}.png"
        image = tmp_path / relative
        image.parent.mkdir(exist_ok=True)
        Image.new("RGB", (640, 480), (index, 0, 0)).save(image)
        source_rows.append(
            {
                "relative_path": relative,
                "caption": f"Caption {index}.",
                "finish_reason": "stop",
            }
        )
        if index < 9:
            artifact_rows.append(
                {
                    "relative_path": relative,
                    "watermarks": ["mark"] if index == 8 else [],
                    "censorship": [],
                }
            )
    source = tmp_path / "captions.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
    artifacts = tmp_path / "artifacts.jsonl"
    artifacts.write_text("".join(json.dumps(row) + "\n" for row in artifact_rows))
    train = tmp_path / "prepared" / "train.jsonl"
    eval_ = tmp_path / "prepared" / "eval.jsonl"

    report = prepare_reddit_split(
        source,
        tmp_path,
        train,
        eval_,
        train_count=8,
        eval_count=2,
        artifact_path=artifacts,
        require_clean_artifacts=True,
        allow_smaller_train=True,
        train_count_multiple=4,
        workers=2,
    )
    assert report["train_count"] == 4
    assert report["eval_count"] == 2
    assert report["counts"]["artifact_flagged"] == 1
    assert report["counts"]["artifact_unscanned"] == 1
    selected = [
        json.loads(line)
        for path in (train, eval_)
        for line in path.read_text().splitlines()
    ]
    assert all(not row.get("watermarks") for row in selected)
    assert all(not row.get("censorship") for row in selected)


def test_token_packs_are_deterministic_rank_shards():
    rows = [{"latent_tokens": value} for value in (4, 4, 6, 2, 8, 4)]
    first = epoch_packs(rows, token_budget=8, seed=7, epoch=2, shuffle=True)
    second = epoch_packs(rows, token_budget=8, seed=7, epoch=2, shuffle=True)
    assert first == second
    flat0 = {
        item
        for pack in epoch_packs(
            rows, token_budget=8, seed=7, epoch=2, rank=0, world_size=2
        )
        for item in pack
    }
    flat1 = {
        item
        for pack in epoch_packs(
            rows, token_budget=8, seed=7, epoch=2, rank=1, world_size=2
        )
        for item in pack
    }
    assert flat0.isdisjoint(flat1)
    assert flat0 | flat1 == set(range(len(rows)))


def test_rectified_flow_loss_is_target_token_mean():
    target = torch.zeros(1, 4, 2)
    prediction = torch.ones_like(target)
    optimization, observed = rectified_flow_loss(prediction, target)
    assert observed.item() == pytest.approx(1.0)
    assert optimization.item() == pytest.approx(1.0)


def test_rectified_flow_velocity_reaches_clean_when_sigma_decreases():
    clean = torch.tensor([[[2.0]]])
    noise = torch.tensor([[[-3.0]]])
    sample, velocity = rectified_flow_path(clean, noise, torch.ones(1, 1, 1))
    assert sample.item() == pytest.approx(noise.item())
    # Official scheduler Euler step from sigma=1 to 0 has dt=-1.
    assert (sample - velocity).item() == pytest.approx(clean.item())


def test_plan_pins_model_and_writes_cpu_offload(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text(
        json.dumps(
            {
                "image": "/tmp/image.png",
                "caption": "caption",
                "train_width": 1024,
                "train_height": 1024,
                "latent_tokens": 4096,
            }
        )
        + "\n"
    )
    config = MageFlowTrainConfig(
        train_manifest=str(train),
        eval_manifest=None,
        output_dir=str(tmp_path / "output"),
        max_steps=2,
    )
    receipt = prepare_run(config, tmp_path / "plan")
    assert receipt["model_revision"] == DEFAULT_MODEL_REVISION
    ds = json.loads((tmp_path / "plan" / "deepspeed_zero2_cpu.json").read_text())
    assert ds["zero_optimization"]["stage"] == 2
    assert ds["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    accelerate_yaml = (tmp_path / "plan" / "accelerate.yaml").read_text()
    assert "mixed_precision:" not in accelerate_yaml
    assert ds["bf16"]["enabled"] is True
    launcher = tmp_path / "plan" / "launch.sh"
    assert launcher.stat().st_mode & 0o111
    assert 'HF_HOME="${MAGE_FLOW_HF_HOME:-$REPO_ROOT/.hf_cache}"' in launcher.read_text()


def test_generation_eval_writes_deterministic_step_zero_dashboard_artifact(tmp_path):
    from PIL import Image

    manifest = tmp_path / "eval.jsonl"
    rows = [
        {
            "image": str(tmp_path / f"target-{index}.png"),
            "caption": f"Held-out prompt {index}.",
            "train_width": 64 + index * 16,
            "train_height": 48 + index * 16,
            "latent_tokens": 12,
            "source": "held-out",
        }
        for index in range(3)
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    class FakeTransformer:
        training = True

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

    class FakeModel:
        transformer = object()

        class txt_enc:
            @staticmethod
            def screen_text(prompt):
                class Verdict:
                    violates = prompt.endswith("0.")

                return Verdict()

    class FakePipeline:
        model = FakeModel()

        def __init__(self):
            self.kwargs = None

        def generate(self, prompts, **kwargs):
            self.kwargs = kwargs
            return [
                Image.new("RGB", (width, height), "blue")
                for width, height in zip(
                    kwargs["widths"], kwargs["heights"], strict=True
                )
            ]

    train_manifest = tmp_path / "train.jsonl"
    train_manifest.write_text(json.dumps(rows[0]) + "\n")
    config = MageFlowTrainConfig(
        train_manifest=str(train_manifest),
        eval_manifest=str(manifest),
        output_dir=str(tmp_path / "run"),
        eval_gen_samples=2,
        seed=7,
    )
    pipeline = FakePipeline()
    transformer = FakeTransformer()
    artifact_path = _generate_eval_snapshot(
        pipeline,
        transformer,
        rows,
        config,
        torch.device("cpu"),
        tmp_path / "run",
        step=0,
        baseline=True,
    )

    artifact = json.loads(artifact_path.read_text())
    assert artifact["step"] == 0
    assert artifact["baseline"] is True
    assert artifact["eval_kind"] == "image_generation"
    assert [item["seed"] for item in artifact["items"]] == [100_007, 100_008]
    assert [item["prompt"] for item in artifact["items"]] == [
        "Held-out prompt 1.",
        "Held-out prompt 2.",
    ]
    selection = json.loads(
        (tmp_path / "run" / "eval_generation_selection.json").read_text()
    )
    assert [item["index"] for item in selection["items"]] == [1, 2]
    assert pipeline.kwargs["steps"] == 30
    assert pipeline.kwargs["cfg"] == 5.0
    assert transformer.training is True
    assert all(Path(item["image"]).is_file() for item in artifact["items"])
