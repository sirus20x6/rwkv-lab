#!/usr/bin/env python3
"""Measure a real multimodal checkpoint's dependence on pixel content.

This is the on-hardware half of ``rwkv_lab.image_conditioning_probe``.  The
decision logic lives in that module and is tested on CPU against stubs; this
script only supplies a real model, real images, and -- crucially -- a negative
control.

The negative control is what makes the result evidence rather than a number.
Running the probe against a working model and reporting "it passed" leaves open
the possibility that the probe passes everything.  So the same loaded model is
measured twice:

``pretrained-vision``
    The checkpoint as it loads, with the vision key remap applied.

``randomized-vision``
    The identical model with its vision tower re-initialised in place, which
    reproduces exactly the defect the Qwen3.6 audit found -- 333 ViT tensors
    freshly initialised because the checkpoint nests them at
    ``model.language_model.visual.*`` and the model class looks for
    ``model.visual.*``.

A trustworthy result is the pair: the first condition passes and the second
fails.  Re-initialising in place rather than loading a second 70 GB copy keeps
the control cheap enough that there is no excuse for skipping it.

Read-only with respect to the checkpoint: nothing is written back to the model
directory, and the in-place randomisation only ever touches the process's own
copy of the weights.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rwkv_lab.image_conditioning_probe import (  # noqa: E402
    ProbeExample,
    build_image_variant,
    evaluate_image_conditioning,
    hf_caption_score_function,
)

# The caption run's own prompts, so the probe measures the model under the
# conditions it was trained and evaluated under.
SYSTEM_PROMPT = (
    "You are an expert image-captioning assistant. Describe only what the image "
    "supports, in natural language suitable for training a text-to-image model."
)
USER_PROMPT = (
    "Write one detailed master caption in 4-8 sentences. Cover the main subject, "
    "secondary objects, actions, spatial relationships, setting, composition, "
    "camera viewpoint, lighting, colors, materials, texture, photographic or "
    "artistic style, and all genuinely legible text. Integrate details naturally "
    "in context rather than listing tags. Do not mention these instructions, "
    "metadata, uncertainty policy, or the image file path."
)


def _checkpoint_identity(model_dir: Path, index: dict) -> dict:
    """Bind the evidence to the checkpoint's content, not to its path.

    A path is not an identity, and this one names a *merged* artifact -- the
    kind that gets regenerated in place. Without a content digest the record
    keeps asserting its result about whatever now sits at that path, and a
    reader cannot tell "still true" from "was true once, about something else".

    Hashing 35B parameters is not needed to get that. Three layers, cheapest
    first, each recorded with what it does and does not cover:

    ``index_sha256``
        The weight map itself: which tensors exist and which shard holds each.
    ``header_digest``
        Every shard's safetensors header -- tensor names, dtypes, shapes, byte
        offsets. A few hundred KB to read, and it moves if the tensor set,
        layout, or precision changes at all.
    ``weight_digest``
        Real per-shard content digests, when the publisher recorded them in a
        receipt beside the weights. This is the only layer that detects weights
        edited in place without a shape change, so when it is unavailable the
        document says so rather than staying quiet.
    """
    identity: dict[str, object] = {
        "index_sha256": "sha256:" + hashlib.sha256(
            (model_dir / "model.safetensors.index.json").read_bytes()
        ).hexdigest(),
    }

    headers = hashlib.sha256()
    for name in sorted(set(index["weight_map"].values())):
        with (model_dir / name).open("rb") as handle:
            length = int.from_bytes(handle.read(8), "little")
            if length > 100 * 1024 * 1024:
                raise SystemExit(f"{name} declares an implausible header length")
            headers.update(name.encode("utf-8"))
            headers.update(handle.read(length))
    identity["header_digest"] = "sha256:" + headers.hexdigest()
    identity["header_digest_covers"] = (
        "tensor names, dtypes, shapes and shard offsets; NOT parameter values"
    )

    receipts = sorted(model_dir.glob("*receipt*.json"))
    for receipt_path in receipts:
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        files = receipt.get("weight_files")
        if not isinstance(files, list) or not files:
            continue
        digests = sorted(
            (str(entry.get("name")), str(entry.get("sha256")))
            for entry in files
            if isinstance(entry, dict) and entry.get("sha256")
        )
        if not digests:
            continue
        identity["weight_digest"] = "sha256:" + hashlib.sha256(
            json.dumps(digests, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        identity["weight_digest_source"] = receipt_path.name
        break
    else:
        identity["weight_digest"] = None
        identity["weight_digest_absent_reason"] = (
            "no receipt beside the weights records per-shard sha256, and rehashing "
            f"{len(set(index['weight_map'].values()))} shards of a 35B checkpoint was "
            "judged not worth the I/O for this probe; the header digest still detects "
            "any change to the tensor set, shapes, dtypes or layout"
        )
    return identity


def _runtime_identity(device: str) -> dict:
    """Versions and hardware, so the numbers are attributable to a stack."""
    import platform

    import torch
    import transformers

    index = int(device.rsplit(":", 1)[1]) if ":" in device else 0
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device_name": torch.cuda.get_device_name(index),
        "cuda": torch.version.cuda,
    }


def _repository_commit() -> dict:
    import subprocess

    root = Path(__file__).resolve().parent.parent
    def git(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True, text=True, check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    # A commit SHA does not survive this repository's squash-merge policy: the
    # SHA the probe ran from is rewritten on merge and becomes unreachable, so
    # a reader who looks it up finds nothing. The digest over the probe's own
    # sources does survive, and identifies the code exactly rather than by
    # where it happened to sit, so both are recorded and the limitation of the
    # SHA is stated rather than left for someone to discover.
    sources = sorted(
        [root / "scripts" / "probe_image_conditioning.py",
         root / "src" / "rwkv_lab" / "image_conditioning_probe.py"],
        key=lambda path: path.name,
    )
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.name.encode("utf-8"))
        digest.update(source.read_bytes())
    identity: dict[str, object] = {
        "probe_source_digest": "sha256:" + digest.hexdigest(),
        "probe_sources": [source.name for source in sources],
    }

    commit = git("rev-parse", "HEAD")
    if commit is None:
        identity["commit"] = None
        identity["commit_absent_reason"] = "probe did not run from a git checkout"
        return identity
    identity["commit"] = commit
    identity["commit_note"] = (
        "pre-squash branch commit; rewritten by the merge, so it may not be "
        "reachable from main -- use probe_source_digest to identify the code"
    )
    # A receipt from a dirty tree describes code that is not in any commit.
    # Recording the fact is the difference between evidence and a claim.
    identity["dirty_worktree"] = bool(git("status", "--porcelain"))
    return identity


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset", required=True, help="a caption JSONL split")
    parser.add_argument("--output", required=True, help="evidence JSON to write")
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--caption-words", type=int, default=60,
                        help="truncate captions to bound sequence length")
    parser.add_argument("--max-pixels", type=int, default=262_144)
    parser.add_argument("--shuffle-seed", type=int, default=20260809,
                        help="seed for the shuffled-tile permutation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-free-vram-gib", type=float, default=80.0)
    return parser.parse_args()


def _examples(dataset: Path, count: int, caption_words: int) -> list[ProbeExample]:
    from PIL import Image

    rows: list[dict] = []
    with dataset.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    # Deterministic selection: sorted by content hash, first N. The probe is a
    # measurement, so which images it used has to be reproducible from the
    # recorded evidence alone.
    rows.sort(key=lambda row: str(row["id"]))
    chosen = rows[:count]
    if len(chosen) < count:
        raise SystemExit(f"{dataset} holds only {len(chosen)} rows, needed {count}")
    return [
        ProbeExample(
            identifier=str(row["id"]),
            image=Image.open(row["image"]).convert("RGB"),
            caption=" ".join(str(row["caption"]).split()[:caption_words]),
        )
        for row in chosen
    ]


def _require_idle_device(device: str, minimum_free_gib: float) -> dict[str, float]:
    """Refuse to start if the device is not actually free.

    The probe shares a host with training runs. A 70 GB load that lands on a
    busy device does not fail politely -- it either OOMs the resident job or
    gets itself killed partway through, so the check happens before the load
    rather than after.
    """
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device is available")
    index = int(device.rsplit(":", 1)[1]) if ":" in device else 0
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    free_gib = free_bytes / (1024 ** 3)
    if free_gib < minimum_free_gib:
        raise SystemExit(
            f"device {device} has {free_gib:.1f} GiB free, below the "
            f"{minimum_free_gib:.1f} GiB floor; refusing to disturb resident work"
        )
    return {
        "free_gib": round(free_gib, 2),
        "total_gib": round(total_bytes / (1024 ** 3), 2),
    }


def _randomize_vision_tower(model) -> int:
    """Re-initialise the vision tower in place, reproducing the audited defect.

    Uses the model's own ``_init_weights``, so the result is distributed exactly
    as a genuinely missing-key load would have left it rather than as some
    convenient stand-in.
    """
    import torch

    visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is None:
        raise SystemExit("model exposes no model.visual tower to randomise")
    torch.manual_seed(20260809)
    initializer = getattr(model, "_init_weights", None)
    if not callable(initializer):
        raise SystemExit("model exposes no _init_weights initialiser")
    with torch.no_grad():
        visual.apply(initializer)
        # _init_weights is a no-op for parameters the class does not recognise
        # (bare nn.Parameter tensors such as pos_embed). Leaving those
        # pretrained would make the control weaker than the real defect, so
        # they are drawn from the configured initializer range explicitly.
        for module in visual.modules():
            for name, parameter in module.named_parameters(recurse=False):
                if isinstance(module, torch.nn.Linear) and name in ("weight", "bias"):
                    continue
                if parameter.dim() >= 2:
                    parameter.normal_(mean=0.0, std=0.02)
    return sum(1 for _ in visual.parameters())


def main() -> int:
    arguments = _arguments()
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device_state = _require_idle_device(arguments.device, arguments.minimum_free_vram_gib)
    examples = _examples(
        Path(arguments.dataset), arguments.examples, arguments.caption_words
    )

    model_dir = Path(arguments.model_dir)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    checkpoint_keys = list(index["weight_map"])
    vision_keys = [key for key in checkpoint_keys if ".visual." in key]
    # The Qwen3.6 checkpoints nest the ViT where the model class does not look.
    # Deriving the remap from the sealed index rather than hardcoding it means
    # the probe still loads a checkpoint that has been republished in the
    # native layout, and fails loudly on one that is in neither.
    remap = (
        {r"^model\.language_model\.visual\.": "model.visual."}
        if any(key.startswith("model.language_model.visual.") for key in checkpoint_keys)
        else {}
    )

    started = time.time()
    model, loading_info = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": arguments.device},
        attn_implementation="sdpa",
        output_loading_info=True,
        **({"key_mapping": remap} if remap else {}),
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_dir, local_files_only=True, max_pixels=arguments.max_pixels
    )
    load_seconds = time.time() - started

    missing = sorted(loading_info.get("missing_keys", ()))
    unexpected = sorted(loading_info.get("unexpected_keys", ()))
    if missing or unexpected:
        raise SystemExit(
            f"refusing to probe an inexact load: {len(missing)} missing, "
            f"{len(unexpected)} unexpected (first: {(missing + unexpected)[:3]})"
        )

    def run(condition: str) -> dict:
        score = hf_caption_score_function(
            model,
            processor,
            system_prompt=SYSTEM_PROMPT,
            prompt=USER_PROMPT,
            device=arguments.device,
        )
        verdict = evaluate_image_conditioning(
            examples,
            score,
            variant_builder=lambda image, variant: build_image_variant(
                image, variant, seed=arguments.shuffle_seed
            ),
            subject=f"{model_dir.name}/{condition}",
        )
        document = verdict.canonical_dict()
        document["condition"] = condition
        return document

    conditions = [run("pretrained-vision")]
    randomized_parameters = _randomize_vision_tower(model)
    conditions.append(run("randomized-vision"))

    evidence = {
        "api_version": "rwkv-lab.image-conditioning-evidence/v1",
        "generated_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repository": _repository_commit(),
        "runtime": _runtime_identity(arguments.device),
        "model_dir": str(model_dir),
        # The path names a MERGED artifact, the kind that gets regenerated in
        # place. These digests are what let a later reader tell "still true"
        # from "was true once, about something else".
        "model_content_identity": _checkpoint_identity(model_dir, index),
        "shuffle_seed": arguments.shuffle_seed,
        "dataset": str(Path(arguments.dataset)),
        "checkpoint_tensor_count": len(checkpoint_keys),
        "checkpoint_vision_tensor_count": len(vision_keys),
        "applied_key_remap": remap,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "randomized_vision_parameter_count": randomized_parameters,
        "device": device_state,
        "load_seconds": round(load_seconds, 1),
        "max_pixels": arguments.max_pixels,
        "caption_words": arguments.caption_words,
        "conditions": conditions,
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for condition in conditions:
        print(
            f"{condition['condition']:>20}: "
            f"consumes_image_content={condition['consumes_image_content']} "
            f"margin={condition['discrimination_margin_nats']:+.4f} nats "
            f"accuracy={condition['discrimination_accuracy']:.0%} "
            f"min_response={condition['minimum_responsiveness_nats']:.4f} "
            f"repeat_drift={condition['repeat_determinism_nats']:.2e}"
        )
    passed = conditions[0]["consumes_image_content"]
    controlled = not conditions[1]["consumes_image_content"]
    print(f"probe verdict: pretrained passes={passed} control fails={controlled}")
    return 0 if (passed and controlled) else 1


if __name__ == "__main__":
    raise SystemExit(main())
