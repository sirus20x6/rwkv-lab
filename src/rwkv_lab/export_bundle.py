"""Create verifiable, deployment-neutral exports from rwkv-lab checkpoints.

The canonical training artifact remains the resumable checkpoint.  Export bundles contain only
safe tensor weights plus explicit architecture, tokenizer/template, adapter, dataset, lineage,
and promotion receipts.  Publishing is intentionally absent: recursive jobs cannot push a model
to an external hub without a separate operator action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch

from rwkv_lab.posttrain_data import DEFAULT_TEMPLATE
from rwkv_lab.safe_torch import safe_torch_load


EXPORT_SCHEMA = "rwkv-lab.export.v1"


def export_bundle(checkpoint: str | Path, output: str | Path, *, tokenizer: str | Path = "",
                  adapter: str | Path = "", dataset_manifest: str | Path = "",
                  promotion_receipt: str | Path = "", use_ema: bool = False,
                  metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    from safetensors.torch import save_file

    source = Path(checkpoint).resolve()
    if not source.is_file():
        raise ValueError("full export currently requires a single-process .pt checkpoint file")
    blob = safe_torch_load(source, map_location="cpu")
    if not isinstance(blob, dict) or not isinstance(blob.get("model"), dict) or not blob.get("arch"):
        raise ValueError("checkpoint is not a self-describing rwkv-lab training artifact")
    state = {str(name): tensor.detach().cpu().contiguous()
             for name, tensor in blob["model"].items() if isinstance(tensor, torch.Tensor)}
    if use_ema:
        ema = blob.get("ema")
        if not isinstance(ema, dict) or not ema:
            raise ValueError("--use-ema requested but checkpoint has no EMA weights")
        for name, tensor in ema.items():
            if name in state and isinstance(tensor, torch.Tensor):
                state[name] = tensor.detach().cpu().to(state[name].dtype).contiguous()
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp-", dir=destination.parent))
    try:
        weights = temporary / "model.safetensors"
        save_file(state, str(weights), metadata={"schema": EXPORT_SCHEMA,
                                                 "architecture": "rwkv-lab-native"})
        files = {weights.name: _sha256(weights)}
        tokenizer_entry = None
        if tokenizer:
            tok = Path(tokenizer).resolve()
            if not tok.is_file():
                raise ValueError("tokenizer path is not a file")
            target = temporary / "tokenizer.txt"
            shutil.copy2(tok, target)
            files[target.name] = _sha256(target)
            tokenizer_entry = {"file": target.name, "source": str(tok), "sha256": files[target.name]}
        template = {**DEFAULT_TEMPLATE.__dict__, "sha256": DEFAULT_TEMPLATE.fingerprint()}
        (temporary / "chat_template.json").write_text(json.dumps(template, indent=2, sort_keys=True) + "\n")
        files["chat_template.json"] = _sha256(temporary / "chat_template.json")
        adapter_entry = _copy_adapter(adapter, temporary, files) if adapter else None
        data_entry = _read_receipt(dataset_manifest, expected="rwkv-lab.posttrain.v1") if dataset_manifest else None
        promotion = (_read_receipt(promotion_receipt) if promotion_receipt else
                     {"status": "unassessed", "reason": "no promotion receipt supplied"})
        manifest = {
            "schema": EXPORT_SCHEMA, "architecture": "rwkv-lab-native", "arch": blob["arch"],
            "config": blob.get("config", ""), "step": int(blob.get("step", 0)),
            "weights": {"file": weights.name, "sha256": files[weights.name], "ema": use_ema,
                        "tensors": len(state)},
            "source": {"checkpoint": str(source), "sha256": _sha256(source)},
            "tokenizer": tokenizer_entry, "chat_template": template,
            "adapter": adapter_entry, "dataset": data_entry, "promotion": promotion,
            "metadata": metadata or {}, "files": files,
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        shutil.rmtree(destination, ignore_errors=True)
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verify_bundle(destination)
    return manifest


def verify_bundle(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != EXPORT_SCHEMA:
        raise ValueError("unsupported export bundle schema")
    for relative, expected in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"export bundle hash mismatch: {relative}")
    from safetensors import safe_open
    with safe_open(str(root / manifest["weights"]["file"]), framework="pt") as handle:
        if len(handle.keys()) != int(manifest["weights"]["tensors"]):
            raise ValueError("export tensor count does not match manifest")
    return manifest


def _copy_adapter(value: str | Path, temporary: Path, files: dict[str, str]) -> dict[str, Any]:
    source = Path(value).resolve()
    manifest = json.loads((source / "adapter.json").read_text())
    if manifest.get("schema") != "rwkv-lab.adapter.v1":
        raise ValueError("unsupported adapter receipt")
    target = temporary / "adapter"
    target.mkdir()
    for name in ("adapter.json", manifest["weights"]):
        shutil.copy2(source / name, target / name)
        files[f"adapter/{name}"] = _sha256(target / name)
    return {**manifest, "directory": "adapter"}


def _read_receipt(path: str | Path, expected: str = "") -> dict[str, Any]:
    source = Path(path).resolve()
    receipt = json.loads(source.read_text())
    if not isinstance(receipt, dict) or (expected and receipt.get("schema") != expected):
        raise ValueError(f"invalid receipt {path}")
    return {**receipt, "_receipt": {"path": str(source), "sha256": _sha256(source)}}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a safe, self-describing rwkv-lab bundle")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--dataset-manifest", default="")
    parser.add_argument("--promotion-receipt", default="")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = (verify_bundle(args.output) if args.verify else
              export_bundle(args.checkpoint, args.output, tokenizer=args.tokenizer,
                            adapter=args.adapter, dataset_manifest=args.dataset_manifest,
                            promotion_receipt=args.promotion_receipt, use_ema=args.use_ema))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
