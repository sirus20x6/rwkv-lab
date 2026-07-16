"""Caption one image with a MoonViT -> RWKV vision adapter checkpoint."""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import torch
from PIL import Image

from rwkv_lab.engram_lmb import LexicalMemoryBank, attach_engram, float_growth_params
from rwkv_lab.deep_vision import DeepVisionInjector, LayerMatchedVisionInjector
from rwkv_lab.generate import SEP, WorldVocab
from rwkv_lab.moonvit import (MoonViT, MoonViTPrefixProjector,
                              checkpoint_fingerprint, pool_features,
                              valid_torch_archive_storages)
from rwkv_lab.rwkv_finetune import load_g1g_fla
from rwkv_lab.vision_fusion import (
    AlignedFrozenVisionFeatures, VisionFusionResidual, VisionTowerConfig)
from rwkv_lab.vision_train import insert_boundary_ids, insert_visual_span
from rwkv_lab.vision_loop import (
    install_factored_timemix,
    load_loop_adapter_state,
    set_loop_enabled,
    set_loop_scale,
)


def _pick(logits: torch.Tensor, *, temperature: float, top_p: float) -> int:
    logits = logits.float().clone()
    logits[0] = -torch.inf  # padding/empty token is never a caption continuation
    if temperature <= 0:
        return int(logits.argmax())
    logits /= temperature
    if 0 < top_p < 1:
        values, indices = logits.sort(descending=True)
        probs = values.softmax(-1)
        remove = probs.cumsum(-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits[indices[remove]] = -torch.inf
    return int(torch.multinomial(logits.softmax(-1), 1))


def checkpoint_runtime_scales(args: dict, step: int) -> tuple[bool, float, float]:
    """Recover the loop and Engram injection scales represented by a checkpoint."""
    loop_count = int(args.get("loop_count", 1))
    loop_start = int(args.get("loop_start_step", 0))
    loop_ramp = int(args.get("loop_ramp_steps", 0))
    loop_enabled = loop_count > 1 and step >= loop_start
    if not loop_enabled:
        loop_scale = 0.0
    elif loop_ramp <= 0:
        loop_scale = 1.0
    else:
        loop_scale = min(1.0, (step - loop_start + 1) / loop_ramp)
    engram_warmup = int(args.get("engram_warmup_steps", 0))
    engram_scale = 1.0 if engram_warmup <= 0 else min(1.0, step / engram_warmup)
    return loop_enabled, loop_scale, engram_scale


@torch.inference_mode()
def caption(checkpoint: str | Path, image_path: str | Path, *, max_new: int = 192,
            temperature: float = 0.0, top_p: float = 0.9, seed: int = 1) -> dict:
    started = time.perf_counter()
    checkpoint = Path(checkpoint)
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not valid_torch_archive_storages(checkpoint, blob):
        raise RuntimeError(f"checkpoint archive integrity check failed: {checkpoint}")
    if int(blob.get("schema", -1)) != 3:
        raise RuntimeError(f"unsupported vision checkpoint schema {blob.get('schema')}")
    args = blob["args"]
    for name in ("rwkv", "moonvit"):
        if checkpoint_fingerprint(args[name]) != args.get(f"{name}_fingerprint"):
            raise RuntimeError(f"{name} weights no longer match the caption checkpoint")
    step = int(blob["step"])
    loop_enabled, loop_scale, engram_scale = checkpoint_runtime_scales(args, step)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    rwkv = load_g1g_fla(args["rwkv"], device="cuda")
    rwkv.requires_grad_(False)
    engram = None
    if bool(args.get("engram", False)):
        sites = sorted({
            int(value.strip())
            for value in str(args.get("engram_sites", "3,15")).split(",")
            if value.strip()
        })
        vocab_size = int(rwkv.config.vocab_size)
        engram = LexicalMemoryBank(
            hidden_size=int(rwkv.config.hidden_size),
            vocab_size=vocab_size,
            layer_sites=sites,
            d_row=int(args.get("engram_drow", 128)),
            table_rows=min(int(args.get("engram_rows", vocab_size)), vocab_size),
            num_heads=int(rwkv.config.num_heads),
            max_loops=int(args["loop_count"]),
            boundary_id=args.get("engram_boundary_id", 0),
        )
        engram.to(device="cuda", dtype=torch.bfloat16)
        float_growth_params(engram)
        # Training attaches Engram before TimeMix is wrapped. Match that order
        # so the value-stream hooks land on the same frozen RWKV modules.
        attach_engram(rwkv, engram, resolve="model.layers")
        rwkv.engram = engram
        state = blob.get("engram")
        if state is None:
            raise RuntimeError("checkpoint enables Engram but contains no Engram state")
        engram.load_state_dict(state)
        engram.set_warmup(engram_scale)
        engram.eval()
    wrappers = install_factored_timemix(
        rwkv, n_loops=int(args["loop_count"]), gate_cap=float(args["loop_gate_cap"]),
        loop_index=bool(args["loop_index"]))
    load_loop_adapter_state(wrappers, blob["loops"])
    set_loop_enabled(wrappers, loop_enabled)
    set_loop_scale(wrappers, loop_scale)
    rwkv.eval()

    vision = MoonViT.from_checkpoint(
        args["moonvit"], device="cuda",
        max_input_patches=int(args["max_input_patches"]),
        tap_layers=tuple(int(value) for value in
                         str(args.get("moonvit_tap_layers", "")).split(",")
                         if value.strip()),
        view_mode=str(args.get("vision_view_mode", "full")))
    vision.requires_grad_(False)
    projector = MoonViTPrefixProjector(
        int(rwkv.config.hidden_size), int(args["prefix_tokens"]),
        resampler_layers=int(args.get("vision_resampler_layers", 0)),
        resampler_width=int(args.get("vision_resampler_width", 1024)),
        resampler_heads=int(args.get("vision_resampler_heads", 8))).cuda().float()
    projector.load_state_dict(blob["projector"])
    projector.eval()
    deep_vision = None
    deep_spec = str(args.get("deep_vision_layers", "")).strip()
    if deep_spec:
        sites = sorted({int(value.strip()) for value in deep_spec.split(",")
                        if value.strip()})
        deep_vision = DeepVisionInjector(
            int(rwkv.config.hidden_size), sites,
            rank=int(args.get("deep_vision_rank", 256))).cuda().float()
        deep_vision.install(rwkv.model.layers)
        state = blob.get("deep_vision")
        if state is None:
            raise RuntimeError("checkpoint enables deep vision but contains no state")
        deep_vision.load_state_dict(state)
        deep_vision.eval()
    layer_vision = None
    layer_spec = str(args.get("layer_vision_layers", "")).strip()
    if layer_spec:
        sites = sorted({int(value.strip()) for value in layer_spec.split(",")
                        if value.strip()})
        layer_vision = LayerMatchedVisionInjector(
            int(rwkv.config.hidden_size), sites,
            rank=int(args.get("layer_vision_rank", 256))).cuda().float()
        layer_vision.install(rwkv.model.layers)
        state = blob.get("layer_vision")
        if state is None:
            raise RuntimeError("checkpoint enables layer-matched vision but has no state")
        layer_vision.load_state_dict(state)
        layer_vision.eval()

    fusion_tower = None
    vision_fusion = None
    if bool(args.get("vision_fusion", False)):
        fusion_config = VisionTowerConfig(
            siglip2=args["siglip2_model"], dinov2=args["dinov2_model"],
            sam=args["sam_model"],
            siglip_width=int(args.get("siglip2_width", 768)))
        if fusion_config.fingerprint() != args.get("vision_fusion_fingerprint"):
            raise RuntimeError("fusion tower weights no longer match the caption checkpoint")
        fusion_tower = AlignedFrozenVisionFeatures(fusion_config).load_pretrained(
                device="cuda", dtype=torch.bfloat16)
        vision_fusion = VisionFusionResidual(
            int(rwkv.config.hidden_size),
            rank=int(args.get("vision_fusion_rank", 512)),
            source_width=fusion_tower.width).cuda().float()
        state = blob.get("vision_fusion")
        if state is None:
            raise RuntimeError("checkpoint enables three-tower fusion but has no state")
        vision_fusion.load_state_dict(state)
        vision_fusion.eval()

    # Match the feature-cache/training contract exactly. Existing MoonViT
    # features were built from stored pixel order, without applying EXIF
    # orientation; inference must not silently rotate a different input.
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    vocab = WorldVocab()
    prompt = str(args["prompt"])
    prompt_ids = vocab.encode(prompt)
    sandwich = bool(args.get("sandwich_prompt", False))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        raw_features = vision([image])
        # Training always feeds pooled prefix-width features (the cacheable
        # contract) to both the projector and the layer-matched injector.
        features = [pool_features(item, projector.prefix_tokens).squeeze(0)
                    for item in raw_features]
        prefix = projector(features)
        if fusion_tower is not None and vision_fusion is not None:
            fusion_features = fusion_tower(
                [image], tokens=int(args["prefix_tokens"]), device="cuda")
            prefix = prefix + vision_fusion(fusion_features).to(prefix.dtype)

    generated: list[int] = []
    stopped = False
    for _ in range(max_new):
        sequence = prompt_ids * (2 if sandwich else 1) + generated
        starts = (len(prompt_ids) if sandwich else 0,)
        ids = torch.tensor([sequence], dtype=torch.long, device="cuda")
        if engram is not None:
            boundary = 0 if engram.boundary_id is None else int(engram.boundary_id)
            engram.set_input_ids(insert_boundary_ids(
                ids, starts, prefix.shape[1], boundary))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            text = rwkv.model.embeddings(ids)
            embeds = insert_visual_span(text, prefix, starts)
            mask = torch.ones(embeds.shape[:2], dtype=torch.bool, device="cuda")
            with contextlib.ExitStack() as stack:
                if deep_vision is not None:
                    stack.enter_context(deep_vision.use_prefix(prefix, starts))
                if layer_vision is not None:
                    stack.enter_context(layer_vision.use_features(
                        torch.stack(features), starts))
                hidden = rwkv.model(inputs_embeds=embeds, attention_mask=mask,
                                    output_hidden_states=False, use_cache=False,
                                    return_dict=True).last_hidden_state
            logits = rwkv.lm_head(hidden[:, -1])
            if engram is not None:
                logits = engram.logit_bias_at(
                    logits,
                    torch.zeros(1, dtype=torch.long, device="cuda"),
                    torch.full((1,), hidden.shape[1] - 1,
                               dtype=torch.long, device="cuda"),
                    inplace=True,
                )
            logits = logits[0]
        token = _pick(logits, temperature=temperature, top_p=top_p)
        if token == SEP:
            stopped = True
            break
        generated.append(token)

    return {
        "checkpoint": str(checkpoint),
        "step": step,
        "image": str(Path(image_path).resolve()),
        "prompt": prompt,
        "engram": engram is not None,
        "engram_scale": engram_scale if engram is not None else 0.0,
        "loop_enabled": loop_enabled,
        "loop_scale": loop_scale,
        "caption": vocab.decode(generated).strip(),
        "tokens": len(generated),
        "stopped_at_eod": stopped,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--max-new", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = caption(args.checkpoint, args.image, max_new=args.max_new,
                     temperature=args.temperature, top_p=args.top_p, seed=args.seed)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(result["caption"])


if __name__ == "__main__":
    main()
