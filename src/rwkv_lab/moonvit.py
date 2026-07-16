"""Frozen MoonViT image encoder used by the RWKV vision experiments.

This is the image-only portion of Moonshot's Kimi K2.6 implementation.  It
loads the separately downloaded ``model-00064-of-000064.safetensors`` shard,
which contains only ``vision_tower.*`` tensors.  The Kimi language model and
its 7,168-wide projector are deliberately not required.  The model math and
normalization match Moonshot's implementation; callers may deliberately use a
smaller input-patch budget than Moonshot's 16,384-patch processor default.
"""
from __future__ import annotations

import hashlib
import math
import pickle
import zipfile
import zlib
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn import functional as F

CACHE_SCHEMA = "moonvit-pooled-v3"


def valid_pooled_feature(item: object, prefix_tokens: int, stages: int = 1) -> bool:
    """Return whether a cache payload matches the pooled MoonViT contract."""
    expected = ((int(prefix_tokens), 4, 1152) if int(stages) == 1 else
                (int(stages), int(prefix_tokens), 4, 1152))
    return (torch.is_tensor(item)
            and tuple(item.shape) == expected
            and item.dtype in (torch.float16, torch.bfloat16, torch.float32))


def valid_pooled_feature_payload(item: object, prefix_tokens: int,
                                 stages: int = 1) -> bool:
    """Validate structure plus numeric payload when admitting an entry from disk."""
    return (valid_pooled_feature(item, prefix_tokens, stages)
            and bool(torch.isfinite(item).all()))


def valid_pooled_feature_archive(path: str | Path, item: object,
                                 prefix_tokens: int, stages: int = 1) -> bool:
    """Validate a cached tensor and its serialized storage checksum.

    ``torch.load`` intentionally does not verify the CRC stored in a PyTorch ZIP
    archive.  A flipped payload bit can therefore remain finite and satisfy the
    shape/dtype contract while silently changing the image feature.  Compare the
    already-loaded CPU storage with the archive's CRC: this scans RAM once, but
    does not read the feature payload from disk a second time.

    Current pooled-v3 entries are single-storage ``torch.save`` ZIP archives.
    Legacy pickle files have no integrity receipt and are rejected so callers
    regenerate them through the frozen vision tower.
    """
    if not valid_pooled_feature_payload(item, prefix_tokens, stages):
        return False
    assert torch.is_tensor(item)
    if item.device.type != "cpu":
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            members = [info for info in archive.infolist()
                       if info.filename.endswith("/data/0")]
            if len(members) != 1:
                return False
            member = members[0]
            storage = torch.as_tensor(item.untyped_storage(), dtype=torch.uint8)
            if member.file_size != storage.numel():
                return False
            checksum = zlib.crc32(storage.numpy()) & 0xFFFFFFFF
            return checksum == member.CRC
    except (FileNotFoundError, OSError, RuntimeError, ValueError,
            pickle.UnpicklingError, zipfile.BadZipFile):
        return False


def valid_torch_archive_storages(path: str | Path, value: object) -> bool:
    """Compare loaded CPU tensor storages with every CRC in a torch ZIP.

    This is a resume/load-time integrity check with no second payload read: the
    checkpoint has already been deserialized to CPU, so CRCs are computed from
    those resident bytes. Non-storage ZIP members are small and are read once so
    corruption of ``data.pkl`` or archive metadata is covered as well.
    """
    expected: Counter[tuple[int, int]] = Counter()
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                parts = info.filename.rsplit("/", 2)
                if len(parts) >= 2 and parts[-2] == "data" and parts[-1].isdigit():
                    expected[(info.file_size, info.CRC)] += 1
                else:
                    # ZipFile.read performs the CRC check omitted by torch.load.
                    archive.read(info)
    except (FileNotFoundError, OSError, RuntimeError, ValueError,
            zipfile.BadZipFile):
        return False
    if not expected:
        return False

    actual: Counter[tuple[int, int]] = Counter()
    seen_storages: set[int] = set()
    seen_containers: set[int] = set()

    def visit(item: object) -> bool:
        if torch.is_tensor(item):
            if item.device.type != "cpu":
                return False
            storage = item.untyped_storage()
            identity = int(storage._cdata)
            if identity not in seen_storages:
                seen_storages.add(identity)
                raw = torch.as_tensor(storage, dtype=torch.uint8)
                checksum = zlib.crc32(raw.numpy()) & 0xFFFFFFFF
                actual[(raw.numel(), checksum)] += 1
            return True
        if isinstance(item, dict):
            identity = id(item)
            if identity in seen_containers:
                return True
            seen_containers.add(identity)
            return all(visit(key) and visit(child) for key, child in item.items())
        if isinstance(item, (list, tuple, set)):
            identity = id(item)
            if identity in seen_containers:
                return True
            seen_containers.add(identity)
            return all(visit(child) for child in item)
        return True

    return visit(value) and actual == expected


def checkpoint_fingerprint(path: str | Path) -> str:
    checkpoint = Path(path).resolve()
    stat = checkpoint.stat()
    return hashlib.sha256(
        f"{checkpoint}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()


def _resize_geometry(width: int, height: int, *, max_input_patches: int) -> tuple[int, int, int, int]:
    """Return resized width/height and right/bottom padding without decoding pixels."""
    if max_input_patches < 4:
        raise ValueError("max_input_patches must be at least 4")
    patch, merge, limit, side = 14, 2, max_input_patches, 512
    factor = patch * merge
    scale = min(1.0, math.sqrt(limit / (max(1.0, width // patch) * max(1.0, height // patch))),
                side * patch / width, side * patch / height)
    while True:
        new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
        new_w, new_h = min(new_w, side * patch), min(new_h, side * patch)
        pad_w, pad_h = (factor - new_w % factor) % factor, (factor - new_h % factor) % factor
        actual_patches = ((new_w + pad_w) // patch) * ((new_h + pad_h) // patch)
        if actual_patches <= limit:
            return new_w, new_h, pad_w, pad_h
        scale *= math.sqrt(limit / actual_patches) * 0.99


def _resize(image: Image.Image, *, max_input_patches: int) -> tuple[Tensor, Tensor]:
    """Kimi-style resize, pad, normalize and patchify with a strict patch cap."""
    image = image.convert("RGB")
    width, height = image.size
    patch = 14
    new_w, new_h, pad_w, pad_h = _resize_geometry(
        width, height, max_input_patches=max_input_patches)
    pixels = np.asarray(image.resize((new_w, new_h), Image.Resampling.BICUBIC), dtype=np.float32)
    pixels = np.pad(pixels, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
    pixels = pixels / 255.0 * 2.0 - 1.0
    h, w = pixels.shape[:2]
    # [H,W,C] -> [number_of_patches,C,14,14], exactly like navit_patchify.
    patches = torch.from_numpy(pixels).view(h // patch, patch, w // patch, patch, 3)
    patches = patches.permute(0, 2, 4, 1, 3).reshape(-1, 3, patch, patch)
    return patches, torch.tensor([[1, h // patch, w // patch]], dtype=torch.long)


def _rope(grid: Tensor, head_dim: int, device: torch.device) -> Tensor:
    """Kimi's 2-D RoPE frequencies for a single image grid."""
    _, height, width = (int(v) for v in grid[0].tolist())
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 4, device=device).float() / head_dim))
    angles = torch.stack((x.reshape(-1, 1) * freq, y.reshape(-1, 1) * freq), dim=-1).reshape(-1, head_dim // 2)
    return torch.polar(torch.ones_like(angles), angles)


def _apply_rope(q: Tensor, k: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    # q/k are [tokens, heads, head_dim] or [batch,tokens,heads,head_dim].
    if freqs.ndim == 2:
        freq = freqs.view((1,) * (q.ndim - 3) + (freqs.shape[0], 1, freqs.shape[1]))
    elif freqs.ndim == 3 and q.ndim == 4:
        freq = freqs.unsqueeze(-2)
    else:
        raise ValueError(f"incompatible q/frequency shapes {tuple(q.shape)} / {tuple(freqs.shape)}")
    def rotate(x: Tensor) -> Tensor:
        z = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(z * freq).flatten(-2).to(x.dtype)
    return rotate(q), rotate(k)


class _Block(nn.Module):
    def __init__(self, hidden: int = 1152, heads: int = 16, intermediate: int = 4304):
        super().__init__()
        self.heads, self.dim = heads, hidden // heads
        self.norm0, self.norm1 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.wqkv, self.wo = nn.Linear(hidden, hidden * 3), nn.Linear(hidden, hidden)
        self.mlp = nn.Module()
        self.mlp.fc0, self.mlp.fc1 = nn.Linear(hidden, intermediate), nn.Linear(intermediate, hidden)

    def forward(self, x: Tensor, freqs: Tensor, valid: Tensor | None = None) -> Tensor:
        residual = x
        qkv = self.wqkv(self.norm0(x)).view(*x.shape[:-1], 3, self.heads, self.dim)
        q, k, v = qkv.unbind(-3)
        q, k = _apply_rope(q, k, freqs)
        # MoonViT attention is bidirectional inside each image.
        if x.ndim == 2:
            attn = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
            attn = attn.transpose(0, 1).reshape_as(x)
        else:
            attention_mask = None if valid is None else valid[:, None, None, :]
            attn = F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3),
                                                   v.permute(0, 2, 1, 3),
                                                   attn_mask=attention_mask).permute(0, 2, 1, 3).reshape_as(x)
        x = residual + self.wo(attn)
        return x + self.mlp.fc1(F.gelu(self.mlp.fc0(self.norm1(x)), approximate="tanh"))


class MoonViT(nn.Module):
    """Faithful image-only MoonViT tower returning merged 4-patch groups."""
    width = 1152

    def __init__(self, *, max_input_patches: int = 1024,
                 tap_layers: Sequence[int] = (), view_mode: str = "full"):
        super().__init__()
        self.max_input_patches = int(max_input_patches)
        taps = tuple(sorted({int(index) for index in tap_layers}))
        if any(index < 0 or index >= 27 for index in taps):
            raise ValueError(f"MoonViT tap layers out of range: {taps}")
        if view_mode not in ("full", "full-quadrants"):
            raise ValueError(f"unsupported MoonViT view mode: {view_mode}")
        self.tap_layers = taps
        self.view_mode = view_mode
        self.feature_stages = len(taps) if taps else 1
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, self.width, kernel_size=14, stride=14)
        self.patch_embed.pos_emb = nn.Module()
        self.patch_embed.pos_emb.weight = nn.Parameter(torch.empty(64, 64, self.width))
        self.encoder = nn.Module()
        self.encoder.blocks = nn.ModuleList(_Block() for _ in range(27))
        self.encoder.final_layernorm = nn.LayerNorm(self.width)

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device: str | torch.device = "cpu",
                        dtype: torch.dtype = torch.bfloat16,
                        max_input_patches: int = 1024,
                        tap_layers: Sequence[int] = (),
                        view_mode: str = "full") -> "MoonViT":
        model = cls(max_input_patches=max_input_patches,
                    tap_layers=tap_layers, view_mode=view_mode)
        checkpoint = Path(path).resolve()
        model.cache_fingerprint = checkpoint_fingerprint(checkpoint)
        state = load_file(str(checkpoint), device="cpu")
        state = {key.removeprefix("vision_tower."): value for key, value in state.items() if key.startswith("vision_tower.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"MoonViT checkpoint mismatch; missing={missing[:3]}, unexpected={unexpected[:3]}")
        return model.to(device=device, dtype=dtype).eval()

    @torch.no_grad()
    def encode_one(self, image: Image.Image) -> Tensor:
        return self.encode_many([image])[0]

    def _encode_patches(self, patches: Tensor, grid: Tensor) -> Tensor:
        """Encode a same-grid patch batch.

        The legacy/default result is ``[B, groups, 4, 1152]``.  With feature
        taps it is ``[B, stages, groups, 4, 1152]``.
        """
        device, dtype = self.patch_embed.proj.weight.device, self.patch_embed.proj.weight.dtype
        if patches.ndim == 4:
            patches = patches.unsqueeze(0)
        batch, count = patches.shape[:2]
        x = self.patch_embed.proj(patches.to(device=device, dtype=dtype).flatten(0, 1)).flatten(1).view(batch, count, self.width)
        height, width = int(grid[0, 1]), int(grid[0, 2])
        pos = self.patch_embed.pos_emb.weight
        pos = F.interpolate(pos.permute(2, 0, 1)[None], size=(height, width), mode="bicubic").squeeze(0).permute(1, 2, 0).reshape(1, count, self.width)
        x = x + pos
        freqs = _rope(grid, 72, device)
        tapped = []
        tap_set = set(self.tap_layers)
        for index, block in enumerate(self.encoder.blocks):
            x = block(x, freqs)
            if index in tap_set:
                tapped.append(self.encoder.final_layernorm(x))
        if not self.tap_layers:
            tapped = [self.encoder.final_layernorm(x)]

        def merge(value: Tensor) -> Tensor:
            return value.view(batch, height // 2, 2, width // 2, 2, self.width).permute(
                0, 1, 3, 2, 4, 5).reshape(batch, -1, 4, self.width)

        merged = [merge(value) for value in tapped]
        return merged[0] if not self.tap_layers else torch.stack(merged, dim=1)

    def _encode_variable(self, prepared: Sequence[tuple[Tensor, Tensor]]) -> list[Tensor]:
        """Encode unlike image grids together with key-padding attention masks."""
        device, dtype = self.patch_embed.proj.weight.device, self.patch_embed.proj.weight.dtype
        counts = [len(patches) for patches, _ in prepared]
        maximum, batch = max(counts), len(prepared)
        padded = torch.zeros(batch, maximum, 3, 14, 14, dtype=torch.float32)
        valid = torch.zeros(batch, maximum, dtype=torch.bool, device=device)
        for i, (patches, _) in enumerate(prepared):
            padded[i, :len(patches)] = patches
            valid[i, :len(patches)] = True
        x = self.patch_embed.proj(
            padded.to(device=device, dtype=dtype).flatten(0, 1)).flatten(1).view(batch, maximum, self.width)
        x = x * valid.unsqueeze(-1)
        positions = torch.zeros_like(x)
        frequencies = torch.ones(batch, maximum, 36, dtype=torch.complex64, device=device)
        pos = self.patch_embed.pos_emb.weight
        for i, ((_, grid), count) in enumerate(zip(prepared, counts)):
            height, width = int(grid[0, 1]), int(grid[0, 2])
            positions[i, :count] = F.interpolate(
                pos.permute(2, 0, 1)[None], size=(height, width), mode="bicubic"
            ).squeeze(0).permute(1, 2, 0).reshape(count, self.width)
            frequencies[i, :count] = _rope(grid, 72, device)
        x = x + positions
        tapped = []
        tap_set = set(self.tap_layers)
        for index, block in enumerate(self.encoder.blocks):
            x = block(x, frequencies, valid)
            if index in tap_set:
                tapped.append(self.encoder.final_layernorm(x))
        if not self.tap_layers:
            tapped = [self.encoder.final_layernorm(x)]
        output = []
        for i, ((_, grid), count) in enumerate(zip(prepared, counts)):
            height, width = int(grid[0, 1]), int(grid[0, 2])
            stages = []
            for value in tapped:
                item = value[i, :count].view(
                    height // 2, 2, width // 2, 2, self.width)
                stages.append(item.permute(0, 2, 1, 3, 4).reshape(
                    -1, 4, self.width))
            output.append(stages[0] if not self.tap_layers else
                          torch.stack(stages, dim=0))
        return output

    @torch.no_grad()
    def _encode_many_full(self, images: Sequence[Image.Image]) -> list[Tensor]:
        """Encode full images in variable-grid batches, preserving caller order."""
        prepared = [_resize(image, max_input_patches=self.max_input_patches) for image in images]
        if not prepared:
            return []
        if len(prepared) == 1:
            patches, grid = prepared[0]
            return [self._encode_patches(patches, grid).squeeze(0)]
        # Avoid padding a tiny panorama all the way to an unrelated 1024-patch
        # image. Offline cache prefill sorts by patch count, so normal training
        # rarely needs more than one same-scale group here.
        order = sorted(range(len(prepared)), key=lambda i: len(prepared[i][0]))
        output: list[Tensor | None] = [None] * len(prepared)
        start = 0
        while start < len(order):
            first_count = len(prepared[order[start]][0])
            end = start + 1
            while end < len(order) and len(prepared[order[end]][0]) <= first_count * 2:
                end += 1
            indices = order[start:end]
            encoded = self._encode_variable([prepared[i] for i in indices])
            if len(encoded) != len(indices):
                raise RuntimeError("MoonViT variable-batch encoder changed batch cardinality")
            for index, item in zip(indices, encoded):
                output[index] = item
            start = end
        if any(item is None for item in output):
            raise RuntimeError("MoonViT variable-batch encoder lost an input")
        return [item for item in output if item is not None]

    @staticmethod
    def _quadrant_views(image: Image.Image) -> list[Image.Image]:
        width, height = image.size
        split_x, split_y = max(1, width // 2), max(1, height // 2)
        boxes = ((0, 0, split_x, split_y), (split_x, 0, width, split_y),
                 (0, split_y, split_x, height), (split_x, split_y, width, height))
        return [image] + [image.crop(box) for box in boxes
                          if box[2] > box[0] and box[3] > box[1]]

    @torch.no_grad()
    def encode_many(self, images: Sequence[Image.Image]) -> list[Tensor]:
        """Encode images, optionally concatenating full and quadrant features."""
        if self.view_mode == "full":
            return self._encode_many_full(images)
        expanded: list[Image.Image] = []
        owners: list[int] = []
        for owner, image in enumerate(images):
            views = self._quadrant_views(image)
            expanded.extend(views)
            owners.extend([owner] * len(views))
        encoded = self._encode_many_full(expanded)
        grouped: list[list[Tensor]] = [[] for _ in images]
        for owner, item in zip(owners, encoded):
            grouped[owner].append(item)
        dimension = 1 if self.tap_layers else 0
        return [torch.cat(items, dim=dimension) for items in grouped]

    @torch.no_grad()
    def forward(self, images: Sequence[Image.Image]) -> list[Tensor]:
        return self.encode_many(images)


class _QueryResamplerBlock(nn.Module):
    """One pre-norm learned-query cross-attention resampler block."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.source_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(
            width, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, queries: Tensor, source: Tensor) -> Tensor:
        normalized_source = self.source_norm(source)
        attended, _ = self.cross_attention(
            self.query_norm(queries), normalized_source, normalized_source,
            need_weights=False)
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class LearnedQueryResampler(nn.Module):
    """Caption-aware residual resampler over the cacheable MoonViT groups.

    The final projection is zero-initialized. Enabling this module on an old
    projector checkpoint is therefore an exact functional no-op until it has
    received an optimizer update, while still allowing the learned queries to
    replace fixed average-pooling decisions over training.
    """

    def __init__(self, output_width: int, query_count: int, *, width: int = 1024,
                 layers: int = 2, heads: int = 8):
        super().__init__()
        if layers < 1 or width < 1 or heads < 1 or width % heads:
            raise ValueError("invalid learned vision resampler geometry")
        self.source_norm = nn.LayerNorm(1152 * 4)
        self.source_projection = nn.Linear(1152 * 4, width)
        self.queries = nn.Parameter(torch.empty(1, query_count, width))
        nn.init.normal_(self.queries, std=width ** -0.5)
        self.blocks = nn.ModuleList(
            [_QueryResamplerBlock(width, heads) for _ in range(layers)])
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, output_width, bias=False)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, pooled: Tensor) -> Tensor:
        source = self.source_projection(self.source_norm(pooled.flatten(2)))
        queries = self.queries.expand(source.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, source)
        return self.output_projection(self.output_norm(queries))


class MoonViTPrefixProjector(nn.Module):
    """Trainable 4-patch MoonViT-to-RWKV prefix bridge with fixed token count."""
    def __init__(self, rwkv_hidden: int = 2560, prefix_tokens: int = 64, *,
                 resampler_layers: int = 0, resampler_width: int = 1024,
                 resampler_heads: int = 8):
        super().__init__()
        self.prefix_tokens = prefix_tokens
        self.norm = nn.LayerNorm(1152)
        self.project = nn.Sequential(nn.Linear(1152 * 4, rwkv_hidden), nn.GELU(), nn.Linear(rwkv_hidden, rwkv_hidden))
        self.position = nn.Parameter(torch.empty(1, prefix_tokens, rwkv_hidden))
        nn.init.normal_(self.position, std=0.02)
        self.resampler = (LearnedQueryResampler(
            rwkv_hidden, prefix_tokens, width=resampler_width,
            layers=resampler_layers, heads=resampler_heads)
            if resampler_layers else None)

    def pool_features(self, item: Tensor) -> Tensor:
        """Reduce variable spatial groups before the trainable projection.

        This is intentionally cacheable: frozen MoonViT emits these 64 groups
        once, while the projector is still free to learn their mapping to RWKV.
        """
        return pool_features(item, self.prefix_tokens)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        pooled_rows = []
        for item in features:
            # Staged caches retain every requested intermediate feature for the
            # layer-matched injector.  The ordinary input bridge uses the
            # deepest stage, which is the closest equivalent to the legacy
            # final MoonViT output.
            if item.ndim == 4:
                item = item[-1]
            if item.shape[0] != self.prefix_tokens:
                item = self.pool_features(item).squeeze(0)
            pooled_rows.append(item)
        pooled = torch.stack(pooled_rows)
        rows = []
        for item in pooled:
            x = self.project(self.norm(item).reshape(item.shape[0], -1)).unsqueeze(0)
            rows.append(x)
        output = torch.cat(rows, dim=0) + self.position
        if self.resampler is not None:
            output = output + self.resampler(pooled)
        return output


def pool_features(item: Tensor, prefix_tokens: int) -> Tensor:
    """Parameter-free spatial pooling shared by training and cache prefill."""
    if item.ndim == 4:
        return torch.stack([
            pool_features(stage, prefix_tokens).squeeze(0) for stage in item
        ], dim=0).unsqueeze(0)
    if item.ndim != 3:
        raise ValueError(f"expected [groups,4,1152] or staged features, got {tuple(item.shape)}")
    return F.adaptive_avg_pool1d(item.flatten(1).transpose(0, 1)[None], prefix_tokens).transpose(1, 2).reshape(
        1, prefix_tokens, 4, 1152)


def feature_cache_key(image_path: str | Path, *, max_input_patches: int,
                      prefix_tokens: int, vision_fingerprint: str = "unknown",
                      source_size: int | None = None,
                      source_mtime_ns: int | None = None,
                      tap_layers: Sequence[int] = (),
                      view_mode: str = "full") -> str:
    image = Path(image_path)
    if source_size is None or source_mtime_ns is None:
        image = image.resolve()
        stat = image.stat()
        source_size, source_mtime_ns = stat.st_size, stat.st_mtime_ns
    source = (f"{CACHE_SCHEMA}|{image}|size={source_size}|mtime={source_mtime_ns}"
              f"|patches={max_input_patches}|prefix={prefix_tokens}"
              f"|vision={vision_fingerprint}")
    # Preserve byte-for-byte legacy keys for the active/default experiment.
    if tap_layers or view_mode != "full":
        source += (f"|taps={','.join(str(int(index)) for index in tap_layers)}"
                   f"|views={view_mode}|staged-v1")
    return hashlib.sha256(source.encode()).hexdigest() + ".pt"
