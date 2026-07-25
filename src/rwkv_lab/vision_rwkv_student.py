"""Raw-pixel spatial RWKV student for multi-teacher vision distillation.

The teachers are training-only.  Their cached features are reconciled by the
frozen canonical compressor into a ``128 x 1024`` target.  This module receives
only image pixels and predicts both that target and the ``64 x 2560`` continuous
embedding prefix consumed by the existing frozen 2.9B caption RWKV.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from rwkv_lab.rwkv_pretrain import Block


@dataclass(frozen=True)
class VisionRWKVConfig:
    image_size: int = 512
    grid_size: int = 16
    canonical_tokens: int = 128
    canonical_width: int = 1024
    native_tokens: int = 64
    native_width: int = 2560
    hidden_size: int = 2048
    layers: int = 26
    head_size: int = 64
    ffn_hidden: int = 7168
    checkpoint_blocks: bool = True

    def validate(self) -> None:
        if self.image_size != self.grid_size * 32:
            raise ValueError("pixel stem has stride 32; image_size must be grid_size * 32")
        if self.hidden_size % self.head_size:
            raise ValueError("hidden_size must be divisible by head_size")
        if self.canonical_tokens != self.grid_size * self.grid_size // 2:
            raise ValueError("canonical_tokens must be half the square patch grid")
        if self.canonical_tokens % self.native_tokens:
            raise ValueError("native_tokens must evenly divide canonical_tokens")
        numeric = [value for value in asdict(self).values()
                   if isinstance(value, int) and not isinstance(value, bool)]
        if min(numeric) < 1:
            raise ValueError("all numeric Vision-RWKV dimensions must be positive")


def spatial_orders(grid: int) -> tuple[Tensor, ...]:
    """Four reversible raster scans over one square spatial grid."""
    row = torch.arange(grid * grid, dtype=torch.long).reshape(grid, grid)
    column = row.transpose(0, 1).contiguous()
    return (row.flatten(), row.flatten().flip(0),
            column.flatten(), column.flatten().flip(0))


class PixelStem(nn.Module):
    """Convolutional anti-aliasing stem that emits a fixed 2-D token grid."""
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 7, stride=4, padding=3, bias=False),
            nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 256), nn.GELU(),
            nn.Conv2d(256, width, 4, stride=4, bias=False),
        )

    def forward(self, pixels: Tensor) -> Tensor:
        return self.net(pixels)


class VisionRWKVStudent(nn.Module):
    """Bidirectional/multi-directional spatial RWKV image encoder.

    Every block is a native RWKV-7 TimeMix + ChannelMix block.  Blocks operate
    in alternating row-forward, row-backward, column-forward, and
    column-backward orders, then restore canonical 2-D order before the next
    block.  Thus no one causal raster direction becomes a blind spot.
    """
    def __init__(self, config: VisionRWKVConfig = VisionRWKVConfig()):
        super().__init__()
        config.validate()
        self.config = config
        d, n = config.hidden_size, config.layers
        self.stem = PixelStem(d)
        self.position = nn.Parameter(torch.empty(1, config.grid_size ** 2, d))
        self.geometry = nn.Sequential(nn.Linear(3, d), nn.Tanh(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([
            Block(d, d // config.head_size, config.head_size, index, n, {},
                  ffn_hidden=config.ffn_hidden)
            for index in range(n)
        ])
        self.output_norm = nn.LayerNorm(d)
        self.canonical_norm = nn.LayerNorm(d)
        self.canonical_projection = nn.Linear(d, config.canonical_width)
        self.native_norm = nn.LayerNorm(config.canonical_width)
        self.native_projection = nn.Linear(config.canonical_width,
                                           config.native_width)
        self.native_position = nn.Parameter(torch.empty(
            1, config.native_tokens, config.native_width))
        orders = spatial_orders(config.grid_size)
        for index, order in enumerate(orders):
            self.register_buffer(f"scan_order_{index}", order, persistent=False)
            self.register_buffer(f"scan_inverse_{index}", torch.argsort(order),
                                 persistent=False)
        self.apply(self._init)
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.native_position, std=0.02)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def load_native_head(self, state: dict) -> None:
        """Warm-start the output contract from the completed calibration arm."""
        self.native_norm.load_state_dict(state["output_norm"])
        self.native_projection.load_state_dict(state["output_projection"])
        if tuple(state["output_position"].shape) != tuple(self.native_position.shape):
            raise ValueError("calibrated native position shape does not match student")
        with torch.no_grad():
            self.native_position.copy_(state["output_position"])

    def _run_block(self, block: nn.Module, x: Tensor,
                   v_first: Tensor | None) -> tuple[Tensor, Tensor]:
        if self.training and self.config.checkpoint_blocks:
            if v_first is None:
                return checkpoint(lambda value: block(value, None), x,
                                  use_reentrant=False)
            return checkpoint(lambda value, first: block(value, first),
                              x, v_first, use_reentrant=False)
        return block(x, v_first)

    def forward(self, pixels: Tensor, geometry: Tensor | None = None
                ) -> tuple[Tensor, Tensor]:
        cfg = self.config
        if tuple(pixels.shape[-2:]) != (cfg.image_size, cfg.image_size):
            raise ValueError(f"expected {cfg.image_size}x{cfg.image_size} pixels")
        grid = self.stem(pixels)
        if tuple(grid.shape[-2:]) != (cfg.grid_size, cfg.grid_size):
            raise RuntimeError(f"pixel stem emitted unexpected grid {tuple(grid.shape)}")
        x = grid.flatten(2).transpose(1, 2) + self.position
        if geometry is not None:
            if tuple(geometry.shape) != (pixels.shape[0], 3):
                raise ValueError("geometry must have shape [batch,3]")
            x = x + self.geometry(geometry).unsqueeze(1)

        v_first = None
        for index, block in enumerate(self.blocks):
            direction = index % 4
            order = getattr(self, f"scan_order_{direction}")
            inverse = getattr(self, f"scan_inverse_{direction}")
            x_scan = x.index_select(1, order)
            first_scan = (None if v_first is None
                          else v_first.index_select(1, order))
            x_scan, first_scan = self._run_block(block, x_scan, first_scan)
            x = x_scan.index_select(1, inverse)
            v_first = first_scan.index_select(1, inverse)

        x = self.output_norm(x).transpose(1, 2).reshape(
            pixels.shape[0], cfg.hidden_size, cfg.grid_size, cfg.grid_size)
        # Preserve canonical row-major geometry while reducing 16x16 to 8x16.
        x = F.adaptive_avg_pool2d(x, (cfg.grid_size // 2, cfg.grid_size))
        x = x.flatten(2).transpose(1, 2)
        canonical = self.canonical_projection(self.canonical_norm(x))

        group = cfg.canonical_tokens // cfg.native_tokens
        pooled = canonical.reshape(
            pixels.shape[0], cfg.native_tokens, group,
            cfg.canonical_width).mean(dim=2)
        native = (self.native_projection(self.native_norm(pooled))
                  + self.native_position)
        return canonical, native


def compact_config() -> VisionRWKVConfig:
    """Small, shape-faithful configuration for smoke tests."""
    return VisionRWKVConfig(image_size=64, grid_size=2,
                            canonical_tokens=2, canonical_width=32,
                            native_tokens=1, native_width=64,
                            hidden_size=64, layers=2, head_size=32,
                            ffn_hidden=128, checkpoint_blocks=False)
