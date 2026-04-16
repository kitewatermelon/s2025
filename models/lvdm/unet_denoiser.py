"""models/lvdm/unet_denoiser.py — U-Net denoiser backbone for VDM.

UViT의 drop-in replacement. 동일한 인터페이스:
    forward(x, timesteps, context=None) → (B, 1, H, W)

구조:
  - Sinusoidal timestep embedding (UViT와 동일한 gamma 정규화)
  - ResBlock마다 time conditioning: scale+shift (AdaGN 방식)
  - Encoder / Bottleneck / Decoder with skip connections
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── helpers ────────────────────────────────────────────────────────────────────

def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding. UViT와 동일한 스케일(×1000) 및 주파수 배열 사용."""
    assert timesteps.ndim == 1
    timesteps = timesteps * 1000.0
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / half
    )
    args = timesteps[:, None].float() * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


def gn(ch: int) -> nn.GroupNorm:
    """GroupNorm with at most 32 groups."""
    return nn.GroupNorm(min(32, ch), ch)


# ── building blocks ────────────────────────────────────────────────────────────

class ResBlock2D(nn.Module):
    """Residual conv block with timestep scale+shift injection."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch * 2)  # scale + shift
        self.norm2 = gn(out_ch)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip_proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_proj(F.silu(t_emb)).chunk(2, dim=-1)  # (B, out_ch)
        h = self.norm2(h) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip_proj(x)


class DownBlock(nn.Module):
    """N ResBlocks → skip 저장 → stride-2 conv 다운샘플."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int,
                 n_res: int, dropout: float = 0.0):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResBlock2D(in_ch if i == 0 else out_ch, out_ch, time_dim, dropout)
            for i in range(n_res)
        ])
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        for res in self.resnets:
            x = res(x, t_emb)
        return self.down(x), x  # (다운샘플된 피처, skip)


class UpBlock(nn.Module):
    """Bilinear 업샘플 → skip concat → N ResBlocks."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int,
                 n_res: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )
        # 첫 ResBlock: concat(in_ch + skip_ch) → out_ch
        self.resnets = nn.ModuleList([
            ResBlock2D(in_ch + skip_ch if i == 0 else out_ch, out_ch, time_dim, dropout)
            for i in range(n_res)
        ])

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
                t_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        for res in self.resnets:
            x = res(x, t_emb)
        return x


# ── U-Net denoiser ─────────────────────────────────────────────────────────────

class UNetDenoiser(nn.Module):
    """U-Net denoiser backbone for VDM — UViT drop-in replacement.

    Args:
        img_size:       Latent 공간 해상도 (ratio-4 → 32, ratio-2 → 64, ratio-8 → 16).
        in_chans:       noisy latent + context 채널 합 (기본 2 = 1+1).
        base_channels:  기본 채널 수. 각 레벨에서 channel_mults 배수로 확장.
        channel_mults:  인코더 각 레벨의 채널 배수. len() = 다운샘플 횟수.
        num_res_blocks: 인코더/디코더 레벨당 ResBlock 수.
        dropout:        ResBlock 내 dropout 확률.
        gamma_min:      UViT/VDM 동일. 노이즈 스케줄 하한.
        gamma_max:      UViT/VDM 동일. 노이즈 스케줄 상한.

    Shape example (img_size=32, base=128, mults=[1,2,4]):
        Input  : (B, 2, 32, 32)
        Enc 0  : 32→16, skip (32×32, 128ch)
        Enc 1  : 16→8,  skip (16×16, 256ch)
        Enc 2  : 8→4,   skip  (8×8, 512ch)
        Mid    : (4×4, 512ch)
        Dec 2  : ×2→8,  concat skip → (8×8, 256ch)
        Dec 1  : ×2→16, concat skip → (16×16, 128ch)
        Dec 0  : ×2→32, concat skip → (32×32, 128ch)
        Output : (B, 1, 32, 32)
    """

    def __init__(
        self,
        img_size: int = 32,
        in_chans: int = 2,
        base_channels: int = 128,
        channel_mults: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        gamma_min: float = -5.0,
        gamma_max: float = 5.0,
    ):
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.in_chans = in_chans

        n_levels = len(channel_mults)
        enc_ch = [base_channels * m for m in channel_mults]
        time_dim = base_channels * 4
        self._t_sin_dim = base_channels

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input projection
        self.input_proj = nn.Conv2d(in_chans, base_channels, 3, padding=1)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        prev = base_channels
        for i in range(n_levels):
            self.enc_blocks.append(
                DownBlock(prev, enc_ch[i], time_dim, num_res_blocks, dropout)
            )
            prev = enc_ch[i]

        # Bottleneck
        self.mid1 = ResBlock2D(prev, prev, time_dim, dropout)
        self.mid2 = ResBlock2D(prev, prev, time_dim, dropout)

        # Decoder
        self.dec_blocks = nn.ModuleList()
        for j in range(n_levels):
            skip_ch = enc_ch[n_levels - 1 - j]   # skip from matching encoder level
            out_ch  = enc_ch[n_levels - 2 - j] if j < n_levels - 1 else base_channels
            self.dec_blocks.append(
                UpBlock(prev, skip_ch, out_ch, time_dim, num_res_blocks, dropout)
            )
            prev = out_ch

        # Output
        self.output_norm = gn(prev)
        self.output_proj = nn.Conv2d(prev, 1, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:          (B, 1, H, W)  noisy latent
            timesteps:  scalar 또는 (B,)  gamma value (UViT와 동일)
            context:    (B, 1, H, W)  conditioning latent (optional)
        Returns:
            (B, 1, H, W)  predicted x_0
        """
        if context is not None:
            x = torch.cat([x, context], dim=1)  # (B, in_chans, H, W)

        # UViT와 동일한 gamma 정규화
        t = timesteps.expand(x.shape[0])
        t_norm = (t - self.gamma_min) / (self.gamma_max - self.gamma_min)
        t_emb = self.time_embed(timestep_embedding(t_norm, self._t_sin_dim))

        # Encode
        h = self.input_proj(x)
        skips = []
        for blk in self.enc_blocks:
            h, skip = blk(h, t_emb)
            skips.append(skip)

        # Bottleneck
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        # Decode (역순 skip)
        for blk in self.dec_blocks:
            h = blk(h, skips.pop(), t_emb)

        return self.output_proj(F.silu(self.output_norm(h)))
