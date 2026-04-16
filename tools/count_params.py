"""
tools/count_params.py — R2-6 심사 대응용 파라미터 수 계산 스크립트

Usage:
    python tools/count_params.py --config configs/2.5D_diffusion_config.yaml
    python tools/count_params.py --config configs/comparison/unet.yaml --model unet
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from monai.bundle import ConfigParser

from models.lvdm.uvit_2d import UViT
from models.lvdm.unet_denoiser import UNetDenoiser
from models.lvdm.vdm_2_5d import VDM


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def count_uvit(cfg) -> None:
    model = UViT(
        img_size=cfg["uvit"]["img_size"],
        patch_size=cfg["uvit"]["patch_size"],
        in_chans=cfg["uvit"]["in_chans"],
        embed_dim=cfg["uvit"]["embed_dim"],
        depth=cfg["uvit"]["depth"],
        num_heads=cfg["uvit"]["num_heads"],
        conv=cfg["uvit"]["conv"],
        gamma_max=cfg["gamma_max"],
        gamma_min=cfg["gamma_min"],
    )

    # Dummy latent shape to build VDM wrapper
    dummy_shape = (cfg["uvit"]["in_chans"], cfg["uvit"]["img_size"], cfg["uvit"]["img_size"])

    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)

    print("=" * 50)
    print(f"Model : UViT (2.5D VDM denoiser)")
    print(f"Config: {cfg['default']['experiment_name']}")
    print("-" * 50)
    print(f"  Total parameters     : {format_params(total)}  ({total:,})")
    print(f"  Trainable parameters : {format_params(trainable)}  ({trainable:,})")
    print("=" * 50)

    # Per-module breakdown
    print("\nPer-module breakdown:")
    for name, module in model.named_children():
        n = count_parameters(module, trainable_only=False)
        print(f"  {name:<30} {format_params(n):>10}  ({n:,})")


def count_unet(cfg) -> None:
    ud = cfg["unet_denoiser"]
    model = UNetDenoiser(
        img_size=ud["img_size"],
        in_chans=ud["in_chans"],
        base_channels=ud["base_channels"],
        channel_mults=tuple(ud["channel_mults"]),
        num_res_blocks=ud["num_res_blocks"],
        dropout=ud.get("dropout", 0.0),
        gamma_min=cfg["gamma_min"],
        gamma_max=cfg["gamma_max"],
    )

    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)

    print("=" * 50)
    print(f"Model : UNetDenoiser (comparison backbone)")
    print(f"Config: {cfg['default']['experiment_name']}")
    print("-" * 50)
    print(f"  Total parameters     : {format_params(total)}  ({total:,})")
    print(f"  Trainable parameters : {format_params(trainable)}  ({trainable:,})")
    print("=" * 50)

    print("\nPer-module breakdown:")
    for name, module in model.named_children():
        n = count_parameters(module, trainable_only=False)
        print(f"  {name:<30} {format_params(n):>10}  ({n:,})")


def main():
    parser = argparse.ArgumentParser(description="Count model parameters for revision R2-6")
    parser.add_argument("-c", "--config", required=True, help="Path to experiment config .yaml")
    parser.add_argument(
        "--model",
        choices=["uvit", "unet"],
        default="uvit",
        help="Which model architecture to count (default: uvit)",
    )
    args = parser.parse_args()

    cfg = ConfigParser()
    cfg.read_config(args.config)

    if args.model == "uvit":
        count_uvit(cfg)
    elif args.model == "unet":
        count_unet(cfg)


if __name__ == "__main__":
    main()
