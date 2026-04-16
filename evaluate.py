"""evaluate.py — CBCT-to-CT Synthesis 통합 평가/추론 스크립트.

Sub-commands:
  eval        PSNR/SSIM/RMSE/FID 평가 → CSV 저장
  infer       3D 볼륨 추론 (center-slice 방식)
  infer_2d    슬라이스별 추론 → .mha + PNG 저장
  test_vqgan  VQ-GAN 체크포인트 sweep 평가

Usage:
  python evaluate.py eval      --ckpt <ckpt.pt>  --config <cfg.yaml> [--split test]
  python evaluate.py infer     --config <cfg.yaml>
  python evaluate.py infer_2d  --config <cfg.yaml>
  python evaluate.py test_vqgan --config <cfg.yaml>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from monai.bundle import ConfigParser
from monai.metrics import PSNRMetric, SSIMMetric
from monai.metrics.fid import FIDMetric
from monai.networks.nets import VQVAE
from monai.transforms import SaveImage
from monai.utils import set_determinism
from torchvision.models import inception_v3
from torchvision.utils import make_grid
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto

from dataset import (
    ResizeBackToOriginalShapeD,
    setup_dataloaders,
    setup_dataloaders_volume,
    setup_datasets_diffusion,
    setup_datasets_inference,
)
from models.lvdm.uvit_2d import UViT
from models.lvdm.vdm_2_5d import VDM
from models.lvdm.utils import init_logger


# ── 공유 헬퍼 함수 ─────────────────────────────────────────────────────────────

def build_vqvae(vq_cfg) -> VQVAE:
    """ConfigParser 객체에서 VQVAE 모델 생성 (가중치 미로딩)."""
    num_channels = tuple(int(x) for x in vq_cfg["vqvae"]["num_channels"].split(", "))
    downsample = tuple(
        tuple(tuple(v) for v in vq_cfg["vqvae"]["downsample_parameters"].values())
    )
    upsample = tuple(
        tuple(tuple(v) for v in vq_cfg["vqvae"]["upsample_parameters"].values())
    )
    return VQVAE(
        spatial_dims=int(vq_cfg["vqvae"]["spatial_dims"]),
        in_channels=int(vq_cfg["vqvae"]["in_channels"]),
        out_channels=int(vq_cfg["vqvae"]["out_channels"]),
        channels=num_channels,
        num_res_channels=int(vq_cfg["vqvae"]["num_res_channels"]),
        num_res_layers=int(vq_cfg["vqvae"]["num_res_layers"]),
        downsample_parameters=downsample,
        upsample_parameters=upsample,
        num_embeddings=int(vq_cfg["vqvae"]["num_embeddings"]),
        embedding_dim=int(vq_cfg["vqvae"]["embedding_dim"]),
    )


def load_vqvae(cfg_path: str, ckpt_path: str, device: torch.device) -> VQVAE:
    """VQVAE 설정 파일과 체크포인트 경로로 모델 로드."""
    vqvae_cfg = ConfigParser()
    vqvae_cfg.read_config(cfg_path)
    model = build_vqvae(vqvae_cfg)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    return model.to(device).eval()


def build_diffusion(cfg, device: torch.device,
                    image_shape, ct_ae: Optional[VQVAE] = None) -> VDM:
    """UViT + VDM 생성."""
    uvit = UViT(
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
    return VDM(uvit, cfg, ct_ae, image_shape=image_shape).to(device)


class Inferencer:
    """CBCT → latent → reverse diffusion → CT 추론."""

    def __init__(self, diffusion_model, ae_cbct, ae_ct, accelerator, cfg):
        self.diffusion = diffusion_model
        self.ae_cbct = ae_cbct.eval()
        self.ae_ct = ae_ct.eval()
        self.accelerator = accelerator
        self.clip_samples = cfg.get("clip_samples", False)

    @torch.no_grad()
    def sample(self, cbct_tensor: torch.Tensor,
               n_steps: int = 50) -> torch.Tensor:
        device = self.accelerator.device
        z_cond = self.ae_cbct.encode_stage_2_inputs(cbct_tensor.to(device)).squeeze(2)
        z = torch.randn(z_cond.shape, device=device)
        steps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
        for i in tqdm_auto(range(n_steps), desc="Sampling", leave=False):
            z = self.diffusion.sample_p_s_t(
                z, steps[i], steps[i + 1],
                clip_samples=self.clip_samples,
                context=z_cond,
            )
        return self.ae_ct.decode_stage_2_outputs(z.unsqueeze(2))


@torch.no_grad()
def sample_conditional(diffusion, z_cond, n_steps: int,
                       device: torch.device) -> torch.Tensor:
    """단순 reverse diffusion 샘플링 (Inferencer 없이 사용 가능)."""
    z = torch.randn(z_cond.shape, device=device)
    steps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    for i in tqdm(range(n_steps), desc="Sampling", leave=False):
        z = diffusion.sample_p_s_t(z, steps[i], steps[i + 1],
                                   clip_samples=False, context=z_cond)
    return z


def compute_region_metrics(ct_gen, ct_gt, region_suffix: str) -> dict:
    """region별 PSNR/SSIM/RMSE 계산. 키: psnr_<suffix> 등."""
    psnr_m = PSNRMetric(max_val=1.0)
    ssim_m = SSIMMetric(data_range=1.0, spatial_dims=2)
    psnr_m(ct_gen, ct_gt)
    ssim_m(ct_gen, ct_gt)
    return {
        f"psnr_{region_suffix}": psnr_m.aggregate().item(),
        f"ssim_{region_suffix}": ssim_m.aggregate().item(),
        f"rmse_{region_suffix}": torch.sqrt(F.mse_loss(ct_gen, ct_gt)).item(),
    }


def save_fig(cbct_slice, generated_slice, output_dir: str, idx: int) -> None:
    grid = make_grid([cbct_slice.unsqueeze(0), generated_slice.unsqueeze(0)],
                     nrow=2, normalize=True)
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.title("Left: Input CBCT | Right: Generated CT")
    plt.savefig(os.path.join(output_dir, f"{idx}.png"),
                bbox_inches="tight", dpi=200)
    plt.close()


# ── eval ──────────────────────────────────────────────────────────────────────

def cmd_eval(args):
    """PSNR / SSIM / RMSE / FID 메트릭 계산 후 CSV 저장."""
    cfg = ConfigParser()
    cfg.read_config(args.config)
    device = torch.device(
        f"cuda:{cfg['default']['device']}" if torch.cuda.is_available() else "cpu"
    )

    cbct_ae = load_vqvae(cfg["paths"]["cbct_vqvae_config"],
                          cfg["paths"]["cbct_vq_checkpoint"], device)
    ct_ae   = load_vqvae(cfg["paths"]["ct_vqvae_config"],
                          cfg["paths"]["ct_vq_checkpoint"], device)

    _, val_loader, test_loader = setup_dataloaders_volume(cfg, save_train_idxs=False)
    loader = val_loader if args.split == "val" else test_loader

    # image_shape: VQ-VAE 인코딩 결과에서 자동 추론
    sample = next(iter(val_loader))
    with torch.no_grad():
        z_shape = ct_ae.encode_stage_2_inputs(
            sample["ct"].to(device).unsqueeze(1) if sample["ct"].dim() == 4
            else sample["ct"].to(device)
        ).squeeze(2)[0].shape

    diffusion = build_diffusion(cfg, device, image_shape=list(z_shape))
    ckpt = torch.load(args.ckpt, map_location=device)
    diffusion.load_state_dict(ckpt["model"])
    diffusion.eval()

    psnr_metric = PSNRMetric(max_val=1.0)
    ssim_metric = SSIMMetric(data_range=1.0, spatial_dims=2)
    fid_metric  = FIDMetric()

    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = torch.nn.Identity()
    inception.eval()

    def inception_feats(x):
        x = x.clamp(0, 1).repeat(1, 3, 1, 1)
        with torch.no_grad():
            return inception(x).flatten(1)

    feats_real, feats_fake = [], []
    mse_sum, n_seen = 0.0, 0
    n_steps = args.n_sample_steps or cfg["n_sample_steps"]

    try:
        region_suffix = cfg["default"]["region_suffix"]
    except (KeyError, TypeError):
        region_suffix = None
    region_psnr_list, region_ssim_list, region_rmse_list = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating [{args.split}]"):
            cbct  = batch["cbct"].to(device)
            ct_gt = batch["ct"].to(device)
            if cbct.dim()  == 4: cbct  = cbct.unsqueeze(1)
            if ct_gt.dim() == 4: ct_gt = ct_gt.unsqueeze(1)

            z_cond   = cbct_ae.encode_stage_2_inputs(cbct).squeeze(2)
            sampled_z = sample_conditional(diffusion, z_cond, n_steps, device).unsqueeze(2)
            ct_gen   = ct_ae.decode_stage_2_outputs(sampled_z)

            # center slice
            ct_gen = ct_gen[:, :, ct_gen.shape[2] // 2, :, :]
            ct_gt  = ct_gt[:,  :, ct_gt.shape[2]  // 2, :, :]

            psnr_metric(ct_gen, ct_gt)
            ssim_metric(ct_gen, ct_gt)
            mse_sum += F.mse_loss(ct_gen, ct_gt, reduction="sum").item()
            n_seen  += ct_gt.numel()
            feats_real.append(inception_feats(ct_gt))
            feats_fake.append(inception_feats(ct_gen))

            if region_suffix:
                rm = compute_region_metrics(ct_gen, ct_gt, region_suffix)
                region_psnr_list.append(rm[f"psnr_{region_suffix}"])
                region_ssim_list.append(rm[f"ssim_{region_suffix}"])
                region_rmse_list.append(rm[f"rmse_{region_suffix}"])

    psnr_val = psnr_metric.aggregate().item()
    ssim_val = ssim_metric.aggregate().item()
    rmse_val = float(np.sqrt(mse_sum / n_seen))
    fid_val  = fid_metric(torch.vstack(feats_fake), torch.vstack(feats_real)).item()

    results = dict(ckpt=str(args.ckpt), config=str(args.config), split=args.split,
                   psnr=psnr_val, ssim=ssim_val, rmse=rmse_val, fid=fid_val)
    if region_suffix and region_psnr_list:
        results[f"psnr_{region_suffix}"] = float(np.mean(region_psnr_list))
        results[f"ssim_{region_suffix}"] = float(np.mean(region_ssim_list))
        results[f"rmse_{region_suffix}"] = float(np.mean(region_rmse_list))

    print(f"\n{'='*55}")
    print(f"  PSNR : {psnr_val:.4f} dB")
    print(f"  SSIM : {ssim_val:.4f}")
    print(f"  RMSE : {rmse_val:.5f}")
    print(f"  FID  : {fid_val:.3f}")
    if region_suffix and region_psnr_list:
        print(f"  PSNR_{region_suffix} : {results[f'psnr_{region_suffix}']:.4f}")
        print(f"  SSIM_{region_suffix} : {results[f'ssim_{region_suffix}']:.4f}")
        print(f"  RMSE_{region_suffix} : {results[f'rmse_{region_suffix}']:.5f}")
    print(f"{'='*55}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
    print(f"\nResults appended to: {output_path}")


# ── infer ─────────────────────────────────────────────────────────────────────

def cmd_infer(args):
    """3D 볼륨 추론 (center-slice 방식). 결과 PNG 저장."""
    cfg = ConfigParser()
    cfg.read_config(args.config)
    device = torch.device(cfg["default"]["device"])
    set_determinism(seed=cfg["default"]["random_seed"])

    cbct_ae = load_vqvae(cfg["paths"]["cbct_vqvae_config"],
                          cfg["paths"]["cbct_vq_checkpoint"], device)
    ct_ae   = load_vqvae(cfg["paths"]["ct_vqvae_config"],
                          cfg["paths"]["ct_vq_checkpoint"], device)

    stage_1_idxs_file = Path(cfg["paths"]["ct_vq_checkpoint"]).parent / "dataset_indices.json"
    if not stage_1_idxs_file.exists():
        raise FileNotFoundError(f"Dataset indices file not found: {stage_1_idxs_file}")
    test_dataset = setup_datasets_inference(cfg, stage_1_idxs_file)

    diffusion = build_diffusion(cfg, device, image_shape=cfg["uvit"]["latent_shape"], ct_ae=ct_ae)
    ckpt = torch.load(cfg["paths"]["diffusion_checkpoint"])
    diffusion.load_state_dict(ckpt["model"])

    accelerator = Accelerator(split_batches=True)
    init_logger(accelerator)
    inferencer = Inferencer(diffusion, cbct_ae, ct_ae, accelerator, cfg)

    sample_data = test_dataset[0]
    sample_cbct = sample_data["cbct"].squeeze(0)   # (D, H, W)
    center_idx  = sample_cbct.shape[0] // 2
    n_slices    = cfg["dataset"]["num_slices"]
    half        = n_slices // 2

    n_slice_window = sample_cbct[center_idx - half: center_idx + half + 1].unsqueeze(0).unsqueeze(0).to(device)
    generated_ct = inferencer.sample(n_slice_window, n_steps=args.n_steps)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    generated_slice = generated_ct[0, 0, half, :, :]
    cbct_slice      = sample_cbct[center_idx]
    save_fig(cbct_slice, generated_slice, output_dir, idx=0)
    print(f"Saved to: {output_dir}")


# ── infer_2d ──────────────────────────────────────────────────────────────────

def cmd_infer_2d(args):
    """슬라이스별 추론 → .mha + PNG 저장. ResizeBackToOriginalShapeD 적용."""
    cfg = ConfigParser()
    cfg.read_config(args.config)
    device = torch.device(cfg["default"]["device"])
    set_determinism(seed=cfg["default"]["random_seed"])

    cbct_ae = load_vqvae(cfg["paths"]["cbct_vqvae_config"],
                          cfg["paths"]["cbct_vq_checkpoint"], device)
    ct_ae   = load_vqvae(cfg["paths"]["ct_vqvae_config"],
                          cfg["paths"]["ct_vq_checkpoint"], device)

    stage_1_idxs_file = Path(cfg["paths"]["ct_vq_checkpoint"]).parent / "dataset_indices.json"
    if not stage_1_idxs_file.exists():
        raise FileNotFoundError(f"Dataset indices file not found: {stage_1_idxs_file}")
    test_dataset = setup_datasets_inference(cfg, stage_1_idxs_file)

    diffusion = build_diffusion(cfg, device, image_shape=cfg["uvit"]["latent_shape"], ct_ae=ct_ae)
    ckpt = torch.load(cfg["paths"]["diffusion_checkpoint"])
    diffusion.load_state_dict(ckpt["model"])

    accelerator = Accelerator(split_batches=True)
    init_logger(accelerator)
    inferencer = Inferencer(diffusion, cbct_ae, ct_ae, accelerator, cfg)

    print(f"Diffusion params : {sum(p.numel() for p in diffusion.parameters()):,}")
    print(f"CBCT AE params   : {sum(p.numel() for p in cbct_ae.parameters()):,}")
    print(f"CT AE params     : {sum(p.numel() for p in ct_ae.parameters()):,}")

    for i in range(len(test_dataset)):
        sample_data = test_dataset[i]
        sample_cbct = sample_data["cbct"].squeeze(0)  # (D, H, W)
        subj_id = sample_data["subj_id"]
        mask = sample_data["mask"].squeeze(0).to(device)
        n_slices = cfg["dataset"]["num_slices"]
        half     = n_slices // 2

        # 앞/뒤 half 슬라이스: CBCT 마스킹값으로 채움
        final_slices = []
        final_slices.extend(list((sample_cbct[:half].to(device) * mask[:half]).unbind(0)))

        for idx in range(half, sample_cbct.shape[0] - half):
            window = sample_cbct[idx - half: idx + half + 1].unsqueeze(0).unsqueeze(0).to(device)
            gen = inferencer.sample(window, n_steps=args.n_steps)
            final_slices.append(gen[0, 0, half, :, :] * mask[idx])

        final_slices.extend(list((sample_cbct[-half:].to(device) * mask[-half:]).unbind(0)))

        # 3D 볼륨 재조합 및 원본 크기 복원
        vol_dhw = torch.stack(final_slices, dim=0)
        sample_data["output"] = vol_dhw.permute(1, 2, 0)                 # H, W, D
        sample_data["cbct"]   = sample_cbct.permute(1, 2, 0)             # H, W, D
        sample_data["output_original_shape"] = sample_data.get("cbct_original_shape")

        if sample_data["output_original_shape"] is None:
            print(f"[WARN] {subj_id}: 원본 shape 정보 없음. Resize 생략.")
            continue

        resized = ResizeBackToOriginalShapeD(keys=["cbct", "output"])(sample_data)
        output_mha = args.output_dir
        os.makedirs(output_mha, exist_ok=True)

        saver = SaveImage(output_dir=output_mha,
                          output_postfix=str(subj_id), output_ext=".mha")
        saver(resized["output"].unsqueeze(0))

        out_vol = resized["output"].permute(2, 0, 1)   # D, H, W
        cbct_vol = resized["cbct"].permute(2, 0, 1)
        for idx in range(out_vol.shape[0]):
            save_fig(cbct_vol[idx].to(device), out_vol[idx].to(device),
                     os.path.join(args.output_dir, "vis", subj_id), idx)

    print(f"Done. Results in: {args.output_dir}")


# ── test_vqgan ────────────────────────────────────────────────────────────────

def cmd_test_vqgan(args):
    """VQ-GAN 체크포인트 sweep (200~1000, step 200) PSNR/SSIM/MAE 평가."""
    cfg = ConfigParser()
    cfg.read_config(args.config)
    device = torch.device(cfg["default"]["device"])
    set_determinism(seed=42)

    cbct_cfg = ConfigParser(); cbct_cfg.read_config(cfg["paths"]["cbct_vqvae_config"])
    ct_cfg   = ConfigParser(); ct_cfg.read_config(cfg["paths"]["ct_vqvae_config"])

    _, _, test_dataset = setup_datasets_diffusion(cfg)
    _, _, test_loader  = setup_dataloaders(cfg, save_train_idxs=False)

    batch = next(iter(test_loader))
    cbct_imgs = batch["cbct"].to(device)
    ct_imgs   = batch["ct"].to(device)

    def mid_slice(imgs):
        if imgs.ndim == 5:
            return imgs[:, :, imgs.shape[2] // 2, :, :]
        return imgs

    cbct_2d = mid_slice(cbct_imgs)
    ct_2d   = mid_slice(ct_imgs)
    psnr_m  = PSNRMetric(max_val=1.0)
    ssim_m  = SSIMMetric(data_range=1.0, spatial_dims=2)

    results = []
    cbct_ckpt_dir = cfg["paths"].get("cbct_ckpt_dir", "checkpoints/2.5D_cbct_256")
    ct_ckpt_dir   = cfg["paths"].get("ct_ckpt_dir",   "checkpoints/2.5D_ct")

    for ckpt in range(args.ckpt_start, args.ckpt_end + 1, args.ckpt_step):
        cbct_path = f"{cbct_ckpt_dir}/model_{ckpt}.pt"
        ct_path   = f"{ct_ckpt_dir}/model_{ckpt}.pt"
        if not os.path.exists(cbct_path) or not os.path.exists(ct_path):
            continue

        cbct_ae = build_vqvae(cbct_cfg).to(device).eval()
        ct_ae   = build_vqvae(ct_cfg).to(device).eval()
        cbct_ae.load_state_dict(torch.load(cbct_path)["model_state_dict"])
        ct_ae.load_state_dict(torch.load(ct_path)["model_state_dict"])

        with torch.no_grad():
            cbct_recon_2d = mid_slice(cbct_ae(cbct_imgs)[0])
            ct_recon_2d   = mid_slice(ct_ae(ct_imgs)[0])

            for tag, inp, out in (("CBCT", cbct_2d, cbct_recon_2d),
                                   ("CT",   ct_2d,   ct_recon_2d)):
                i0, o0 = inp[[0]], out[[0]]
                psnr = psnr_m(y_pred=o0, y=i0).item()
                ssim = ssim_m(y_pred=o0, y=i0).item()
                mae  = torch.abs(o0 - i0).mean().item()
                if tag == "CBCT":
                    row = dict(checkpoint=ckpt,
                               PSNR_CBCT=psnr, SSIM_CBCT=ssim, MAE_CBCT=mae)
                else:
                    row.update(PSNR_CT=psnr, SSIM_CT=ssim, MAE_CT=mae)

        results.append(row)
        print(f"ckpt {ckpt}: CBCT PSNR={row['PSNR_CBCT']:.2f} SSIM={row['SSIM_CBCT']:.4f} | "
              f"CT PSNR={row['PSNR_CT']:.2f} SSIM={row['SSIM_CT']:.4f}")

    if not results:
        print("No checkpoints found.")
        return

    df = pd.DataFrame(results)
    out_csv = args.output or "results/vqvae_metrics.csv"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nMetrics saved to {out_csv}")

    best_cbct = df.loc[df["PSNR_CBCT"].idxmax()]
    best_ct   = df.loc[df["PSNR_CT"].idxmax()]
    print(f"Best CBCT ckpt {int(best_cbct['checkpoint'])}: "
          f"PSNR={best_cbct['PSNR_CBCT']:.2f}  SSIM={best_cbct['SSIM_CBCT']:.4f}")
    print(f"Best CT   ckpt {int(best_ct['checkpoint'])}: "
          f"PSNR={best_ct['PSNR_CT']:.2f}  SSIM={best_ct['SSIM_CT']:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CBCT-to-CT 통합 평가/추론 스크립트")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- eval ----
    p_eval = sub.add_parser("eval", help="PSNR/SSIM/RMSE/FID 메트릭 평가")
    p_eval.add_argument("--ckpt",    required=True, help="diffusion 체크포인트 경로")
    p_eval.add_argument("--config",  required=True, help="config YAML 경로")
    p_eval.add_argument("--split",   choices=["val", "test"], default="test")
    p_eval.add_argument("--output",  default=None,  help="CSV 저장 경로")
    p_eval.add_argument("--n_sample_steps", type=int, default=None)

    # ---- infer ----
    p_inf = sub.add_parser("infer", help="3D center-slice 추론 → PNG 저장")
    p_inf.add_argument("--config",     required=True)
    p_inf.add_argument("--output_dir", default="./results/infer")
    p_inf.add_argument("--n_steps",    type=int, default=50)

    # ---- infer_2d ----
    p_2d = sub.add_parser("infer_2d", help="슬라이스별 추론 → .mha + PNG")
    p_2d.add_argument("--config",     required=True)
    p_2d.add_argument("--output_dir", default="./results/infer_2d")
    p_2d.add_argument("--n_steps",    type=int, default=20)

    # ---- test_vqgan ----
    p_vq = sub.add_parser("test_vqgan", help="VQ-GAN 체크포인트 sweep 평가")
    p_vq.add_argument("--config",     required=True)
    p_vq.add_argument("--output",     default=None, help="CSV 저장 경로")
    p_vq.add_argument("--ckpt_start", type=int, default=200)
    p_vq.add_argument("--ckpt_end",   type=int, default=1000)
    p_vq.add_argument("--ckpt_step",  type=int, default=200)

    args = parser.parse_args()

    # eval: --output 기본값 자동 설정
    if args.command == "eval" and args.output is None:
        tmp = ConfigParser()
        tmp.read_config(args.config)
        args.output = f"results/{tmp['default']['experiment_name']}.csv"

    dispatch = {
        "eval":       cmd_eval,
        "infer":      cmd_infer,
        "infer_2d":   cmd_infer_2d,
        "test_vqgan": cmd_test_vqgan,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
