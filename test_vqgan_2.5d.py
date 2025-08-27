import torch
import os
import pandas as pd
from monai.bundle import ConfigParser
from monai.utils import set_determinism
from monai.metrics import PSNRMetric, SSIMMetric
from dataset_2_5d import setup_datasets_diffusion, setup_dataloaders
import matplotlib.pyplot as plt
import numpy as np

from monai.networks.nets import VQVAE

def extract_middle_slice(images):
    if images.ndim == 5:  # (B,C,D,H,W)
        D = images.shape[2]
        images = images[:, :, D//2, :, :]
    return images

def compute_mae(input_img, recon_img):
    return torch.abs(input_img - recon_img).mean().item()

def build_vqvae(vq_cfg):
    num_channels = tuple(int(x) for x in vq_cfg["vqvae"]["num_channels"].split(', '))
    downsample = tuple(tuple(tuple(v)) for v in vq_cfg["vqvae"]["downsample_parameters"].values())
    upsample = tuple(tuple(tuple(v)) for v in vq_cfg["vqvae"]["upsample_parameters"].values())
    model = VQVAE(
        spatial_dims=int(vq_cfg["vqvae"]["spatial_dims"]),
        in_channels=int(vq_cfg["vqvae"]["in_channels"]),
        out_channels=int(vq_cfg["vqvae"]["out_channels"]),
        channels=num_channels,
        num_res_channels=int(vq_cfg["vqvae"]["num_res_channels"]),
        num_res_layers=int(vq_cfg["vqvae"]["num_res_layers"]),
        downsample_parameters=downsample,
        upsample_parameters=upsample,
        num_embeddings=int(vq_cfg["vqvae"]["num_embeddings"]),
        embedding_dim=int(vq_cfg["vqvae"]["embedding_dim"])
    )
    return model

def main(cfg):
    device = torch.device(cfg["default"]["device"])
    set_determinism(seed=42)

    # Load configs
    cbct_cfg = ConfigParser(); cbct_cfg.read_config(cfg["paths"]["cbct_vqvae_config"])
    ct_cfg = ConfigParser(); ct_cfg.read_config(cfg["paths"]["ct_vqvae_config"])

    # Load dataset
    stage_1_idxs_file = None
    train_dataset, val_dataset, test_dataset = setup_datasets_diffusion(cfg, stage_1_idxs_file)
    train_loader, test_loader = setup_dataloaders(cfg, save_train_idxs=False)
    
    # 첫 배치만 사용
    batch = next(iter(test_loader))
    cbct_imgs, ct_imgs = batch["cbct"].to(device), batch["ct"].to(device)
    cbct_imgs_2d = extract_middle_slice(cbct_imgs)
    ct_imgs_2d = extract_middle_slice(ct_imgs)
    psnr_metric = PSNRMetric(max_val=1.0)  # For data in range [-1, 1]
    ssim_metric = SSIMMetric(data_range=1.0, spatial_dims=2)  # Changed to 2D


    # 결과 저장 리스트
    results = []

    for ckpt in range(200, 1001, 200):
        cbct_ckpt_path = f'checkpoints/2.5D_cbct_256/model_{ckpt}.pt'
        ct_ckpt_path = f'checkpoints/2.5D_ct/model_{ckpt}.pt'
        if not os.path.exists(cbct_ckpt_path) or not os.path.exists(ct_ckpt_path):
            continue
        
        # 모델 초기화
        cbct_ae = build_vqvae(cbct_cfg).to(device).eval()
        ct_ae = build_vqvae(ct_cfg).to(device).eval()
        
        # 가중치 로드
        cbct_ae.load_state_dict(torch.load(cbct_ckpt_path)["model_state_dict"])
        ct_ae.load_state_dict(torch.load(ct_ckpt_path)["model_state_dict"])

        with torch.no_grad():
            cbct_recon = cbct_ae(cbct_imgs)[0]
            ct_recon = ct_ae(ct_imgs)[0]

            cbct_recon_2d = extract_middle_slice(cbct_recon)
            ct_recon_2d = extract_middle_slice(ct_recon)

            # 첫 번째 이미지만
            cbct_in, cbct_out = cbct_imgs_2d[0], cbct_recon_2d[0]
            ct_in, ct_out = ct_imgs_2d[0], ct_recon_2d[0]
            cbct_in = cbct_in.unsqueeze(0)   # (C,H,W) → (1,C,H,W)
            cbct_out = cbct_out.unsqueeze(0)
            ct_in = ct_in.unsqueeze(0)
            ct_out = ct_out.unsqueeze(0)

            psnr_cbct = psnr_metric(y_pred=cbct_out, y=cbct_in).item()
            ssim_cbct = ssim_metric(y_pred=cbct_out, y=cbct_in).item()
            mae_cbct = compute_mae(cbct_out, cbct_in)

            psnr_ct = psnr_metric(y_pred=ct_out, y=ct_in).item()
            ssim_ct = ssim_metric(y_pred=ct_out, y=ct_in).item()
            mae_ct = compute_mae(ct_out, ct_in)

            results.append({
                "checkpoint": ckpt,
                "PSNR_CBCT": psnr_cbct,
                "SSIM_CBCT": ssim_cbct,
                "MAE_CBCT": mae_cbct,
                "PSNR_CT": psnr_ct,
                "SSIM_CT": ssim_ct,
                "MAE_CT": mae_ct
            })
            print(f"Checkpoint {ckpt}: CBCT PSNR={psnr_cbct:.2f}, SSIM={ssim_cbct:.4f} | CT PSNR={psnr_ct:.2f}, SSIM={ssim_ct:.4f}")

    # CSV 저장
    df = pd.DataFrame(results)
    df.to_csv("vqvae_metrics.csv", index=False)
    print("Metrics saved to vqvae_metrics.csv")

    # 가장 좋은 체크포인트
    best_cbct = df.loc[df["PSNR_CBCT"].idxmax()]
    best_ct = df.loc[df["PSNR_CT"].idxmax()]
    print("\nBest CBCT Checkpoint:", best_cbct["checkpoint"], f"PSNR={best_cbct['PSNR_CBCT']:.2f}, SSIM={best_cbct['SSIM_CBCT']:.4f}")
    print("Best CT Checkpoint:", best_ct["checkpoint"], f"PSNR={best_ct['PSNR_CT']:.2f}, SSIM={best_ct['SSIM_CT']:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VQVAE Checkpoints Evaluation")
    parser.add_argument("-c", "--config", type=str, default="configs/diffusion_config.yaml")
    args = parser.parse_args()

    config = ConfigParser()
    config.read_config(args.config)

    main(config)
