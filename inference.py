import torch
import torchinfo
import os
from pathlib import Path

from torch.utils.data import Subset

from monai.bundle import ConfigParser
from monai.utils import set_determinism
from datetime import datetime
from ema_pytorch import EMA

import copy
import argparse
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np
from dataset_2_5d import  setup_datasets_inference, setup_dataloaders


from monai.networks.nets import VQVAE
from dataset_2_5d import  setup_datasets_inference, setup_dataloaders
import pdb

import json, matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.nn.functional as F
import torchvision.transforms as T
from monai.metrics import PSNRMetric, SSIMMetric
from monai.metrics.fid import FIDMetric
from torchvision.models import inception_v3

from models.lvdm.uvit_2d import UViT
from models.lvdm.vdm_2_5d import VDM
from models.lvdm.utils import (
    DeviceAwareDataLoader,
    TrainConfig,
    check_config_matches_checkpoint,
    cycle,
    evaluate_model_and_log,
    get_date_str,
    handle_results_path,
    has_int_squareroot,
    init_config_from_args,
    init_logger,
    log,
    make_cifar,
    print_model_summary,
    sample_batched,
)


class Inferencer:
    def __init__(self, diffusion_model, ae_model_ct, ae_model_cbct, accelerator, cfg):
        self.diffusion_model = diffusion_model
        self.ae_model_ct = ae_model_ct.eval()
        self.ae_model_cbct = ae_model_cbct.eval()
        self.accelerator = accelerator
        self.cfg = cfg
        self.clip_samples = cfg.get("clip_samples", False)

    def sample_from_cbct(self, cbct_tensor, n_sample_steps=50):
        device = self.accelerator.device
        with torch.no_grad():
            # Encode CBCT → latent
            z_cond = self.ae_model_cbct.encode_stage_2_inputs(cbct_tensor.to(device))
            z_cond = z_cond.squeeze(2)

            # Random noise init
            z = torch.randn((z_cond.shape[0], *self.diffusion_model.image_shape), device=device)
            print(z.shape, z_cond.shape)
            steps = torch.linspace(1.0, 0.0, n_sample_steps + 1, device=device)

            for i in tqdm(range(n_sample_steps), desc="Sampling"):
                z = self.diffusion_model.sample_p_s_t(
                    z, steps[i], steps[i+1],
                    clip_samples=self.clip_samples,
                    context=z_cond
                )

            # Decode to image
            recon_ct = self.ae_model_ct.decode_stage_2_outputs(z.unsqueeze(2))
        return recon_ct
        

def main(cfg):
    
    # Override config with command line arguments if provided
    DEVICE = cfg["default"]["device"]
    device = torch.device(DEVICE)
    seed = cfg["default"]["device"]
    set_determinism(seed=seed)

    # Check if CBCT VQVAE config and checkpoint files exist
    cbct_config_path = cfg["paths"]["cbct_vqvae_config"]
    cbct_checkpoint_path = cfg["paths"]["cbct_vq_checkpoint"]
    
    if not os.path.exists(cbct_config_path):
        raise FileNotFoundError(f"CBCT VQVAE config file not found: {cbct_config_path}")
    if not os.path.exists(cbct_checkpoint_path):
        raise FileNotFoundError(f"CBCT VQVAE checkpoint file not found: {cbct_checkpoint_path}")
    
    # Check if CT VQVAE config and checkpoint files exist
    ct_config_path = cfg["paths"]["ct_vqvae_config"]
    ct_checkpoint_path = cfg["paths"]["ct_vq_checkpoint"]
    
    if not os.path.exists(ct_config_path):
        raise FileNotFoundError(f"CT VQVAE config file not found: {ct_config_path}")
    if not os.path.exists(ct_checkpoint_path):
        raise FileNotFoundError(f"CT VQVAE checkpoint file not found: {ct_checkpoint_path}")

    print(f"Loading CBCT VQVAE from:")
    print(f"  Config: {cbct_config_path}")
    print(f"  Checkpoint: {cbct_checkpoint_path}")
    
    print(f"Loading CT VQVAE from:")
    print(f"  Config: {ct_config_path}")
    print(f"  Checkpoint: {ct_checkpoint_path}")

    # Load CBCT VQVAE configuration
    cbct_vqvae_config = ConfigParser()
    cbct_vqvae_config.read_config(cbct_config_path)

    # Load CT VQVAE configuration  
    ct_vqvae_config = ConfigParser()
    ct_vqvae_config.read_config(ct_config_path)

    # Create CBCT autoencoder using CBCT config
    cbct_num_channels_tuple = tuple(
        int(x) for x in cbct_vqvae_config["vqvae"]["num_channels"].split(', ')
    )
    cbct_downsample_tuple = tuple(
        tuple(
            tuple(v)
        ) for v in cbct_vqvae_config["vqvae"]["downsample_parameters"].values()
    )
    cbct_upsample_tuple = tuple(
        tuple(
            tuple(v)
        ) for v in cbct_vqvae_config["vqvae"]["upsample_parameters"].values()
    )

    cbct_ae = VQVAE(
        spatial_dims=int(cbct_vqvae_config["vqvae"]["spatial_dims"]),
        in_channels=int(cbct_vqvae_config["vqvae"]["in_channels"]),
        out_channels=int(cbct_vqvae_config["vqvae"]["out_channels"]),
        channels=cbct_num_channels_tuple,
        num_res_channels=int(cbct_vqvae_config["vqvae"]["num_res_channels"]),
        num_res_layers=int(cbct_vqvae_config["vqvae"]["num_res_layers"]),
        downsample_parameters=cbct_downsample_tuple,
        upsample_parameters=cbct_upsample_tuple,
        num_embeddings=int(cbct_vqvae_config["vqvae"]["num_embeddings"]),  # codebook length
        embedding_dim=int(cbct_vqvae_config["vqvae"]["embedding_dim"])
    )

    # Create CT autoencoder using CT config
    ct_num_channels_tuple = tuple(
        int(x) for x in ct_vqvae_config["vqvae"]["num_channels"].split(', ')
    )
    ct_downsample_tuple = tuple(
        tuple(
            tuple(v)
        ) for v in ct_vqvae_config["vqvae"]["downsample_parameters"].values()
    )
    ct_upsample_tuple = tuple(
        tuple(
            tuple(v)
        ) for v in ct_vqvae_config["vqvae"]["upsample_parameters"].values()
    )

    ct_ae = VQVAE(
        spatial_dims=int(ct_vqvae_config["vqvae"]["spatial_dims"]),
        in_channels=int(ct_vqvae_config["vqvae"]["in_channels"]),
        out_channels=int(ct_vqvae_config["vqvae"]["out_channels"]),
        channels=ct_num_channels_tuple,
        num_res_channels=int(ct_vqvae_config["vqvae"]["num_res_channels"]),
        num_res_layers=int(ct_vqvae_config["vqvae"]["num_res_layers"]),
        downsample_parameters=ct_downsample_tuple,
        upsample_parameters=ct_upsample_tuple,
        num_embeddings=int(ct_vqvae_config["vqvae"]["num_embeddings"]),  # codebook length
        embedding_dim=int(ct_vqvae_config["vqvae"]["embedding_dim"])
    )
    cbct_ae.load_state_dict(torch.load(cfg["paths"]["cbct_vq_checkpoint"])["model_state_dict"])
    ct_ae.load_state_dict(torch.load(cfg["paths"]["ct_vq_checkpoint"])["model_state_dict"])

    cbct_ae.to(device)
    ct_ae.to(device)

    cbct_ae.eval()
    ct_ae.eval()
    
    # Use CT VQGAN directory for dataset indices
    stage_1_idxs_file = Path(cfg["paths"]["ct_vq_checkpoint"]).parent / "dataset_indices.json"
    
    # Check if dataset indices file exists
    if not stage_1_idxs_file.exists():
        raise FileNotFoundError(f"Dataset indices file not found: {stage_1_idxs_file}")
    test_dataset =  setup_datasets_inference(cfg, stage_1_idxs_file)

    # train_loader, _ = setup_dataloaders(cfg, save_train_idxs=False)

    # check_data = next(iter(train_loader))

    # with torch.no_grad():
    #     with torch.amp.autocast("cuda", enabled=True):
    #         z = ct_ae.encode_stage_2_inputs(check_data["cbct"].to(device))
    #         z = z.squeeze(2)
    # del train_loader
    
    # print(f"Codebook latent shape: {z.shape}")
    model = UViT(
        img_size=cfg["uvit"]["img_size"],
        patch_size=cfg["uvit"]["patch_size"],
        in_chans=cfg["uvit"]["in_chans"],
        embed_dim=cfg["uvit"]["embed_dim"],
        depth=cfg["uvit"]["depth"],
        num_heads=cfg["uvit"]["num_heads"],
        conv=cfg["uvit"]['conv'],
        gamma_max=cfg["gamma_max"],
        gamma_min=cfg["gamma_min"],
    )
    
    diffusion = VDM(
        model,
        cfg,
        ct_ae,
        image_shape=[6, 1, 32, 32]
    )
    diffusion.to(device)

    # 학습된 diffusion 모델 체크포인트 로딩 (필요하다면 추가)
    checkpoint = torch.load(cfg["paths"]["diffusion_checkpoint"])
    diffusion.load_state_dict(checkpoint["model"])

    accelerator = Accelerator(split_batches=True)
    init_logger(accelerator)

    # --- Inferencer 클래스 생성 및 실행 ---
    inferencer = Inferencer(
        diffusion_model=diffusion,
        ae_model_ct=ct_ae,
        ae_model_cbct=cbct_ae,
        accelerator=accelerator,
        cfg=cfg
    )

    sample_data = test_dataset[0]
    sample_cbct = sample_data["cbct"].squeeze(0)
    print(f"sample_cbct shape: {sample_cbct.shape}")
    
    center_idx = 43
    five_slice = sample_cbct[center_idx - 2 : center_idx + 3]  # shape: (5, 128, 128)
    five_slice = five_slice.unsqueeze(0).unsqueeze(0).to(device)  # shape: (1, 5, 128, 128)
    print(f"sample_cbct shape: {sample_cbct.shape}")  # 예: (87, 128, 128)
    print(f"center_idx: {center_idx}")
    print(f"slice range: {center_idx - 2} to {center_idx + 2}")

    print(five_slice.shape)
    generated_ct = inferencer.sample_from_cbct(five_slice, n_sample_steps=50)


    # 생성된 이미지 저장
    # 저장할 디렉토리 경로 (필요에 따라 수정)
    output_dir = "./Folder"
    os.makedirs(output_dir, exist_ok=True)
    
    generated_slice = generated_ct[0, 0, 2, :, :]
    cbct_slice = sample_cbct[0, 0, 2, :, :]

    img_grid = make_grid([cbct_slice.unsqueeze(0), generated_slice.unsqueeze(0)], nrow=2, normalize=True)
    
    plt.figure(figsize=(8, 4))
    plt.axis("off")
    plt.imshow(img_grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.title("Left: Input CBCT | Right: Generated CT")
    output_filename = os.path.join(output_dir, "generated_sample.png")
    plt.savefig(output_filename, bbox_inches="tight", dpi=200)
    plt.close()
    
    print(f"\nImage successfully generated and saved to: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start train")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="path to configuration *.yaml file",
        required=False,
        default="configs/diffusion_config.yaml"
    )

    args = parser.parse_args()

    config = ConfigParser()
    config.read_config(args.config)

    main(config)
