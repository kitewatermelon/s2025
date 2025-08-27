import torch
import os
from pathlib import Path

from monai.bundle import ConfigParser
from monai.utils import set_determinism
import argparse
from tqdm.auto import tqdm

from accelerate import Accelerator
from torchvision.utils import make_grid
from dataset_2_5d import  setup_datasets_inference, ResizeBackToOriginalShapeD
from monai.networks.nets import VQVAE
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from models.lvdm.uvit_2d import UViT
from models.lvdm.vdm_2_5d import VDM
from models.lvdm.utils import (
    init_logger,
)
from monai.transforms import SaveImage

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
            z_cond = self.ae_model_cbct.encode_stage_2_inputs(cbct_tensor.to(device))
            z_cond = z_cond.squeeze(2)
            z = torch.randn(z_cond.shape, device=device)          
            steps = torch.linspace(1.0, 0.0, n_sample_steps + 1, device=device)
            for i in tqdm(range(n_sample_steps), desc="Sampling"):
                z = self.diffusion_model.sample_p_s_t(
                    z, steps[i], steps[i+1],
                    clip_samples=self.clip_samples,
                    context=z_cond
                )
            recon_ct = self.ae_model_ct.decode_stage_2_outputs(z.unsqueeze(2))
        return recon_ct

def save_fig(cbct_slice, generated_slice, output_dir, i):
    img_grid = make_grid([cbct_slice.unsqueeze(0), generated_slice.unsqueeze(0)], nrow=2, normalize=True)
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.imshow(img_grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.title("Left: Input CBCT | Right: Generated CT")
    output_filename = os.path.join(output_dir, f"{i}.png")
    plt.savefig(output_filename, bbox_inches="tight", dpi=200)
    plt.close()
    
    # print(f"\nImage successfully generated and saved to: {output_filename}")


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
        image_shape=[6, 1, 64, 64]
    )
    diffusion.to(device)

    # 학습된 diffusion 모델 체크포인트 로딩 (필요하다면 추가)
    checkpoint = torch.load(cfg["paths"]["diffusion_checkpoint"])
    diffusion.load_state_dict(checkpoint["model"])

    accelerator = Accelerator(split_batches=True)
    init_logger(accelerator)
    diffusion_params = sum(p.numel() for p in diffusion.parameters())
    cbct_ae_params = sum(p.numel() for p in cbct_ae.parameters())
    ct_ae_params = sum(p.numel() for p in ct_ae.parameters())

    # 출력
    print(f"Diffusion model parameters: {diffusion_params:,}")
    print(f"CBCT AE parameters: {cbct_ae_params:,}")
    print(f"CT AE parameters: {ct_ae_params:,}")

    # 전체 합
    total_params = diffusion_params + cbct_ae_params + ct_ae_params
    print(f"\nTotal parameters (all combined): {total_params:,}")

    # --- Inferencer 클래스 생성 및 실행 ---
    inferencer = Inferencer(
        diffusion_model=diffusion,
        ae_model_ct=ct_ae,
        ae_model_cbct=cbct_ae,
        accelerator=accelerator,
        cfg=cfg
    )
    for i in range(len(test_dataset)):
        sample_data = test_dataset[i]
        sample_cbct = sample_data["cbct"].squeeze(0)
        subj_id = sample_data["subj_id"]
        mask = sample_data["mask"].squeeze(0)
        mask = mask.to(device)
        print(sample_cbct.shape, sample_data["cbct"].shape, mask.shape)
        final_generated_slices = []
        output_dir = f"/mnt/d/synthrad/TRAIN"
        os.makedirs(output_dir, exist_ok=True)

        first_two_slices = sample_cbct[:2, :, :]
        first_two_masks = mask[:2, :, :]
        masked_first_two_slices = first_two_slices.to(device) * first_two_masks
        final_generated_slices.extend(list(masked_first_two_slices.unbind(dim=0)))

        for idx in range(2, sample_cbct.shape[0] - 2):
            five_slice = sample_cbct[idx - 2 : idx + 3]
            five_slice = five_slice.unsqueeze(0).unsqueeze(0).to(device)

            generated_ct = inferencer.sample_from_cbct(five_slice, n_sample_steps=20)
            generated_slice = generated_ct[0, 0, 2, :, :]

            masked_generated_slice = generated_slice * mask[idx, :, :]
            
            final_generated_slices.append(masked_generated_slice)

        last_two_slices = sample_cbct[-2:, :, :]
        last_two_masks = mask[-2:, :, :]
        masked_last_two_slices = last_two_slices.to(device) * last_two_masks
        final_generated_slices.extend(list(masked_last_two_slices.unbind(dim=0)))

        final_generated_ct = torch.stack(final_generated_slices, dim=0)
        generated_ct_hw_d = final_generated_ct.permute(1, 2, 0)
        sample_data["output"] = generated_ct_hw_d

        sample_data["cbct_original_shape"] = sample_data.get("cbct_original_shape")
        sample_data["output_original_shape"] = sample_data.get("cbct_original_shape")
        sample_data["cbct"] = sample_cbct.permute(1, 2, 0)

        if sample_data["output_original_shape"] is None:
            print("[ERROR] 원본 shape 정보가 없습니다. Resize를 할 수 없습니다.")
        else:
            resized_data = ResizeBackToOriginalShapeD(keys=["cbct", "output"])(sample_data)

            resized_cbct_tensor = resized_data["cbct"]
            resized_output_tensor = resized_data["output"]

            saver = SaveImage(output_dir=output_dir, output_postfix=f"{subj_id}", output_ext=".mha")

            image_to_save = resized_output_tensor.unsqueeze(0)
            saver(image_to_save)
            resized_output_tensor = resized_output_tensor.permute(2, 0, 1)
            resized_cbct_tensor = resized_cbct_tensor.permute(2, 0, 1)

            for idx in range(len(resized_output_tensor)):
                resized_cbct_slice = resized_cbct_tensor[idx, :, :].to(device)
                resized_generated_slice = resized_output_tensor[idx, :, :].to(device)
                save_fig(resized_cbct_slice, resized_generated_slice, f"/mnt/d/synthrad/inference/{subj_id}", idx)

        # ... (생략) ...
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
