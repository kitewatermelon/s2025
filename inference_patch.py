import torch
import os
from pathlib import Path
from tqdm.auto import tqdm
import argparse
import matplotlib.pyplot as plt
from monai.bundle import ConfigParser
from monai.utils import set_determinism
from monai.networks.nets import VQVAE
from monai.transforms import SaveImage
from monai.data import DataLoader

from accelerate import Accelerator
from torchvision.utils import make_grid
from dataset import setup_dataloaders # dataset_2_5d.py에 dataloader를 반환하는 함수가 있다고 가정
from models.lvdm.uvit_2d import UViT
from models.lvdm.vdm_2_5d import VDM
from models.lvdm.utils import init_logger

class SlidingWindowMerger:
    def __init__(self, volume_shape):
        # volume_shape: (H, W)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_volume = torch.zeros(volume_shape, dtype=torch.float32, device=device)
        self.weight_map = torch.zeros(volume_shape, dtype=torch.float32, device=device)

        sigma = min(volume_shape) / 6.0
        weights_y = torch.exp(-((torch.arange(volume_shape[0]) - (volume_shape[0] - 1) / 2) ** 2) / (2 * sigma ** 2))
        weights_x = torch.exp(-((torch.arange(volume_shape[1]) - (volume_shape[1] - 1) / 2) ** 2) / (2 * sigma ** 2))
        self.gaussian_weights = weights_y.unsqueeze(1) * weights_x.unsqueeze(0)
        self.gaussian_weights = self.gaussian_weights.to(device)
        print("gaussian_weights shape:", self.gaussian_weights.shape)
        print("Initialized SlidingWindowMerger with volume shape:", volume_shape)
        print("output_volume shape:", self.output_volume.shape)
        print("weight_map shape:", self.weight_map.shape)


    def add_patch(self, patch, origin):
        y_start, x_start = origin[1], origin[2]
        h, w = patch.shape
        print("y_start, x_start:", y_start, x_start)
        print("h, w:", h, w)
        self.output_volume[y_start:y_start+h, x_start:x_start+w] += patch * self.gaussian_weights[:h, :w]
        self.weight_map[y_start:y_start+h, x_start:x_start+w] += self.gaussian_weights[:h, :w]
        print("ouput_volume shape:", self.output_volume.shape)
    def get_result(self):
        return self.output_volume / (self.weight_map + 1e-8)



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
def save_patch_fig(cbct_patch, generated_patch, output_dir, patch_idx):
    """
    CBCT 패치와 생성된 CT 패치를 나란히 시각화하여 저장
    """
    os.makedirs(output_dir, exist_ok=True)
    img_grid = make_grid([cbct_patch.unsqueeze(0), generated_patch.unsqueeze(0)], nrow=2, normalize=True)
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(img_grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.title(f"Patch {patch_idx}: Left=CBCT, Right=Generated CT")
    plt.savefig(os.path.join(output_dir, f"patch_{patch_idx}.png"), bbox_inches="tight", dpi=200)
    plt.close()


def main(cfg):
    DEVICE = cfg["default"]["device"]
    device = torch.device(DEVICE)
    seed = cfg["default"]["device"]
    set_determinism(seed=seed)

    # Check for config and checkpoint files
    cbct_config_path = cfg["paths"]["cbct_vqvae_config"]
    cbct_checkpoint_path = cfg["paths"]["cbct_vq_checkpoint"]
    ct_config_path = cfg["paths"]["ct_vqvae_config"]
    ct_checkpoint_path = cfg["paths"]["ct_vq_checkpoint"]
    for path in [cbct_config_path, cbct_checkpoint_path, ct_config_path, ct_checkpoint_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    print("Loading VQVAE models...")
    cbct_vqvae_config = ConfigParser()
    cbct_vqvae_config.read_config(cbct_config_path)
    ct_vqvae_config = ConfigParser()
    ct_vqvae_config.read_config(ct_config_path)

    def create_vqvae(config_parser):
        num_channels_tuple = tuple(int(x) for x in config_parser["vqvae"]["num_channels"].split(', '))
        downsample_tuple = tuple(tuple(tuple(v) for v in config_parser["vqvae"]["downsample_parameters"].values()))
        upsample_tuple = tuple(tuple(tuple(v) for v in config_parser["vqvae"]["upsample_parameters"].values()))
        return VQVAE(
            spatial_dims=int(config_parser["vqvae"]["spatial_dims"]),
            in_channels=int(config_parser["vqvae"]["in_channels"]),
            out_channels=int(config_parser["vqvae"]["out_channels"]),
            channels=num_channels_tuple,
            num_res_channels=int(config_parser["vqvae"]["num_res_channels"]),
            num_res_layers=int(config_parser["vqvae"]["num_res_layers"]),
            downsample_parameters=downsample_tuple,
            upsample_parameters=upsample_tuple,
            num_embeddings=int(config_parser["vqvae"]["num_embeddings"]),
            embedding_dim=int(config_parser["vqvae"]["embedding_dim"])
        )

    cbct_ae = create_vqvae(cbct_vqvae_config)
    ct_ae = create_vqvae(ct_vqvae_config)

    cbct_ae.load_state_dict(torch.load(cbct_checkpoint_path)["model_state_dict"])
    ct_ae.load_state_dict(torch.load(ct_checkpoint_path)["model_state_dict"])
    cbct_ae.to(device).eval()
    ct_ae.to(device).eval()

    # Load dataloaders
    stage_1_idxs_file = Path(ct_checkpoint_path).parent / "dataset_indices.json"
    if not stage_1_idxs_file.exists():
        raise FileNotFoundError(f"Dataset indices file not found: {stage_1_idxs_file}")
    
    # dataloader로 바로 받도록 수정
    train_dl, val_dl, test_dl = setup_dataloaders(cfg, stage_1_idxs_file, True)

    check_data = next(iter(val_dl))
    check_data = check_data["ct"]
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True):
            if check_data.dim() == 4:  # (B, H, W, D)
                check_data = check_data.unsqueeze(1)
            z = ct_ae.encode_stage_2_inputs(check_data.to(device))
            z = z.squeeze(2)

    print(f"Codebook latent shape: {z.shape}")
    model = UViT(
        img_size=cfg["uvit"]["img_size"],
        patch_size=cfg["uvit"]["patch_size"],
        in_chans=cfg["uvit"]["in_chans"],
        embed_dim=cfg["uvit"]["embed_dim"],
        depth=cfg["uvit"]["depth"],
        num_heads=cfg["uvit"]["num_heads"],
        conv=cfg["uvit"]['conv'],
        # conv=cfg["uvit"],
        gamma_max=cfg["gamma_max"],
        gamma_min=cfg["gamma_min"],
    )
    print(z[0].shape)
    diffusion = VDM(
        model,
        cfg,
        ct_ae,
        image_shape=z.shape
    )
    diffusion.to(device).eval()

    # Load diffusion model checkpoint
    diffusion_checkpoint_path = cfg["paths"]["diffusion_checkpoint"]
    checkpoint = torch.load(diffusion_checkpoint_path, map_location=device)
    diffusion.load_state_dict(checkpoint["model"])

    accelerator = Accelerator(split_batches=True)
    init_logger(accelerator)

    inferencer = Inferencer(
        diffusion_model=diffusion,
        ae_model_ct=ct_ae,
        ae_model_cbct=cbct_ae,
        accelerator=accelerator,
        cfg=cfg
    )

    # Dataloader를 순회하며 배치 단위로 추론 및 병합
    for i, batch in enumerate(tqdm(test_dl, desc="Processing Batches")):
        cbct_batch = batch["cbct"].to(device)
        mask_batch = batch["mask"].to(device)
        subj_id = batch["subj_id"][0]
        mask_batch = mask_batch.to(device)
        origin = batch["origin"]
        print(batch["cbct"].shape, batch["mask"].shape)
        print(f"Processing subject: {subj_id} with origin: {origin}")
        output_dir = f"/mnt/d/synthrad/TRAIN/{subj_id}"
        os.makedirs(output_dir, exist_ok=True)
        final_generated_slices = []
        first_two_slices = cbct_batch[:2, :, :]
        first_two_masks = mask_batch[:2, :, :]
        masked_first_two_slices = first_two_slices.to(device) * first_two_masks
        final_generated_slices.extend(list(masked_first_two_slices.unbind(dim=0)))

        original_shape = tuple(batch["original_shape"][0][1:])  # (H, W)
        # patch_size: 2D patch 크기 (H, W)
        patch_size = tuple(cbct_batch.shape[-2:])  # (H, W)
        merger = SlidingWindowMerger(original_shape)

        
        for j in range(cbct_batch.shape[0]):
            print(f"Processing patch {j+1}/{cbct_batch.shape[0]} for subject {subj_id}")
            five_slice_patch = cbct_batch[j].unsqueeze(0)
            if len(five_slice_patch.shape) == 4:
                five_slice_patch = five_slice_patch.unsqueeze(1)
            generated_ct_patch = inferencer.sample_from_cbct(five_slice_patch, n_sample_steps=20)
            generated_slice = generated_ct_patch[0, 0, 2, :, :]
            masked_generated_slice = generated_slice * mask_batch[j, :, :]
            final_generated_slices.append(masked_generated_slice)
            cbct_patch = five_slice_patch[0, 0, 2, :, :]

    # 저장
            save_patch_fig(cbct_patch, generated_slice, ".patch", j)

            origin = batch["origin"][j].tolist()
            merger.add_patch(generated_slice.squeeze(0).squeeze(0), origin)

        final_generated_volume = merger.get_result()
        last_two_slices = cbct_batch[-2:, :, :]
        last_two_masks = mask_batch[-2:, :, :]
        masked_last_two_slices = last_two_slices.to(device) * last_two_masks
        final_generated_slices.extend(list(masked_last_two_slices.unbind(dim=0)))

        # 3D 볼륨 저장
        saver = SaveImage(output_dir=output_dir, output_postfix="generated", output_ext=".mha")
        saver(final_generated_volume.unsqueeze(0).unsqueeze(0))
        print(f"Generated volume saved to: {output_dir}/generated.mha")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start inference")
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