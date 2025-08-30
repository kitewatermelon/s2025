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

# 이전 답변에서 제공된 SlidingWindowMerger 클래스를 여기에 포함하거나 별도 파일로 임포트
class SlidingWindowMerger:
    def __init__(self, original_shape, patch_size):
        self.original_shape = original_shape
        self.patch_size = patch_size
        self.output_volume = torch.zeros(self.original_shape, dtype=torch.float32)
        self.weight_map = torch.zeros(self.original_shape, dtype=torch.float32)
        sigma = min(self.patch_size) / 6.0
        weights_z = torch.exp(-((torch.arange(self.patch_size[0]) - (self.patch_size[0] - 1) / 2) ** 2) / (2 * sigma ** 2))
        weights_y = torch.exp(-((torch.arange(self.patch_size[1]) - (self.patch_size[1] - 1) / 2) ** 2) / (2 * sigma ** 2))
        weights_x = torch.exp(-((torch.arange(self.patch_size[2]) - (self.patch_size[2] - 1) / 2) ** 2) / (2 * sigma ** 2))
        self.gaussian_weights = weights_z.unsqueeze(1).unsqueeze(2) * weights_y.unsqueeze(0).unsqueeze(2) * weights_x.unsqueeze(0).unsqueeze(1)
        self.gaussian_weights = self.gaussian_weights.to(self.output_volume.device)

    def add_patch(self, patch, origin):
        z_start, y_start, x_start = origin
        dz, dy, dx = self.patch_size
        output_slice = self.output_volume[z_start:z_start+dz, y_start:y_start+dy, x_start:x_start+dx]
        weight_slice = self.weight_map[z_start:z_start+dz, y_start:y_start+dy, x_start:x_start+dx]
        patch_to_add = patch.squeeze() * self.gaussian_weights
        self.output_volume[z_start:z_start+dz, y_start:y_start+dy, x_start:x_start+dx] = output_slice + patch_to_add
        self.weight_map[z_start:z_start+dz, y_start:y_start+dy, x_start:x_start+dx] = weight_slice + self.gaussian_weights
        
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
    train_dl, val_dl, test_dl = setup_dataloaders(cfg, stage_1_idxs_file)

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
        
        output_dir = f"/mnt/d/synthrad/TRAIN/{subj_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # SlidingWindowMerger 초기화
        # MONAI dataloader가 원본 shape 정보를 제공한다고 가정
        original_shape = tuple(batch["cbct_original_shape"][0].tolist())
        patch_size = tuple(cbct_batch.shape[-3:]) # (D, H, W)
        merger = SlidingWindowMerger(original_shape, patch_size)
        
        # 배치 단위로 추론 실행
        # 이 예제에서는 배치 내 패치들이 같은 환자에서 온다고 가정
        # 그렇지 않다면 환자별로 분리하여 처리해야 함
        for j in range(cbct_batch.shape[0]):
            five_slice_patch = cbct_batch[j].unsqueeze(0) # (1, 1, D, H, W)
            generated_ct_patch = inferencer.sample_from_cbct(five_slice_patch, n_sample_steps=20)
            
            # 여기서 generated_ct_patch는 (1, 1, D, H, W) 형태이므로
            # 병합기에 추가할 패치를 (D, H, W) 형태로 만들어야 함
            # 그리고 각 패치의 원본 위치(origin)도 알아야 함
            # 현재 코드에서는 dataloader가 origin을 제공하지 않으므로,
            # 이 부분을 `dataset_2_5d.py`에서 수정해야 합니다.
            # 임시로 더미 origin을 사용합니다.
            origin = (0, 0, 0) # 이 부분은 실제 origin 정보로 대체해야 함
            merger.add_patch(generated_ct_patch.squeeze(0).squeeze(0), origin)
            
        final_generated_volume = merger.get_result()
        
        # 3D 볼륨 저장
        saver = SaveImage(output_dir=output_dir, output_postfix="generated", output_ext=".mha")
        saver(final_generated_volume.unsqueeze(0).unsqueeze(0))
        print(f"Generated volume saved to: {output_dir}/generated.mha")
        
        # 중간 슬라이스 시각화
        mid_slice_idx = final_generated_volume.shape[0] // 2
        generated_slice = final_generated_volume[mid_slice_idx, :, :]
        cbct_slice = batch["cbct_original"][0].squeeze(0)[mid_slice_idx, :, :]
        
        save_fig(cbct_slice, generated_slice, f"/mnt/d/synthrad/inference/{subj_id}", mid_slice_idx)
        print(f"Visualization saved for subject {subj_id}")

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