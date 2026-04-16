"""train_vdm_uvit_2.5d.py — 2.5D Latent Diffusion Model 학습."""
from __future__ import annotations

import argparse

import torch
import wandb
from accelerate import Accelerator
from monai.bundle import ConfigParser
from monai.utils import set_determinism
from tqdm.auto import tqdm

from dataset import setup_dataloaders_volume as setup_dataloaders
from lib.base_trainer import BaseVDMTrainer
from lib.experiment import setup_experiment
from lib.vqvae_utils import load_vqvae
from models.lvdm.uvit_2d import UViT
from models.lvdm.unet_denoiser import UNetDenoiser
from models.lvdm.vdm import VDM
from models.lvdm.utils import init_logger


class Trainer25D(BaseVDMTrainer):
    def __init__(
        self,
        diffusion_model,
        ae_model_ct,
        ae_model_cbct,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        accelerator,
        optimizer,
        cfg,
        num_steps: int = 100_000,
        save_and_eval_every: int = 1000,
        num_samples: int = 8,
        clip_samples: bool = False,
    ):
        self.diffusion_model = diffusion_model
        self.ae_model_ct = ae_model_ct
        self.ae_model_cbct = ae_model_cbct
        self.validation_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.accelerator = accelerator
        self.save_and_eval_every = save_and_eval_every
        self.num_samples = num_samples
        self.cfg = cfg
        self.num_steps = num_steps
        self.clip_samples = clip_samples
        self.step = 0
        self.device = torch.device(cfg["default"]["device"])

        self.experiment_dir, self.writer = setup_experiment(cfg)

        self._train_dl = train_dataloader          # kept for train() loop
        self.diffusion_model = accelerator.prepare(diffusion_model)
        self.opt = accelerator.prepare(optimizer)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_5d(img: torch.Tensor) -> torch.Tensor:
        """(B,D,H,W) → (B,1,D,H,W). Already 5D tensors are returned as-is."""
        return img.unsqueeze(1) if img.dim() == 4 else img

    # ── template methods ──────────────────────────────────────────────────────

    def _val_generate(self, data):
        cbct = self._to_5d(data["cbct"].to(self.device))
        ct   = self._to_5d(data["ct"].to(self.device))
        c    = self.cfg["dataset"]["num_slices"] // 2  # center slice index

        z_target = self.ae_model_ct.encode_stage_2_inputs(ct).squeeze(2)
        z_cond   = self.ae_model_cbct.encode_stage_2_inputs(cbct).squeeze(2)
        loss, _  = self.diffusion_model(z_target, z_cond, ct)

        sampled_z = self.sample_conditional(z_cond, self.cfg["n_sample_steps"]).unsqueeze(2)
        ct_gen    = self.ae_model_ct.decode_stage_2_outputs(sampled_z)[:, :, c, :, :]
        ct_gt     = self.ae_model_ct.decode_stage_2_outputs(z_target.unsqueeze(2))[:, :, c, :, :]
        cbct_2d   = cbct[:, :, c, :, :]
        return cbct_2d, ct_gen, ct_gt, loss

    def _eval_generate(self, batch):
        cbct    = self._to_5d(batch["cbct"].to(self.device))
        ct_gt5d = self._to_5d(batch["ct"].to(self.device))
        c       = self.cfg["dataset"]["num_slices"] // 2  # center slice index

        z_cond    = self.ae_model_cbct.encode_stage_2_inputs(cbct).squeeze(2)
        sampled_z = self.sample_conditional(z_cond, self.cfg["n_sample_steps"]).unsqueeze(2)
        ct_gen    = self.ae_model_ct.decode_stage_2_outputs(sampled_z)[:, :, c, :, :]
        ct_gt     = ct_gt5d[:, :, c, :, :]
        cbct_2d   = cbct[:, :, c, :, :]
        return cbct_2d, ct_gen, ct_gt

    # ── training loop ─────────────────────────────────────────────────────────

    def train(self):
        with tqdm(
            total=self.num_steps,
            desc="Training",
            disable=not self.accelerator.is_main_process,
        ) as pbar:
            while self.step < self.num_steps:
                for data in self._train_dl:
                    cbct_img = self._to_5d(data["cbct"].to(self.device))
                    ct_img   = self._to_5d(data["ct"].to(self.device))

                    with torch.no_grad():
                        z      = self.ae_model_ct.encode_stage_2_inputs(ct_img).squeeze(2)
                        z_cond = self.ae_model_cbct.encode_stage_2_inputs(cbct_img).squeeze(2)

                    self.opt.zero_grad()
                    loss, metrics_tr = self.diffusion_model(z, z_cond, ct_img)
                    self.accelerator.backward(loss)
                    self.opt.step()

                    self.step += 1
                    pbar.set_description(f"loss: {loss.item():.4f}")
                    pbar.update(1)
                    self.accelerator.wait_for_everyone()

                    if self.accelerator.is_main_process:
                        if hasattr(self, "ema"):
                            self.ema.update()
                        if self.step % self.save_and_eval_every == 0:
                            self.validation()
                            self.eval()
                            print(f"Completed step {self.step}/{self.num_steps}")

                    if self.step % 100 == 0 and self.writer is not None:
                        self.writer.add_scalar("train/diff_loss",   metrics_tr["diff_loss"].item(),   self.step)
                        self.writer.add_scalar("train/latent_loss", metrics_tr["latent_loss"].item(), self.step)
                        self.writer.add_scalar("train/recon_loss",  metrics_tr["recon_loss"].item(),  self.step)
                        if self.cfg["default"]["make_logs"]:
                            wandb.log({
                                "train/diff_loss":   metrics_tr["diff_loss"].item(),
                                "train/latent_loss": metrics_tr["latent_loss"].item(),
                                "train/recon_loss":  metrics_tr["recon_loss"].item(),
                            }, step=self.step)

                    if self.step >= self.num_steps:
                        return


def build_backbone(cfg) -> torch.nn.Module:
    """config의 default.backbone 키에 따라 백본 생성. 기본값: uvit."""
    backbone = cfg["default"].get("backbone", "uvit")
    if backbone == "uvit":
        return UViT(
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
    elif backbone == "unet":
        ud = cfg["unet_denoiser"]
        return UNetDenoiser(
            img_size=ud["img_size"],
            in_chans=ud["in_chans"],
            base_channels=ud["base_channels"],
            channel_mults=tuple(ud["channel_mults"]),
            num_res_blocks=ud["num_res_blocks"],
            dropout=ud.get("dropout", 0.0),
            gamma_min=cfg["gamma_min"],
            gamma_max=cfg["gamma_max"],
        )
    else:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose 'uvit' or 'unet'.")


def main(cfg):
    device = torch.device(cfg["default"]["device"])
    set_determinism(seed=cfg["default"]["random_seed"])

    cbct_ae = load_vqvae(cfg["paths"]["cbct_vqvae_config"], cfg["paths"]["cbct_vq_checkpoint"], device)
    ct_ae   = load_vqvae(cfg["paths"]["ct_vqvae_config"],   cfg["paths"]["ct_vq_checkpoint"],   device)

    accelerator = Accelerator(split_batches=True)
    init_logger(accelerator)

    train_dl, val_dl, test_dl = setup_dataloaders(cfg, save_train_idxs=False)

    # Probe latent shape
    check_ct = next(iter(val_dl))["ct"]
    if check_ct.dim() == 4:
        check_ct = check_ct.unsqueeze(1)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        z = ct_ae.encode_stage_2_inputs(check_ct.to(device)).squeeze(2)
    print(f"Codebook latent shape: {z.shape}")

    backbone_type = cfg["default"].get("backbone", "uvit")
    model = build_backbone(cfg)
    print(f"Backbone: {backbone_type}  |  params: {sum(p.numel() for p in model.parameters()):,}")

    diffusion = VDM(model, cfg, ct_ae, image_shape=z[0].shape)
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        eval(cfg["optim"]["lr"]),
        betas=(0.9, 0.99),
        weight_decay=cfg["optim"]["weight_decay"],
        eps=1e-8,
    )

    Trainer25D(
        diffusion, ct_ae, cbct_ae,
        train_dl, val_dl, test_dl,
        accelerator, optimizer, cfg,
        num_steps=cfg["num_steps"],
        save_and_eval_every=cfg["save_and_eval_every"],
    ).train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 2.5D VDM")
    parser.add_argument("-c", "--config", default="configs/2.5D_diffusion_config.yaml")
    args = parser.parse_args()
    config = ConfigParser()
    config.read_config(args.config)
    main(config)
