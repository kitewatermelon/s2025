"""train_vdm_uvit_2d.py — 2D Latent Diffusion Model 학습."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from monai.bundle import ConfigParser
from monai.utils import set_determinism
from tqdm.auto import tqdm

from dataset import setup_datasets_diffusion, setup_dataloaders
from lib.base_trainer import BaseVDMTrainer
from lib.experiment import setup_experiment
from lib.vqvae_utils import load_vqvae
from models.lvdm.uvit_2d import UViT
from models.lvdm.vdm import VDM
from models.lvdm.utils import DeviceAwareDataLoader, cycle, init_logger


class Trainer2D(BaseVDMTrainer):
    def __init__(
        self,
        diffusion_model,
        ae_model_ct,
        ae_model_cbct,
        train_set,
        val_set,
        test_set,
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
        self.accelerator = accelerator
        self.save_and_eval_every = save_and_eval_every
        self.num_samples = num_samples
        self.cfg = cfg
        self.num_steps = num_steps
        self.clip_samples = clip_samples
        self.step = 0

        self.experiment_dir, self.writer = setup_experiment(cfg)

        def make_dataloader(dataset, *, train: bool = False):
            dl = DeviceAwareDataLoader(
                dataset,
                cfg["dataset"]["train_batch_size"],
                shuffle=train,
                pin_memory=True,
                num_workers=cfg["dataset"]["num_workers"],
                drop_last=True,
                device=accelerator.device if not train else None,
            )
            if train:
                dl = accelerator.prepare(dl)
            return dl

        self.train_dataloader = cycle(make_dataloader(train_set, train=True))
        self.validation_dataloader = make_dataloader(val_set)
        self.test_dataloader = make_dataloader(test_set)

        self.diffusion_model = accelerator.prepare(diffusion_model)
        self.opt = accelerator.prepare(optimizer)

    # ── template methods ──────────────────────────────────────────────────────

    def _val_generate(self, data):
        cbct = data["cbct"]
        ct = data["ct"]
        z_target = self.ae_model_ct.encode_stage_2_inputs(ct)
        z_cond = self.ae_model_cbct.encode_stage_2_inputs(cbct)
        loss, _ = self.diffusion_model(z_target, z_cond, ct)
        sampled_z = self.sample_conditional(z_cond, self.cfg["n_sample_steps"])
        ct_gen = self.ae_model_ct.decode_stage_2_outputs(sampled_z)
        ct_gt = self.ae_model_ct.decode_stage_2_outputs(z_target)
        return cbct, ct_gen, ct_gt, loss

    def _eval_generate(self, batch):
        cbct = batch["cbct"]
        ct_gt = batch["ct"]
        z_cond = self.ae_model_cbct.encode_stage_2_inputs(cbct)
        ct_gen = self.ae_model_ct.decode_stage_2_outputs(
            self.sample_conditional(z_cond, self.cfg["n_sample_steps"])
        )
        return cbct, ct_gen, ct_gt

    # ── training loop ─────────────────────────────────────────────────────────

    def train(self):
        with tqdm(
            total=self.num_steps,
            desc="Training",
            disable=not self.accelerator.is_main_process,
        ) as pbar:
            while self.step < self.num_steps:
                data = next(self.train_dataloader)
                cbct_img, ct_img = data["cbct"], data["ct"]

                with torch.no_grad():
                    z = self.ae_model_ct.encode_stage_2_inputs(ct_img)
                    z_cond = self.ae_model_cbct.encode_stage_2_inputs(cbct_img)

                self.opt.zero_grad()
                loss, metrics_tr = self.diffusion_model(z, z_cond, ct_img)
                self.accelerator.backward(loss)
                self.opt.step()
                pbar.set_description(f"loss: {loss.item():.4f}")

                self.step += 1
                self.accelerator.wait_for_everyone()

                if self.accelerator.is_main_process:
                    if hasattr(self, "ema"):
                        self.ema.update()
                    if self.step % self.save_and_eval_every == 0:
                        self.validation()
                        self.eval()
                        print(f"Completed step {self.step}/{self.num_steps}")

                if self.step % 100 == 0 and self.writer is not None:
                    self.writer.add_scalar("train/diff_loss", metrics_tr["diff_loss"].item(), self.step)
                    self.writer.add_scalar("train/latent_loss", metrics_tr["latent_loss"].item(), self.step)
                    self.writer.add_scalar("train/recon_loss", metrics_tr["recon_loss"].item(), self.step)
                    if self.cfg["default"]["make_logs"]:
                        wandb.log({
                            "train/diff_loss": metrics_tr["diff_loss"].item(),
                            "train/latent_loss": metrics_tr["latent_loss"].item(),
                            "train/recon_loss": metrics_tr["recon_loss"].item(),
                        }, step=self.step)

                pbar.update()


def main(cfg):
    device = torch.device(cfg["default"]["device"])
    set_determinism(seed=cfg["default"]["random_seed"])

    cbct_ae = load_vqvae(cfg["paths"]["cbct_vqvae_config"], cfg["paths"]["cbct_vq_checkpoint"], device)
    ct_ae   = load_vqvae(cfg["paths"]["ct_vqvae_config"],   cfg["paths"]["ct_vq_checkpoint"],   device)

    accelerator = Accelerator(split_batches=True)
    init_logger(accelerator)

    stage_1_idxs_file = Path(cfg["paths"]["ct_vq_checkpoint"]).parent / "dataset_indices.json"
    if not stage_1_idxs_file.exists():
        raise FileNotFoundError(f"Dataset indices file not found: {stage_1_idxs_file}")
    train_dataset, val_dataset, test_dataset = setup_datasets_diffusion(cfg, stage_1_idxs_file)

    # Probe latent shape
    probe_loader, _, _ = setup_dataloaders(cfg, save_train_idxs=False)
    check_data = next(iter(probe_loader))
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        z = ct_ae.encode_stage_2_inputs(check_data["ct"].to(device))
    del probe_loader
    print(f"Codebook latent shape: {z.shape}")

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
    diffusion = VDM(model, cfg, ct_ae, image_shape=z[0].shape)
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        eval(cfg["optim"]["lr"]),
        betas=(0.9, 0.99),
        weight_decay=cfg["optim"]["weight_decay"],
        eps=1e-8,
    )

    Trainer2D(
        diffusion, ct_ae, cbct_ae,
        train_dataset, val_dataset, test_dataset,
        accelerator, optimizer, cfg,
        num_steps=cfg["num_steps"],
        save_and_eval_every=cfg["save_and_eval_every"],
    ).train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 2D VDM")
    parser.add_argument("-c", "--config", default="configs/diffusion_config.yaml")
    args = parser.parse_args()
    config = ConfigParser()
    config.read_config(args.config)
    main(config)
