"""train_vqgan.py — VQ-GAN Stage-1 학습 (2D / 2.5D 통합).

Config에서 `vqvae.spatial_dims == 2` 이면 MONAI 파이프라인(setup_dataloaders),
`spatial_dims == 3` 이면 Volume 파이프라인(setup_dataloaders_volume) 으로 자동 분기.
"""
from __future__ import annotations

import argparse
import pathlib
import random

import torch
from monai.bundle import ConfigParser
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.networks.nets import PatchDiscriminator
from monai.utils import set_determinism
from torch.nn import L1Loss
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import Mean
from tqdm.auto import tqdm

from lib.vqvae_utils import build_vqvae
from models.lvdm.utils import get_lr, setup_scheduler


def main(config):
    is_2d = int(config["vqvae"]["spatial_dims"]) == 2

    if is_2d:
        from dataset import setup_dataloaders
    else:
        from dataset import setup_dataloaders_volume as setup_dataloaders
    train_loader, val_loader, _ = setup_dataloaders(config, save_train_idxs=True)

    experiment_name = config["default"].get("experiment_name", "default_experiment")
    checkpoint_dir = pathlib.Path(config["default"]["checkpoint_dir"]) / experiment_name
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    config_path = checkpoint_dir / "config.yaml"
    if not config_path.exists():
        config.export_config_file(config.get_parsed_content(), str(config_path), fmt="yaml")

    writer = SummaryWriter(log_dir=str(checkpoint_dir / "tb"))

    DEVICE = config["default"]["device"] if isinstance(config["default"].get("device"), int) else "cpu"
    seed   = config["default"]["random_seed"] if isinstance(config["default"].get("random_seed"), int) else 42
    set_determinism(seed=seed)

    if DEVICE != "cpu":
        torch.cuda.set_device(DEVICE)
        print("Using GPU#:", torch.cuda.current_device(),
              "Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    device = torch.device(DEVICE)

    model = build_vqvae(config, commitment_cost=0.4).to(device)

    discriminator = PatchDiscriminator(
        spatial_dims=int(config["discriminator"]["spatial_dims"]),
        in_channels=int(config["discriminator"]["in_channels"]),
        num_layers_d=int(config["discriminator"]["num_layers_d"]),
        channels=int(config["discriminator"]["num_channels"]),
    ).to(device)

    perceptual_loss = PerceptualLoss(
        spatial_dims=int(config["discriminator"]["spatial_dims"]),
        network_type="squeeze",
    ).to(device)

    optimizer_g = torch.optim.Adam(model.parameters(),         lr=eval(config["optim"]["lr_generator"]))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=eval(config["optim"]["lr_discriminator"]))

    scheduler_g = setup_scheduler(optimizer_g, config["optim"]["num_epochs"], warmup_epochs=0,
                                  min_lr=eval(config["optim"]["lr_generator"]) * 0.1)
    scheduler_d = setup_scheduler(optimizer_d, config["optim"]["num_epochs"], warmup_epochs=0,
                                  min_lr=eval(config["optim"]["lr_discriminator"]) * 0.1)

    n_epochs        = config["optim"]["num_epochs"]
    val_every       = config["optim"]["val_every"]
    save_interval   = config["optim"]["save_interval"]
    amp_enabled     = config["optim"]["amp"]
    modality_list   = config["dataset"]["modality"]

    l1_loss    = L1Loss()
    adv_loss   = PatchAdversarialLoss(criterion="least_squares")
    adv_w      = 0.01
    percep_w   = 0.001

    def pick_modality(batch_data):
        mod = random.choice(modality_list) if len(modality_list) > 1 else modality_list[0]
        return batch_data[mod].to(device), mod

    def prep(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (data_5d_or_4d, center_2d). For 2D, both are the same tensor."""
        if is_2d:
            return data, data                  # (B,1,H,W)
        if data.dim() == 4:
            data = data.unsqueeze(1)           # (B,D,H,W) → (B,1,D,H,W)
        return data, data[:, :, 2, :, :]      # full 5D, center slice

    epbar = tqdm(range(1, n_epochs + 1), desc="Training Progress", position=0)

    for epoch in epbar:
        model.train()
        discriminator.train()

        train_metrics = {k: Mean(device=device) for k in
                         ("recon_loss", "vq_loss", "gen_loss", "disc_loss", "perceptual_loss", "perplexity")}

        bbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, position=1)
        for batch_data in bbar:
            raw, modality = pick_modality(batch_data)
            data_full, data_2d = prep(raw)

            optimizer_g.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.bfloat16):
                recon, vq_loss = model(images=data_full)
                recon_2d = recon[:, :, 2, :, :] if not is_2d else recon
                logits_fake = discriminator(recon_2d.contiguous().float())[-1]
                recon_loss = l1_loss(recon_2d.float(), data_2d.float())
                p_loss     = perceptual_loss(recon_2d.float(), data_2d.float())
                gen_loss   = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g     = recon_loss + vq_loss + percep_w * p_loss + adv_w * gen_loss
            loss_g.backward()
            optimizer_g.step()

            optimizer_d.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.bfloat16):
                logits_fake = discriminator(recon_2d.contiguous().detach())[-1]
                logits_real = discriminator(data_2d.contiguous().detach())[-1]
                loss_d = adv_w * (
                    adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    + adv_loss(logits_real, target_is_real=True,  for_discriminator=True)
                ) * 0.5
            loss_d.backward()
            optimizer_d.step()

            train_metrics["recon_loss"].update(recon_loss)
            train_metrics["vq_loss"].update(vq_loss)
            train_metrics["gen_loss"].update(gen_loss)
            train_metrics["disc_loss"].update(loss_d)
            train_metrics["perceptual_loss"].update(p_loss)
            train_metrics["perplexity"].update(model.quantizer.perplexity)

            bbar.set_postfix(recon=f"{recon_loss.item():.4f}", vq=f"{vq_loss.item():.4f}",
                             gen=f"{gen_loss.item():.4f}", disc=f"{loss_d.item():.4f}",
                             modality=modality)

        tmv = {k: m.compute().item() for k, m in train_metrics.items()}
        for k, v in tmv.items():
            writer.add_scalar(f"Train/{k}", v, epoch)

        if epoch % val_every == 0:
            model.eval()
            discriminator.eval()
            val_metrics = {k: Mean(device=device) for k in ("recon_loss", "gen_loss", "perceptual_loss")}

            with torch.no_grad():
                for val_step, batch_data in enumerate(tqdm(val_loader, desc="Validating", leave=False)):
                    raw, _ = pick_modality(batch_data)
                    data_full, data_2d = prep(raw)

                    with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.bfloat16):
                        recon, _ = model(images=data_full)
                        recon_2d   = recon[:, :, 2, :, :] if not is_2d else recon
                        recon_loss = l1_loss(recon_2d.float(), data_2d.float())
                        p_loss     = perceptual_loss(recon_2d.float(), data_2d.float())
                        logits_fake = discriminator(recon_2d.contiguous().float())[-1]
                        gen_loss    = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

                    val_metrics["recon_loss"].update(recon_loss)
                    val_metrics["gen_loss"].update(gen_loss)
                    val_metrics["perceptual_loss"].update(p_loss)

                    if val_step == 0:
                        writer.add_images("val/data_axial",  data_2d.float(),  global_step=epoch, dataformats="NCHW")
                        writer.add_images("val/recon_axial", recon_2d.float(), global_step=epoch, dataformats="NCHW")

            vmv = {k: m.compute().item() for k, m in val_metrics.items()}
            for k, v in vmv.items():
                writer.add_scalar(f"Val/{k}", v, epoch)

        epbar.set_postfix(recon=f"{tmv['recon_loss']:.4f}", vq=f"{tmv['vq_loss']:.4f}",
                          gen=f"{tmv['gen_loss']:.4f}", disc=f"{tmv['disc_loss']:.4f}",
                          lr=f"{get_lr(optimizer_g):.2e}")

        scheduler_g.step()
        scheduler_d.step()

        if epoch % save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "scheduler_g_state_dict": scheduler_g.state_dict(),
                "scheduler_d_state_dict": scheduler_d.state_dict(),
            }, checkpoint_dir / f"model_{epoch}.pt")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VQ-GAN Stage 1")
    parser.add_argument("-c", "--config", default="configs/ae_config.yaml")
    args = parser.parse_args()
    config = ConfigParser()
    config.read_config(args.config)
    main(config)
