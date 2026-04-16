"""lib/base_trainer.py — 2D / 2.5D Trainer 공통 로직.

추출된 공통 메서드:
  sample_conditional  — 두 파일에 완전 동일
  save_checkpoint     — 두 파일에 완전 동일
  validation          — 구조 동일, _val_generate() template method로 분기
  eval                — 구조 동일, _eval_generate() template method로 분기

서브클래스가 반드시 구현:
  _val_generate(data)  → (cbct_2d, ct_gen_2d, ct_gt_2d, loss)
  _eval_generate(batch) → (cbct_2d, ct_gen_2d, ct_gt_2d)
  train()
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchvision.utils import make_grid
from torchvision.models import inception_v3
from monai.metrics import PSNRMetric, SSIMMetric
from monai.metrics.fid import FIDMetric
import wandb


class BaseVDMTrainer:
    # ── 서브클래스가 반드시 설정해야 하는 속성 ───────────────────────────────
    # self.diffusion_model, self.ae_model_ct, self.ae_model_cbct
    # self.opt, self.accelerator, self.cfg
    # self.step, self.num_steps, self.num_samples, self.clip_samples
    # self.save_and_eval_every
    # self.experiment_dir (Path | None), self.writer (SummaryWriter | None)
    # self.validation_dataloader, self.test_dataloader

    # ── 서브클래스 template methods ──────────────────────────────────────────
    def _val_generate(self, data: dict):
        """배치 하나에 대해 검증 생성 수행.

        Returns:
            cbct_2d   (B,1,H,W)
            ct_gen_2d (B,1,H,W)  — 생성된 CT
            ct_gt_2d  (B,1,H,W)  — GT CT
            loss      scalar Tensor
        """
        raise NotImplementedError

    def _eval_generate(self, batch: dict):
        """테스트 배치 하나에 대해 평가 생성 수행.

        Returns:
            cbct_2d   (B,1,H,W)
            ct_gen_2d (B,1,H,W)
            ct_gt_2d  (B,1,H,W)
        """
        raise NotImplementedError

    # ── 공통 메서드 ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample_conditional(self, z_cond: torch.Tensor, n_sample_steps: int) -> torch.Tensor:
        """역확산 샘플링 (conditioning 포함)."""
        z = torch.randn(
            (z_cond.shape[0], *self.diffusion_model.image_shape),
            device=self.accelerator.device,
        )
        steps = torch.linspace(1.0, 0.0, n_sample_steps + 1, device=self.accelerator.device)
        disable = not self.accelerator.is_main_process
        for i in tqdm(range(n_sample_steps), desc="Sampling", leave=False, disable=disable):
            z = self.diffusion_model.sample_p_s_t(
                z, steps[i], steps[i + 1],
                clip_samples=self.clip_samples,
                context=z_cond,
            )
        return z

    def save_checkpoint(self) -> None:
        ckpt = {
            "step": self.step,
            "model": self.accelerator.unwrap_model(self.diffusion_model).state_dict(),
            "opt": self.opt.state_dict(),
        }
        if hasattr(self, "ema") and self.accelerator.is_main_process:
            ckpt["ema"] = self.ema.state_dict()
        torch.save(ckpt, self.experiment_dir / f"model_{self.step}.pt")
        torch.save(ckpt, self.experiment_dir / "latest.pt")
        cp = self.experiment_dir / "config.yaml"
        if not cp.exists():
            self.cfg.export_config_file(self.cfg.get_parsed_content(), str(cp), fmt="yaml")

    def validation(self) -> float:
        print("=" * 30 + " VALID " + "=" * 30)
        self.diffusion_model.eval()
        psnr_m = PSNRMetric(max_val=1.0)
        ssim_m = SSIMMetric(data_range=1.0, spatial_dims=2)
        val_losses = []

        with torch.no_grad():
            for data in self.validation_dataloader:
                cbct_2d, ct_gen_2d, ct_gt_2d, loss = self._val_generate(data)
                psnr_m(ct_gen_2d, ct_gt_2d)
                ssim_m(ct_gen_2d, ct_gt_2d)
                val_losses.append(loss.item())

        val_psnr = psnr_m.aggregate().item()
        val_ssim = ssim_m.aggregate().item()
        val_loss = sum(val_losses) / len(val_losses)

        if self.writer is not None:
            self.writer.add_scalar("validation/loss", val_loss, self.step)
            self.writer.add_scalar("validation/psnr", val_psnr, self.step)
            self.writer.add_scalar("validation/ssim", val_ssim, self.step)
            if ct_gen_2d.shape[0] > 0:
                img_grid = make_grid(
                    [cbct_2d[0], ct_gen_2d[0], ct_gt_2d[0]], nrow=3, normalize=True
                )
                self.writer.add_image("validation/samples", img_grid, self.step)

        if self.cfg["default"]["make_logs"]:
            wandb.log(
                {"val/loss": val_loss, "val/psnr": val_psnr, "val/ssim": val_ssim},
                step=self.step,
            )

        print(f"\nValidation @ step {self.step}: PSNR={val_psnr:.4f}, SSIM={val_ssim:.4f}, Loss={val_loss:.4f}")

        if self.cfg["default"]["make_logs"]:
            self.save_checkpoint()

        self.diffusion_model.train()
        return val_loss

    def eval(self) -> None:
        """테스트셋 평가: PSNR / SSIM / RMSE / FID + qualitative grid 저장."""
        self.diffusion_model.eval()

        psnr_m = PSNRMetric(max_val=1.0)
        ssim_m = SSIMMetric(data_range=1.0, spatial_dims=2)
        fid_metric = FIDMetric()

        inception = inception_v3(pretrained=True, transform_input=False).to(self.accelerator.device)
        inception.fc = torch.nn.Identity()
        inception.eval()

        def get_feats(x: torch.Tensor) -> torch.Tensor:
            x = x.clamp(0, 1).repeat(1, 3, 1, 1)
            with torch.no_grad():
                return inception(x).flatten(1)

        feats_real, feats_fake, qual_imgs = [], [], []
        mse_sum, n_seen, samples_done = 0.0, 0, 0

        with torch.no_grad():
            for batch in self.test_dataloader:
                cbct_2d, ct_gen_2d, ct_gt_2d = self._eval_generate(batch)

                psnr_m(ct_gen_2d, ct_gt_2d)
                ssim_m(ct_gen_2d, ct_gt_2d)
                mse_sum += F.mse_loss(ct_gen_2d, ct_gt_2d, reduction="sum").item()
                n_seen  += ct_gt_2d.numel()
                feats_real.append(get_feats(ct_gt_2d))
                feats_fake.append(get_feats(ct_gen_2d))

                for i in range(cbct_2d.size(0)):
                    if len(qual_imgs) >= 4 * self.num_samples:
                        break
                    diff = (ct_gen_2d[i] - ct_gt_2d[i]).abs()
                    qual_imgs.extend(
                        t.detach().cpu() for t in (cbct_2d[i], ct_gen_2d[i], ct_gt_2d[i], diff)
                    )
                samples_done += cbct_2d.size(0)
                if samples_done >= self.num_samples:
                    break

        if qual_imgs and self.experiment_dir is not None:
            grid = make_grid(
                torch.stack(qual_imgs), nrow=4, normalize=False, value_range=(0, 1)
            )
            plt.figure(figsize=(10, 3 * self.num_samples))
            plt.axis("off")
            plt.imshow(grid.permute(1, 2, 0), cmap="gray")
            plt.savefig(
                self.experiment_dir / f"qualitative_step-{self.step}.png",
                bbox_inches="tight", dpi=200,
            )
            plt.close()

        psnr_val = psnr_m.aggregate().item()
        ssim_val = ssim_m.aggregate().item()
        mse_val  = mse_sum / n_seen
        rmse_val = float(np.sqrt(mse_val))
        fid_val  = fid_metric(torch.vstack(feats_fake), torch.vstack(feats_real)).item()

        results = dict(psnr=psnr_val, ssim=ssim_val, rmse=rmse_val, fid=fid_val)

        if self.experiment_dir is not None:
            (self.experiment_dir / f"test_metrics_{self.step}.json").write_text(
                json.dumps(results, indent=2)
            )

        if self.cfg["default"]["make_logs"]:
            wandb.log(
                {"test/psnr": psnr_val, "test/ssim": ssim_val,
                 "test/rmse": rmse_val, "test/fid": fid_val},
                step=self.step,
            )

        print(
            f"\nTEST ▸  PSNR {psnr_val:.3f}  SSIM {ssim_val:.3f}  "
            f"RMSE {rmse_val:.5f}  FID {fid_val:.3f}"
        )
        self.diffusion_model.train()
