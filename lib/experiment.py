"""lib/experiment.py — 실험 디렉터리 · SummaryWriter · WandB 셋업 통합.

두 Trainer.__init__에 중복된 ~20줄 블록을 하나의 함수로 통합.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import wandb
from torch.utils.tensorboard import SummaryWriter


def setup_experiment(cfg) -> tuple[Path | None, SummaryWriter | None]:
    """실험 디렉터리 생성 + SummaryWriter + wandb.init.

    cfg["default"]["make_logs"] == False 이면 (None, None) 반환.

    Returns:
        (experiment_dir, writer)  — make_logs=False 시 (None, None)
    """
    if not cfg["default"]["make_logs"]:
        return None, None

    name = (
        cfg["default"]["experiment_name"]
        if "experiment_name" in cfg.default.keys()
        else f"diffusion@{datetime.now().strftime('%d.%m.%Y-%H:%M')}"
    )

    exp_dir = Path(cfg["default"]["checkpoint_dir"]) / name
    exp_dir.mkdir(exist_ok=True, parents=True)
    cfg.export_config_file(
        cfg.get_parsed_content(), str(exp_dir / "config.yaml"), fmt="yaml"
    )

    writer = SummaryWriter(exp_dir, "tb")

    # wandb — 이미 init 된 경우 재사용
    if wandb.run is None:
        wandb.init(
            project="cbct2ct-revision",
            name=name,
            config=cfg.get_parsed_content(),
            tags=cfg["default"].get("wandb_tags", []),
            resume="allow",
        )

    return exp_dir, writer
