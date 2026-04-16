"""lib/vqvae_utils.py — VQVAE 빌드 & 로드 공통 유틸.

5곳에 중복된 ~55줄짜리 VQVAE 빌드 블록을 2개 함수로 통합.
사용: train_vqgan.py / train_vdm_uvit_*.py / evaluate.py / tools/count_params.py
"""
from __future__ import annotations

import torch
from monai.bundle import ConfigParser
from monai.networks.nets import VQVAE


def build_vqvae(vqvae_cfg, commitment_cost: float | None = None) -> VQVAE:
    """
    ConfigParser 섹션(vqvae_cfg["vqvae"])에서 VQVAE 인스턴스 생성.

    Args:
        vqvae_cfg: ConfigParser 객체 (vqvae_cfg["vqvae"] 섹션 포함)
    """
    sec = vqvae_cfg["vqvae"]
    num_channels = tuple(int(x) for x in sec["num_channels"].split(", "))
    downsample = tuple(
        tuple(tuple(v) for v in row.values()) if hasattr(row, "values") else tuple(row)
        for row in sec["downsample_parameters"].values()
    )
    upsample = tuple(
        tuple(tuple(v) for v in row.values()) if hasattr(row, "values") else tuple(row)
        for row in sec["upsample_parameters"].values()
    )
    kwargs = dict(
        spatial_dims=int(sec["spatial_dims"]),
        in_channels=int(sec["in_channels"]),
        out_channels=int(sec["out_channels"]),
        channels=num_channels,
        num_res_channels=int(sec["num_res_channels"]),
        num_res_layers=int(sec["num_res_layers"]),
        downsample_parameters=downsample,
        upsample_parameters=upsample,
        num_embeddings=int(sec["num_embeddings"]),
        embedding_dim=int(sec["embedding_dim"]),
    )
    if commitment_cost is not None:
        kwargs["commitment_cost"] = commitment_cost
    return VQVAE(**kwargs)


def load_vqvae(config_path: str, ckpt_path: str, device: torch.device) -> VQVAE:
    """VQVAE config 로드 → 인스턴스 생성 → 가중치 적용 → eval 모드.

    train_vqgan.py가 저장하는 키 이름("model_state_dict")과
    train_vdm_uvit_*.py가 저장하는 키 이름("model") 모두 처리.
    """
    cfg = ConfigParser()
    cfg.read_config(config_path)
    model = build_vqvae(cfg)
    state = torch.load(ckpt_path, map_location=device)
    key = "model_state_dict" if "model_state_dict" in state else "model"
    model.load_state_dict(state[key])
    model.to(device).eval()
    return model
