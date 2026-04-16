"""dataset.py — 통합 데이터셋 모듈.

두 가지 로딩 방식을 제공합니다.

  - _volume  : SubjectVolumeDataset (SimpleITK 직접 로딩, 3D 볼륨)
               → 2.5D diffusion 학습/평가 (train_vdm_uvit_2.5d, evaluate)
  - (접미사 없음): MONAI LoadImaged 기반 (2D/2.5D 공용)
               → VQ-GAN 학습/테스트, 2D diffusion, inference
               2.5D는 모델 내부에서 인접 슬라이스를 채널로 쌓으므로,
               데이터로더는 3D 볼륨을 그대로 반환하면 충분합니다.
"""
from __future__ import annotations

import json
import os
import pathlib
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from monai.config import KeysCollection
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    Resize,
    Resized,
    ScaleIntensityd,
    ToTensord,
)
from sklearn.model_selection import train_test_split


# ── 공유 Transforms ───────────────────────────────────────────────────────────

class ApplyMaskd(MapTransform):
    """마스크를 이미지에 적용: 배경을 이미지 최솟값으로 채움."""

    def __init__(self, keys: KeysCollection, mask_key: str = "mask",
                 allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key

    def __call__(self, data):
        d = dict(data)
        mask = d[self.mask_key]
        for key in self.keys:
            if key in d:
                img = d[key]
                d[key] = img * mask + img.min() * (1.0 - mask)
        return d


class ClipHUValues(MapTransform):
    """HU 값을 [-1024., 3071] 범위로 클리핑."""

    def __init__(self, keys: KeysCollection):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                d[key] = torch.clamp(d[key], -1024., 3071.)
        return d


# ── Inference 전용 보조 Transforms ────────────────────────────────────────────

class SaveOriginalShapeD(MapTransform):
    """Transform 전 원본 이미지 형태를 <key>_original_shape 에 저장."""

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                d[f"{key}_original_shape"] = np.array(d[key].shape)
        return d


class ResizeBackToOriginalShapeD(MapTransform):
    """SaveOriginalShapeD 로 저장한 크기로 MONAI Resize 복원."""

    def __init__(self, keys, mode: str = "nearest"):
        super().__init__(keys)
        self.mode = mode

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            shape_key = f"{key}_original_shape"
            if shape_key not in d:
                warnings.warn(f"[WARN] Original shape not found for key: {key}, skipping.")
                continue
            target_shape = d[shape_key]
            img = d[key]
            is_tensor = isinstance(img, torch.Tensor)
            device = img.device if is_tensor else None
            if img.ndim == 3:
                img = np.expand_dims(img, axis=0)
            resized = Resize(spatial_size=tuple(target_shape[1:]), mode=self.mode)(
                torch.from_numpy(img) if not is_tensor else img
            ).squeeze(0).cpu().numpy()
            d[key] = torch.from_numpy(resized).to(device) if is_tensor else resized
        return d


class ResizeVolumed(MapTransform):
    """3D 볼륨 (D, H, W) 의 H, W 를 target_hw 크기로 리사이즈.
    image_keys 는 trilinear, mask_keys 는 nearest 사용.
    """

    def __init__(self, keys: KeysCollection, target_hw: Tuple[int, int],
                 mode: str = "bilinear", allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.target_hw = target_hw
        self.mode = mode

    def __call__(self, data):
        import torch.nn.functional as F
        d = dict(data)
        for key in self.keys:
            if key not in d:
                continue
            img = d[key]
            is_tensor = isinstance(img, torch.Tensor)
            if not is_tensor:
                img = torch.from_numpy(np.array(img))
            # (D, H, W) → (1, 1, D, H, W) for F.interpolate → (D, H, W)
            img_5d = img.unsqueeze(0).unsqueeze(0).float()
            interp_mode = "nearest" if self.mode == "nearest" else "trilinear"
            align = None if interp_mode == "nearest" else False
            resized = F.interpolate(
                img_5d,
                size=(img.shape[0], *self.target_hw),
                mode=interp_mode,
                align_corners=align,
            ).squeeze(0).squeeze(0)
            d[key] = resized if is_tensor else resized.numpy()
        return d


class PadToMatchLongestSideD(MapTransform):
    """모든 공간 차원이 최장 변과 같아지도록 패딩. 원본 형태·패딩 정보 저장."""

    def __init__(self, keys, allow_missing_keys: bool = False, value: float = 0):
        super().__init__(keys, allow_missing_keys)
        self.value = value

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key not in d:
                continue
            img = d[key]
            is_tensor = isinstance(img, torch.Tensor)
            if not is_tensor:
                img = torch.from_numpy(img)
            spatial = img.shape[-3:]
            max_dim = max(spatial)
            pads = []
            for s in reversed(spatial):
                diff = max_dim - s
                pads += [diff // 2, diff - diff // 2]
            img_padded = torch.nn.functional.pad(img, tuple(pads), value=self.value)
            d[key] = img_padded if is_tensor else img_padded.numpy()
            d[f"{key}_orig_shape"] = spatial
            d[f"{key}_pad_info"] = tuple(pads)
        return d


class RemovePaddingD(MapTransform):
    """PadToMatchLongestSideD 로 추가한 패딩 제거."""

    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key not in d:
                continue
            if f"{key}_orig_shape" not in d:
                warnings.warn(f"[WARN] No padding info for key {key}. Skipping.")
                continue
            h, w, dpth = d[f"{key}_orig_shape"]
            img = d[key]
            is_tensor = isinstance(img, torch.Tensor)
            if not is_tensor:
                img = torch.from_numpy(img)
            cropped = img[..., :h, :w, :dpth]
            d[key] = cropped if is_tensor else cropped.numpy()
        return d


# ── Volume 전용 Dataset ───────────────────────────────────────────────────────

class SubjectVolumeDataset(Dataset):
    """피험자별 전체 3D 볼륨 데이터셋 (SimpleITK 직접 로딩).

    각 아이템: {"cbct":(D,H,W), "ct":(D,H,W), "mask":(D,H,W), "subj_id":str}
    HU 범위 [-1024, 3071] → [0, 1] 정규화 후 마스킹 적용.
    """

    HU_MIN: float = -1024.0
    HU_MAX: float = 3071.0

    def __init__(self, subject_dirs: List[str], modality: List[str],
                 transform=None):
        self.modality = modality
        self.transform = transform
        valid, skipped = [], 0
        for d in subject_dirs:
            has_mod = all(os.path.exists(os.path.join(d, f"{m}.mha")) for m in modality)
            has_mask = os.path.exists(os.path.join(d, "mask.mha"))
            if has_mod and has_mask:
                valid.append(d)
            else:
                skipped += 1
        if skipped:
            print(f"[SubjectVolumeDataset] {skipped} subject(s) skipped (missing files)")
        self.subject_dirs = valid

    def _load_norm(self, path: str) -> torch.Tensor:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
        arr = np.clip(arr, self.HU_MIN, self.HU_MAX)
        return torch.from_numpy((arr - self.HU_MIN) / (self.HU_MAX - self.HU_MIN))

    def __len__(self) -> int:
        return len(self.subject_dirs)

    def __getitem__(self, idx: int) -> dict:
        case_dir = self.subject_dirs[idx]
        data: dict = {"subj_id": os.path.basename(case_dir)}
        for m in self.modality:
            data[m] = self._load_norm(os.path.join(case_dir, f"{m}.mha"))
        mask_arr = sitk.GetArrayFromImage(
            sitk.ReadImage(os.path.join(case_dir, "mask.mha"))
        ).astype(np.float32)
        data["mask"] = torch.from_numpy(mask_arr)
        if self.transform:
            data = self.transform(data)
        return data


# ── 공유 유틸리티 함수 ────────────────────────────────────────────────────────

def load_dataset_indices(indices_path: str) -> Dict[str, List[int]]:
    with open(indices_path, "r") as f:
        return json.load(f)


def df_to_dict_list(df: pd.DataFrame) -> List[dict]:
    """DataFrame → MONAI Dataset 용 dict 리스트 변환."""
    result = []
    for _, row in df.iterrows():
        entry: dict = {
            "subj_id": row["subj_id"],
            "anatomy": row["anatomy"],
            "mask": pathlib.Path(row["mask"]),
        }
        if "cbct" in row:
            entry["cbct"] = pathlib.Path(row["cbct"])
        if "ct" in row:
            entry["ct"] = pathlib.Path(row["ct"])
        result.append(entry)
    return result


def create_datafiles(
    config: Dict,
    anatomy: List[str] = ["AB", "HN", "TH"],
    modality: List[str] = ["cbct", "ct"],
) -> List[Dict]:
    """anatomy/modality 조합에 따라 파일 경로 dict 리스트 생성."""
    for mod in modality:
        if mod not in ("cbct", "ct"):
            raise ValueError(f"Unknown modality: {mod}")
    anatomy = [a.upper() for a in anatomy]
    for anat in anatomy:
        if anat not in ("AB", "HN", "TH"):
            raise ValueError(f"Unknown anatomy: {anat}")

    files: List[Dict] = []
    data_path = pathlib.Path(config["dataset"]["data_path"])
    for anat in anatomy:
        anat_path = data_path / anat
        if not anat_path.exists():
            print(f"Warning: {anat} directory not found in {data_path}")
            continue
        for subj_dir in sorted(anat_path.glob("*")):
            if not subj_dir.is_dir():
                continue
            required = {"mask": subj_dir / "mask.mha"}
            if "cbct" in modality:
                required["cbct"] = subj_dir / "cbct.mha"
            if "ct" in modality:
                required["ct"] = subj_dir / "ct.mha"
            if all(p.exists() for p in required.values()):
                entry = {"subj_id": subj_dir.name, "anatomy": anat}
                entry.update({k: str(v) for k, v in required.items()})
                files.append(entry)
    return files


def _split_df(df: pd.DataFrame, seed: int):
    """anatomy-stratified 70/12/18 분할 (train / val / test)."""
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["anatomy"], random_state=seed)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.6, stratify=temp_df["anatomy"], random_state=seed)
    return train_df, val_df, test_df


def _save_split_indices(config: Dict, train_df, val_df, test_df) -> None:
    indices = {
        "train": train_df["subj_id"].tolist(),
        "validation": val_df["subj_id"].tolist(),
        "test": test_df["subj_id"].tolist(),
    }
    ckpt_root = (config.get("optim") or config.get("default", {})).get(
        "checkpoint_dir", "checkpoints")
    exp_dir = os.path.join(ckpt_root, config["default"]["experiment_name"])
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "dataset_indices.json"), "w") as f:
        json.dump(indices, f)


# ── Volume (3D) 파이프라인  ───────────────────────────────────────────────────
# SubjectVolumeDataset 기반. 2.5D diffusion 학습/평가에 사용.

def setup_transforms_volume(config: Dict):
    """SimpleITK 직접 로딩 후 마스킹/증강 (SubjectVolumeDataset 용)."""
    modality = config["dataset"].get("modality", ["ct", "cbct"])
    image_keys = [m for m in modality if m in ("ct", "cbct")]
    all_keys = image_keys + ["mask"]
    common = [ApplyMaskd(keys=image_keys)]

    if config["dataset"].get("interpolate", False):
        target_hw = tuple(config["dataset"]["interpolation_size"][-2:])
        common.append(ResizeVolumed(keys=image_keys, target_hw=target_hw, mode="trilinear"))
        common.append(ResizeVolumed(keys=["mask"],   target_hw=target_hw, mode="nearest"))

    if config["dataset"].get("augment", False):
        aug = []
        rc = config["dataset"].get("rand_adj_contrast", {})
        if rc.get("prob", 0.0) > 0:
            aug.append(RandAdjustContrastd(
                keys=image_keys, prob=rc["prob"], gamma=tuple(rc["gamma"])))
        ra = config["dataset"].get("rand_affine", {})
        if ra.get("prob", 0.0) > 0:
            aug.append(RandAffined(
                keys=all_keys, rotate_range=[(-0.26, 0.26)] * 3,
                prob=ra["prob"], padding_mode="border"))
        aug.append(RandFlipd(keys=all_keys, spatial_axis=0, prob=0.5))
        train_transf = Compose(common + aug)
    else:
        train_transf = Compose(common)

    return train_transf, Compose(common)


def setup_dataloaders_volume(
    config: Dict,
    save_train_idxs: bool = False,
    inference_mode: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """SubjectVolumeDataset 기반 DataLoader 3개 반환 (train / val / test)."""
    data_path = pathlib.Path(config["dataset"]["data_path"])
    anatomy_list = [a.upper() for a in config["dataset"]["anatomy"]]

    all_subjects: List[str] = []
    for anat in anatomy_list:
        p = data_path / anat
        if p.exists():
            all_subjects.extend(sorted(str(s) for s in p.glob("*") if s.is_dir()))

    df = pd.DataFrame({"subject_path": all_subjects})
    seed = config["default"]["random_seed"]

    if inference_mode:
        if len(df) > 20:
            df = df.sample(20, random_state=seed)
        train_subjects = df.iloc[:14]["subject_path"].tolist()
        val_subjects   = df.iloc[14:17]["subject_path"].tolist()
        test_subjects  = df.iloc[17:]["subject_path"].tolist()
    else:
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=seed)
        val_df, test_df   = train_test_split(temp_df, test_size=0.6, random_state=seed)
        train_subjects = train_df["subject_path"].tolist()
        val_subjects   = val_df["subject_path"].tolist()
        test_subjects  = test_df["subject_path"].tolist()

    train_transf, val_transf = setup_transforms_volume(config)
    modality = config["dataset"]["modality"]
    train_ds = SubjectVolumeDataset(train_subjects, modality, train_transf)
    val_ds   = SubjectVolumeDataset(val_subjects,   modality, val_transf)
    test_ds  = SubjectVolumeDataset(test_subjects,  modality, val_transf)

    nw = config["dataset"]["num_workers"]
    bs_train = config["dataset"]["train_batch_size"]
    bs_val   = config["dataset"]["val_batch_size"]

    train_loader = DataLoader(train_ds, batch_size=bs_train,
                              shuffle=config["dataset"]["train_shuffle"],
                              num_workers=nw, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs_val,
                              shuffle=False, num_workers=nw, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs_val,
                              shuffle=False, num_workers=nw)

    if save_train_idxs:
        indices = {"train": train_subjects, "validation": val_subjects,
                   "test": test_subjects}
        exp_dir = os.path.join(config["default"]["checkpoint_dir"],
                               config["default"]["experiment_name"])
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, "dataset_indices.json"), "w") as f:
            json.dump(indices, f)

    print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} | "
          f"Test: {len(test_loader)}")
    return train_loader, val_loader, test_loader


def setup_inference_dataloaders(config: Dict) -> DataLoader:
    """Inference 전용 loader (batch_size=1, 전체 피험자, 순서 고정)."""
    data_path = pathlib.Path(config["dataset"]["data_path"])
    all_subjects: List[str] = []
    for anat in [a.upper() for a in config["dataset"]["anatomy"]]:
        p = data_path / anat
        if p.exists():
            all_subjects.extend(sorted(str(s) for s in p.glob("*") if s.is_dir()))
    _, val_transf = setup_transforms_volume(config)
    ds = SubjectVolumeDataset(all_subjects, config["dataset"]["modality"], val_transf)
    return DataLoader(ds, batch_size=1, shuffle=False,
                      num_workers=config["dataset"]["num_workers"])


# ── MONAI 파이프라인 (2D / 2.5D 공용)  ───────────────────────────────────────
# 2D:   각 슬라이스가 독립 샘플 → 모델이 단일 채널 처리
# 2.5D: 3D 볼륨 로딩 → 모델 내부에서 인접 슬라이스를 채널로 쌓음
# → 데이터로더 레벨에서는 동일하게 3D 볼륨을 반환하면 됩니다.

def setup_transforms(config: Dict):
    """MONAI LoadImaged 기반 3D 볼륨 transforms (2D/2.5D 공용)."""
    interpolate = config["dataset"]["interpolate"]
    modality = config["dataset"].get("modality", ["ct", "cbct"])
    image_keys = [m for m in ("cbct", "ct") if m in modality]
    all_keys = image_keys + ["mask"]
    spatial_size_conf = (tuple(config["dataset"]["interpolation_size"])
                         if interpolate else None)

    base = [
        LoadImaged(keys=all_keys, image_only=True,
                   ensure_channel_first=True, reader="ITKReader"),
        ApplyMaskd(keys=image_keys),
        ClipHUValues(keys=image_keys),
        ScaleIntensityd(keys=image_keys,
                        minv=config["dataset"]["minv"],
                        maxv=config["dataset"]["maxv"]),
    ]
    if interpolate:
        base.append(Resized(keys=all_keys, spatial_size=spatial_size_conf))
    base += [EnsureTyped(keys=all_keys, dtype=torch.float), ToTensord(keys=all_keys)]
    val_transf = Compose(base)

    if config["dataset"]["augment"]:
        rand_adj_prob  = config["dataset"].get("rand_adj_contrast", {}).get("prob", 0.0)
        rand_adj_gamma = config["dataset"].get("rand_adj_contrast", {}).get("gamma", (0.5, 1.5))
        rand_aff_prob  = config["dataset"].get("rand_affine",       {}).get("prob", 0.0)
        aug = list(base)
        if rand_adj_prob > 0.0:
            aug.append(RandAdjustContrastd(
                keys=image_keys, prob=rand_adj_prob, gamma=tuple(rand_adj_gamma)))
        if rand_aff_prob > 0.0:
            aug.append(RandAffined(
                keys=all_keys,
                rotate_range=[(-np.pi / 36, np.pi / 36)] * 2,
                translate_range=[(-1, 1)] * 2,
                scale_range=[(-0.05, 0.05)] * 2,
                padding_mode="zeros",
                prob=rand_aff_prob,
            ))
        train_transf = Compose(aug)
    else:
        train_transf = val_transf

    return train_transf, val_transf


def setup_dataloaders(config: Dict, save_train_idxs: bool = False) -> Tuple:
    """MONAI 파이프라인 DataLoader 3개 반환 (train / val / test)."""
    files = create_datafiles(config, anatomy=config["dataset"]["anatomy"],
                             modality=config["dataset"]["modality"])
    df = pd.DataFrame(files)
    train_df, val_df, test_df = _split_df(df, config["default"]["random_seed"])

    train_transf, val_transf = setup_transforms(config)
    train_ds = Dataset(data=df_to_dict_list(train_df), transform=train_transf)
    val_ds   = Dataset(data=df_to_dict_list(val_df),   transform=val_transf)
    test_ds  = Dataset(data=df_to_dict_list(test_df),  transform=val_transf)

    nw = config["dataset"]["num_workers"]
    train_loader = DataLoader(train_ds,
                              batch_size=config["dataset"]["train_batch_size"],
                              shuffle=config["dataset"]["train_shuffle"],
                              num_workers=nw, drop_last=True)
    val_loader   = DataLoader(val_ds,
                              batch_size=config["dataset"]["val_batch_size"],
                              shuffle=config["dataset"].get("val_shuffle", False),
                              num_workers=nw, drop_last=True)
    test_loader  = DataLoader(test_ds,
                              batch_size=config["dataset"]["val_batch_size"],
                              shuffle=False, num_workers=nw)

    if save_train_idxs:
        _save_split_indices(config, train_df, val_df, test_df)
    return train_loader, val_loader, test_loader


def setup_datasets(config: Dict) -> Tuple:
    """MONAI 파이프라인 Dataset 3개 반환 (train / val / test)."""
    files = create_datafiles(config, anatomy=config["dataset"]["anatomy"],
                             modality=config["dataset"]["modality"])
    df = pd.DataFrame(files)
    train_df, val_df, test_df = _split_df(df, config["default"]["random_seed"])
    train_transf, val_transf = setup_transforms(config)
    return (
        Dataset(data=df_to_dict_list(train_df), transform=train_transf),
        Dataset(data=df_to_dict_list(val_df),   transform=val_transf),
        Dataset(data=df_to_dict_list(test_df),  transform=val_transf),
    )


def setup_datasets_diffusion(
    config: Dict,
    stage_1_idxs_file: Optional[str] = None,
) -> Tuple:
    """Stage-2 diffusion 용 Dataset 3개 반환.

    stage_1_idxs_file 이 주어지면 해당 JSON 의 subject id 기준으로 분할.
    없으면 stratified 랜덤 분할.
    """
    files = create_datafiles(config, anatomy=config["dataset"]["anatomy"],
                             modality=config["dataset"]["modality"])
    df = pd.DataFrame(files)
    train_transf, val_transf = setup_transforms(config)

    if stage_1_idxs_file is not None:
        with open(stage_1_idxs_file, "r") as f:
            idxs = json.load(f)
        splits = {
            "train": df[df["subj_id"].isin(idxs["train"])].reset_index(drop=True),
            "val":   df[df["subj_id"].isin(idxs["validation"])].reset_index(drop=True),
            "test":  df[df["subj_id"].isin(idxs["test"])].reset_index(drop=True),
        }
    else:
        train_df, val_df, test_df = _split_df(df, config["default"]["random_seed"])
        splits = {"train": train_df, "val": val_df, "test": test_df}

    return (
        Dataset(data=df_to_dict_list(splits["train"]), transform=train_transf),
        Dataset(data=df_to_dict_list(splits["val"]),   transform=val_transf),
        Dataset(data=df_to_dict_list(splits["test"]),  transform=val_transf),
    )


def setup_datasets_inference(config: Dict, stage_1_idxs_file=None):
    """Inference 전용 Dataset (전체 데이터, val_transf + 원본 형태 저장 적용)."""
    files = create_datafiles(config, anatomy=config["dataset"]["anatomy"],
                             modality=config["dataset"]["modality"])
    df = pd.DataFrame(files)

    modality = config["dataset"].get("modality", ["ct", "cbct"])
    image_keys = [m for m in ("cbct", "ct") if m in modality]
    all_keys = image_keys + ["mask"]

    _, val_transf = setup_transforms(config)
    # inference 시 원본 크기 복원을 위해 shape 저장 삽입
    inference_transf = Compose([
        LoadImaged(keys=all_keys, image_only=True,
                   ensure_channel_first=True, reader="ITKReader"),
        SaveOriginalShapeD(keys=all_keys),
        *val_transf.transforms[1:],   # LoadImaged 중복 제거
    ])
    return Dataset(data=df_to_dict_list(df), transform=inference_transf)
