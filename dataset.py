import os
import glob
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Dict, List, Tuple, Any, Mapping, Hashable
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
import json, warnings
import random  # random 모듈 추가

from monai.data import DataLoader, Dataset as MonaiDataset
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
    EnsureChannelFirstd,
    SpatialPadd,
    ScaleIntensityRanged,
    RandAdjustContrastd,
    RandAffined,
    RandSpatialCropd,
    RandFlipd,
    Resize,
    MapTransform,
    Orientationd,
    Spacingd,
)


# -------------------------------
# MONAI 호환 커스텀 Transforms (수정 없음)
# -------------------------------
class ApplyMaskd(MapTransform):
    """Apply a mask to the input images."""
    def __init__(self, keys: KeysCollection, mask_key: str = "mask", allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key

    def __call__(self, data):
        d = dict(data)
        mask = d[self.mask_key]
        inv_mask = 1.0 - mask
        for key in self.keys:
            if key in d:
                img = d[key]
                min_img = img.min()
                d[key] = img * mask + min_img * inv_mask
        return d


# -------------------------------
# 사용자 정의 데이터셋
# -------------------------------
class SlidingWindowPatchDataset(Dataset):
    def __init__(self, root_dir: str, modality: str, subject_dirs: List[str] = None,
                 patch_size: Tuple[int, int, int] = (128, 128, 0), overlap: float = 0.5, transform=None):
        self.root_dir = root_dir
        self.modality = modality.lower()
        self.patch_size = patch_size
        self.overlap = overlap
        self.transform = transform
        self.subject_dirs = subject_dirs

        if self.modality not in ['ct', 'cbct']:
            raise ValueError("modality는 'ct' 또는 'cbct'여야 합니다.")
        
        self.file_list = []
        self.data_cache = {}  # <--- 이미지 메모리에 캐싱
        self._pre_calculate_patches()

    def _pre_calculate_patches(self):
        if self.subject_dirs is None:
            tasks = ['AB', 'HN', 'TH']
            for task in tasks:
                task_dir = os.path.join(self.root_dir, task)
                self.subject_dirs = [os.path.join(task_dir, d) for d in os.listdir(task_dir)
                                     if os.path.isdir(os.path.join(task_dir, d))]
        
        for case_dir in self.subject_dirs:
            input_path = os.path.join(case_dir, f'{self.modality}.mha')
            mask_path = os.path.join(case_dir, 'mask.mha')
            
            if not (os.path.exists(input_path) and os.path.exists(mask_path)):
                print(f"경고: {case_dir}에서 파일이 누락되었습니다. 이 케이스는 건너뜁니다.")
                continue
            
            # --- 캐싱: 파일을 numpy array로 미리 변환해 저장 ---
            if input_path not in self.data_cache:
                input_img = sitk.ReadImage(input_path)
                mask_img = sitk.ReadImage(mask_path)
                
                input_array = sitk.GetArrayFromImage(input_img)
                mask_array = sitk.GetArrayFromImage(mask_img)

                # 전체 볼륨에 대한 정규화
                a_min = -1024.
                a_max = 3071.
                b_min = 0.0
                b_max = 1.0
                input_array = np.clip(input_array, a_min, a_max)
                input_array = ((input_array - a_min) / (a_max - a_min)) * (b_max - b_min) + b_min
                
                self.data_cache[input_path] = input_array
                self.data_cache[mask_path] = mask_array
            
            input_array = self.data_cache[input_path]
            _, original_height, original_width = input_array.shape  # (D,H,W)

            target_width, target_height = self.patch_size[0], self.patch_size[1]
            stride_x = int(target_width * (1.0 - self.overlap))
            stride_y = int(target_height * (1.0 - self.overlap))
            if stride_x == 0: stride_x = 1
            if stride_y == 0: stride_y = 1
            
            for y in range(0, original_height, stride_y):
                for x in range(0, original_width, stride_x):
                    start_x = x
                    start_y = y
                    if start_x + target_width > original_width:
                        start_x = original_width - target_width
                    if start_y + target_height > original_height:
                        start_y = original_height - target_height
                    
                    self.file_list.append({
                        self.modality: input_path,
                        'mask': mask_path,
                        'origin': (0, start_y, start_x),  # (z,y,x) 순서
                        'size': (input_array.shape[0], target_height, target_width)
                    })

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        patch_info = self.file_list[idx]
        input_array = self.data_cache[patch_info[self.modality]]
        mask_array = self.data_cache[patch_info['mask']]

        z, y, x = patch_info['origin']
        dz, dy, dx = patch_info['size']

        # numpy 슬라이싱으로 패치 추출
        input_patch = input_array[z:z+dz, y:y+dy, x:x+dx]
        mask_patch  = mask_array[z:z+dz, y:y+dy, x:x+dx]

        data = {
            self.modality: input_patch,
            'mask': mask_patch
        }
        if self.transform:
            data = self.transform(data)
        return data

# -------------------------------
# Transforms setup (3D pipeline)
# -------------------------------
def setup_transforms(config: Dict):
    modality = config["dataset"].get("modality", ["ct", "cbct"])
    image_keys = [m for m in modality if m in ["ct", "cbct"]]
    all_keys = image_keys + ["mask"]
    
    # 공통 트랜스폼: 채널 추가
    common_transforms = [
        ApplyMaskd(keys=image_keys, mask_key="mask"),
        EnsureChannelFirstd(keys=all_keys),
        EnsureTyped(keys=all_keys, dtype=torch.float),
    ]
    
    val_transf = Compose(common_transforms)

    # 학습용 트랜스폼: 증강
    train_transf = Compose(
        common_transforms + [
            RandFlipd(keys=all_keys, spatial_axis=0, prob=0.5),
            RandAdjustContrastd(keys=image_keys, prob=0.5, gamma=(0.5, 1.5)),
            RandAffined(keys=all_keys, rotate_range=[(-15, 15)] * 3, prob=0.5, padding_mode="border"),
        ]
    )
    
    return train_transf, val_transf


# -------------------------------
# Dataset + Loader setups
# -------------------------------
def setup_dataloaders(config: Dict, save_train_idxs=False):
    """
    SlidingWindowPatchDataset을 사용하여 데이터셋과 데이터로더를 설정합니다.
    """
    data_path = pathlib.Path(config["dataset"]["data_path"])
    anatomy_list = [anat.upper() for anat in config["dataset"]["anatomy"]]
    
    all_subjects = []
    for anat in anatomy_list:
        anat_path = data_path / anat
        if not anat_path.exists():
            continue
        all_subjects.extend([str(p) for p in anat_path.glob('*') if p.is_dir()])
        
    df = pd.DataFrame({"subject_path": all_subjects})
    
    train_df, temp_df = train_test_split(df, test_size=.3, random_state=config["default"]["random_seed"])
    val_df, test_df = train_test_split(temp_df, test_size=.60, random_state=config["default"]["random_seed"])
    
    train_subjects = train_df["subject_path"].tolist()
    val_subjects = val_df["subject_path"].tolist()
    test_subjects = test_df["subject_path"].tolist()

    train_transf, val_transf = setup_transforms(config)

    # 전체 훈련 데이터셋을 먼저 생성
    full_train_ds = SlidingWindowPatchDataset(
        root_dir=config["dataset"]["data_path"],
        modality=config["dataset"]["modality"][0],
        subject_dirs=train_subjects,
        patch_size=tuple(config["dataset"]["patch_size"]),
        overlap=config["dataset"]["overlap"],
        transform=train_transf     
        )
    
    # 훈련 데이터셋에서 1%만 무작위로 선택
    num_samples = len(full_train_ds) // 100
    if num_samples < 1:
        warnings.warn("데이터셋 크기가 너무 작아 1% 샘플링이 불가능합니다. 전체 데이터셋을 사용합니다.")
        num_samples = len(full_train_ds)
        
    subset_indices = random.sample(range(len(full_train_ds)), num_samples)
    train_ds = Subset(full_train_ds, subset_indices)
    print(f"훈련 데이터셋: {len(full_train_ds)}개 패치 중 {len(train_ds)}개 ({num_samples}) 패치 사용")

    # 검증/테스트 데이터셋 생성 (전체 사용)
    val_ds = SlidingWindowPatchDataset(
        root_dir=config["dataset"]["data_path"],
        modality=config["dataset"]["modality"][0],
        subject_dirs=val_subjects,
        patch_size=tuple(config["dataset"]["patch_size"]),
        overlap=config["dataset"]["overlap"],
        transform=val_transf     
        )
    test_ds = SlidingWindowPatchDataset(
        root_dir=config["dataset"]["data_path"],
        modality=config["dataset"]["modality"][0],
        subject_dirs=test_subjects,
        patch_size=tuple(config["dataset"]["patch_size"]),
        overlap=config["dataset"]["overlap"],
        transform=val_transf,
        )

    train_loader = DataLoader(
        train_ds, batch_size=config["dataset"]["train_batch_size"],
        shuffle=config["dataset"]["train_shuffle"],
        num_workers=config["dataset"]["num_workers"],
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=config["dataset"]["val_batch_size"],
        shuffle=config["dataset"]["val_shuffle"],
        num_workers=config["dataset"]["num_workers"],
        drop_last=True
    )

    test_loader = DataLoader(
        test_ds, batch_size=config["dataset"]["val_batch_size"],
        shuffle=config["dataset"]["val_shuffle"],
        num_workers=config["dataset"]["num_workers"]
    )

    if save_train_idxs:
        indices = {
            'train': train_subjects,
            'validation': val_subjects,
            'test': test_subjects
        }
        exp_dir = os.path.join(config["optim"]["checkpoint_dir"], config['default']['experiment_name'])
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, 'dataset_indices.json'), 'w') as f:
            json.dump(indices, f)

    return train_loader, val_loader, test_loader