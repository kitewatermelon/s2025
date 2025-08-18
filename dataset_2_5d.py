import pathlib
import torch
from typing import Dict, List, Tuple, Any

import numpy as np
from monai.data import DataLoader, Dataset
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
    ToTensord,
    Resized,
    ScaleIntensityd,
    MapTransform,
    RandAdjustContrastd,
    RandAffined
)

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import pdb
import os
import json
import SimpleITK as sitk

from monai.transforms import MapTransform
from typing import Mapping, Hashable
import numpy as np

class SaveOriginalShapeD(MapTransform):
    """Save the original image shape before transformation"""
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data: Mapping[Hashable, np.ndarray]):
        d = dict(data)
        for key in self.keys:
            if key in d:
                d[f"{key}_original_shape"] = np.array(d[key].shape)
        return d

def load_dataset_indices(indices_path: str) -> Dict[str, List[int]]:
    """
    Load dataset split indices from a JSON file.

    Args:
        indices_path: Path to the JSON file containing dataset indices

    Returns:
        Dictionary containing train, validation and test indices
    """
    with open(indices_path, 'r') as f:
        indices = json.load(f)
    return indices


def df_to_dict_list(df):
    dict_list = []
    for _, row in df.iterrows():
        # Start with required fields
        dict_entry = {
            "subj_id": row["subj_id"],
            "anatomy": row["anatomy"],
            "mask": pathlib.Path(row["mask"])
        }
        # Add modality paths only if they exist in the dataframe
        if "cbct" in row:
            dict_entry["cbct"] = pathlib.Path(row["cbct"])
        if "ct" in row:
            dict_entry["ct"] = pathlib.Path(row["ct"])
        dict_list.append(dict_entry)
    return dict_list


class ApplyMaskd(MapTransform):
    """
    Apply a mask to the input images.

    Args:
        keys: Keys to be used for the input images
        mask_key: Key to be used for the mask

    Returns:
        Dict
    """

    def __init__(
        self,
        keys: KeysCollection,
        mask_key: str = "mask",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key

    def __call__(self, data):
        d = dict(data)
        mask = d[self.mask_key]  # assumed to be 0/1 float or byte
        inv_mask = 1.0 - mask
        for key in self.keys:
            if key in d:
                img = d[key]
                min_img = img.min()
                # background → fill_value, foreground → original
                d[key] = img * mask + min_img * inv_mask
        return d


class ClipHUValues(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
    ):
        super().__init__(keys)
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                d[key] = torch.clamp(d[key], -800, 1500)
        return d


class Extract2DSlicesd(MapTransform):
    """
    Extract 2D slices from a 3D image and store them as a list of 2D tensors.
    Assumes input shape (C, H, W, D).
    Output shape will be (C, H, W) for each slice in a list.
    """
    def __init__(self, keys: KeysCollection, dim: int = -1, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.dim = dim # Dimension to slice along (D)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d and d[key] is not None:
                # Assuming input is (C, H, W, D)
                img_3d = d[key]
                slices = [img_3d.select(self.dim, i).squeeze(self.dim) for i in range(img_3d.shape[self.dim])]
                d[key] = slices # Store a list of 2D slices
        return d
class Compose2DTransformsd(MapTransform):
    """
    Applies a sequence of 2D transforms to a list of 2D slices.
    """
    def __init__(self, keys: KeysCollection, transforms: Compose, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.transforms = transforms

    def __call__(self, data):
        d = dict(data)
        # Assuming all keys in 'd' that are lists of slices have the same number of slices (D dimension)
        if not d or not all(isinstance(d[k], list) for k in self.keys if k in d):
            raise ValueError("Input data for Compose2DTransformsd must contain lists of slices for specified keys.")

        # Determine the number of slices (D) from any of the keys
        num_slices = 0
        for k in self.keys:
            if k in d and isinstance(d[k], list):
                num_slices = len(d[k])
                break
        
        if num_slices == 0: # Handle case where no lists are found or lists are empty
            return d

        processed_data_slices = {k: [] for k in self.keys}

        for i in range(num_slices):
            # Create a dictionary for the current 2D slice across all relevant keys
            slice_data = {}
            for key in self.keys:
                if key in d and i < len(d[key]):
                    slice_data[key] = d[key][i]

            # Apply 2D transforms to the slice_data (which now includes both image and mask for this slice)
            processed_slice_data = self.transforms(slice_data)

            # Store the processed slice back into the processed_data_slices structure
            for key in self.keys:
                if key in processed_slice_data:
                    processed_data_slices[key].append(processed_slice_data[key])
        
        # Update the original dictionary with the processed lists of slices
        for key in self.keys:
            if key in d: # Only update if the key existed in the original data
                d[key] = processed_data_slices[key]
        return d

class Reconstruct3Dd(MapTransform):
    """
    Reconstructs a 3D image from a list of 2D slices.
    Assumes input is a list of (C, H, W) tensors.
    Output will be (C, D, H, W).
    """
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d and d[key] is not None and isinstance(d[key], list):
                # Stack to (D, C, H, W)
                stacked = torch.stack(d[key], dim=0)

                # Permute to (C, D, H, W)
                d[key] = stacked.permute(1, 0, 2, 3)
        return d
from monai.transforms import MapTransform, Resize
import torch
import numpy as np
import warnings

class ResizeBackToOriginalShapeD(MapTransform):
    """
    Resize tensors back to their original shape saved under <key>_original_shape using MONAI's Resize.
    """
    def __init__(self, keys, mode: str = "nearest"):
        super().__init__(keys)
        self.mode = mode

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            original_shape_key = f"{key}_original_shape"
            if original_shape_key not in d:
                warnings.warn(f"[WARN] Original shape not found for key: {key}, skipping resize.")
                continue

            target_shape = d[original_shape_key]
            img = d[key]

            is_tensor = isinstance(img, torch.Tensor)
            original_device = img.device if is_tensor else None

            # `Resize`는 채널 차원을 포함한 텐서를 입력으로 기대합니다.
            # 만약 `img`가 (H, W, D) 형태라면 (C, H, W, D)로 만들어야 합니다.
            if img.ndim == 3:
                img = np.expand_dims(img, axis=0) # (H, W, D) -> (1, H, W, D)

            # `target_shape`에서 채널 차원을 포함한 크기 (C, H, W, D)를 가져옵니다.
            # `target_shape`은 보통 [1, 395, 443, 68]와 같은 형태입니다.
            spatial_size = tuple(target_shape[1:])
            
            # MONAI Resize는 이미지 차원 (C, H, W, D)와 리사이즈 타겟 (H', W', D')를 인자로 받습니다.
            resizer = Resize(spatial_size=spatial_size, mode=self.mode)
            
            # NumPy 배열을 텐서로 변환하여 Resize에 전달
            if not is_tensor:
                img = torch.from_numpy(img).to(original_device)

            resized_tensor = resizer(img)

            # 리사이즈된 텐서에서 채널 차원 제거 후 NumPy로 변환
            # (1, 395, 443, 68) -> (395, 443, 68)
            resized_numpy = resized_tensor.squeeze(0).cpu().numpy()

            if is_tensor:
                d[key] = torch.from_numpy(resized_numpy).to(original_device)
            else:
                d[key] = resized_numpy

        return d

def setup_transforms(config: Dict):
    interpolate = config["dataset"]["interpolate"]
    modality = config["dataset"].get("modality", ["ct", "cbct"])

    # Determine which keys to use based on modality
    image_keys = []
    if "cbct" in modality:
        image_keys.append("cbct")
    if "ct" in modality:
        image_keys.append("ct")

    # Always include mask
    all_keys = image_keys + ["mask"]
    
    if interpolate:
        # For 2D transforms, spatial_size should be (H, W) or (H, W, 1) if Resized handles 3D input with D=1
        spatial_size_conf = tuple(config["dataset"]["interpolation_size"][:2]) # Only H and W for 2D

    # Common transforms applied to individual 2D slices
    common_2d_transforms = [
        ApplyMaskd(keys=image_keys, mask_key="mask"), # Masking operates on 2D
        ClipHUValues(keys=image_keys), # Clipping operates on 2D
        ScaleIntensityd(
            keys=image_keys,
            minv=config["dataset"]["minv"],
            maxv=config["dataset"]["maxv"]
        ),
    ] + ([Resized(keys=image_keys + ["mask"], spatial_size=spatial_size_conf)] if interpolate else []) + \
    [ # EnsureTyped and ToTensord should be applied to individual slices
        EnsureTyped(keys=image_keys + ["mask"], dtype=torch.float),
        ToTensord(keys=image_keys + ["mask"])
    ]

    # val_transf = Compose(
    #     [
    #         LoadImaged(
    #             keys=all_keys,
    #             image_only=True,
    #             ensure_channel_first=True, # Will result in (C, H, W, D)
    #             reader="ITKReader"
    #         ),
    #         Extract2DSlicesd(keys=all_keys, dim=-1), # Extract slices from (C, H, W, D) to list of (C, H, W)
    #         Compose2DTransformsd(keys=all_keys, transforms=Compose(common_2d_transforms)), # Apply 2D transforms
    #         Reconstruct3Dd(keys=all_keys) # Reconstruct back to (C, H, W, D)
    #     ]
    # )
    val_transf = Compose(
        [
            LoadImaged(
                keys=all_keys,
                image_only=True,
                ensure_channel_first=True,
                reader="ITKReader"
            ),
            SaveOriginalShapeD(keys=all_keys),  # 여기에 삽입!
            Extract2DSlicesd(keys=all_keys, dim=-1),
            Compose2DTransformsd(keys=all_keys, transforms=Compose(common_2d_transforms)),
            Reconstruct3Dd(keys=all_keys)
        ]
    )

    if config["dataset"]["augment"]:
        rand_adj_prob = config["dataset"]["rand_adj_contrast"]["prob"] if "rand_adj_contrast" in config["dataset"] else 0.0
        rand_adj_gamma = config["dataset"]["rand_adj_contrast"]["gamma"] if "rand_adj_contrast" in config["dataset"] else (0.5, 1.5)
        rand_affine_prob = config["dataset"]["rand_affine"]["prob"] if "rand_affine" in config["dataset"] else 0.0

        augment_2d_transforms = []
        if rand_adj_prob > 0.0:
            augment_2d_transforms.append(
                RandAdjustContrastd(
                    keys=image_keys,
                    prob=rand_adj_prob,
                    gamma=tuple(rand_adj_gamma)
                )
            )

        if rand_affine_prob > 0.0:
            augment_2d_transforms.append(
                RandAffined(
                    keys=all_keys, # Apply affine to both image and mask
                    rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
                    translate_range=[(-1, 1), (-1, 1)],
                    scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
                    padding_mode="zeros",
                    prob=rand_affine_prob
                )
            )

        # Combined 2D transforms for training, including augmentation
        train_2d_transforms = Compose(common_2d_transforms + augment_2d_transforms)

        train_transf = Compose(
            [
                LoadImaged(
                    keys=all_keys,
                    image_only=True,
                    ensure_channel_first=True,
                    reader="ITKReader"
                ),
                Extract2DSlicesd(keys=all_keys, dim=-1),
                Compose2DTransformsd(keys=all_keys, transforms=train_2d_transforms),
                Reconstruct3Dd(keys=all_keys)
            ]
        )
    else:
        train_transf = val_transf

    return train_transf, val_transf


def create_datafiles(config: Dict, anatomy: List[str]=["AB", "HN", "TH"], modality: List[str]=["cbct", "ct"]) -> Tuple[List[Dict]]:
    """
    Create a list of dictionaries containing paths to image files based on specified modality and anatomy regions

    Args:
        config: Configuration dictionary containing dataset parameters
        anatomy: List of anatomical anatomies to include (e.g., ["ab", "hn", "th"] (abdomen, head-neck, thorax))
        modality: List of modalities to include - can contain "cbct", "ct", or both

    Returns:
        List of dictionaries containing file paths and metadata
    """
    files = []
    data_path = pathlib.Path(config["dataset"]["data_path"])

    for mod in modality:
        if mod not in ["cbct", "ct"]:
            raise ValueError("modality must be one of: 'cbct', or 'ct'")

    anatomy = [anat.upper() for anat in anatomy]

    for anat in anatomy:
        if anat not in ["AB", "HN", "TH"]:
            raise ValueError(f"anatomy '{anat}' must be one of: 'AB', 'HN', 'TH'")

    for anat in anatomy:
        anat_path = data_path / anat
        if not anat_path.exists():
            print(f"Warning: {anat} directory not found in {data_path}")
            continue

        for subj_dir in anat_path.glob('*'):
            if not subj_dir.is_dir():
                continue

            required_files = {"mask": subj_dir / "mask.mha"}

            if "cbct" in modality and "ct" in modality:
                required_files.update({
                    "cbct": subj_dir / "cbct.mha",
                    "ct": subj_dir / "ct.mha"
                })
            elif "cbct" in modality:
                required_files["cbct"] = subj_dir / "cbct.mha"
            elif "ct" in modality:
                required_files["ct"] = subj_dir / "ct.mha"

            if all(p.exists() for p in required_files.values()):
                subj_dict = {
                    "subj_id": str(subj_dir.name),
                    "anatomy": anat,
                }
                subj_dict.update({k: str(v) for k,v in required_files.items()})
                files.append(subj_dict)
    return files


def setup_datasets_diffusion(config: Dict, stage_1_idxs_file) -> DataLoader:
    """
    Create a dataloader for inference using test indices.

    Args:
        config: Configuration dictionary containing dataset parameters
        stage_1_idxs_file: json containing

    Returns:
        DataLoader for test dataset
    """
    files = create_datafiles(
        config,
        anatomy=config["dataset"]["anatomy"],
        modality=config["dataset"]["modality"]
    )
    df = pd.DataFrame(files)

    train_df, temp_df = train_test_split(df, test_size=.3, stratify=df["anatomy"], random_state=config["default"]["random_seed"])

    val_df, hold_out_df = train_test_split(temp_df, test_size=.60, stratify=temp_df["anatomy"], random_state=config["default"]["random_seed"])

    train_transf, val_transf = setup_transforms(config)
    log_df = pd.concat([train_df, val_df, hold_out_df])

    train_ds = Dataset(
        data=df_to_dict_list(train_df),
        transform=train_transf
    )
    val_ds = Dataset(
        data=df_to_dict_list(val_df),
        transform=val_transf
    )

    test_ds = Dataset(
        data=df_to_dict_list(hold_out_df),
        transform=val_transf
    )

    return train_ds, val_ds, test_ds

def setup_dataloaders(config: Dict, save_train_idxs=False):

    files = create_datafiles(
        config,
        anatomy=config["dataset"]["anatomy"],
        modality=config["dataset"]["modality"]
    )
    df = pd.DataFrame(files)

    train_df, temp_df = train_test_split(df, test_size=.3, stratify=df["anatomy"], random_state=config["default"]["random_seed"])

    val_df, hold_out_df = train_test_split(temp_df, test_size=.60, stratify=temp_df["anatomy"], random_state=config["default"]["random_seed"])

    train_transf, val_transf = setup_transforms(config)
    log_df = pd.concat([train_df, val_df, hold_out_df])

    train_ds = Dataset(
        data=df_to_dict_list(train_df),
        transform=train_transf
    )
    val_ds = Dataset(
        data=df_to_dict_list(val_df),
        transform=val_transf
    )

    test_ds = Dataset(
        data=df_to_dict_list(hold_out_df),
        transform=val_transf
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config["dataset"]["train_batch_size"],
        shuffle=config["dataset"]["train_shuffle"],
        num_workers=config["dataset"]["num_workers"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["dataset"]["val_batch_size"],
        shuffle=config["dataset"]["val_shuffle"],
        num_workers=config["dataset"]["num_workers"],
        drop_last=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config["dataset"]["val_batch_size"],
        shuffle=config["dataset"]["val_shuffle"],
        num_workers=config["dataset"]["num_workers"],
    )

    if save_train_idxs:
        indices = {
            'train': train_df["subj_id"].tolist(),
            'validation': val_df["subj_id"].tolist(),
            'test': hold_out_df["subj_id"].tolist()
        }
        exp_dir = os.path.join(config["optim"]["checkpoint_dir"], config['default']['experiment_name'])
        os.makedirs(exp_dir, exist_ok=True) # Ensure directory exists
        indices_path = os.path.join(exp_dir, 'dataset_indices.json')
        with open(indices_path, 'w') as f:
            json.dump(indices, f)
    return train_loader, val_loader


def setup_datasets(config: Dict):
    files = create_datafiles(
        config,
        anatomy=config["dataset"]["anatomy"],
        modality=config["dataset"]["modality"]
    )
    df = pd.DataFrame(files)

    train_df, temp_df = train_test_split(df, test_size=.3, stratify=df["anatomy"], random_state=config["default"]["random_seed"])

    val_df, hold_out_df = train_test_split(temp_df, test_size=.60, stratify=temp_df["anatomy"], random_state=config["default"]["random_seed"])

    train_transf, val_transf = setup_transforms(config)
    log_df = pd.concat([train_df, val_df, hold_out_df])

    train_ds = Dataset(
        data=df_to_dict_list(train_df),
        transform=train_transf
    )
    val_ds = Dataset(
        data=df_to_dict_list(val_df),
        transform=val_transf
    )

    test_ds = Dataset(
        data=df_to_dict_list(hold_out_df),
        transform=val_transf
    )

    return train_ds, val_ds

def setup_datasets_inference(config: Dict, stage_1_idxs_file) -> DataLoader:
    """
    Create a dataloader for inference using test indices.

    Args:
        config: Configuration dictionary containing dataset parameters
        stage_1_idxs_file: json containing

    Returns:
        DataLoader for test dataset
    """
    files = create_datafiles(
        config,
        anatomy=config["dataset"]["anatomy"],
        modality=config["dataset"]["modality"]
    )
    df = pd.DataFrame(files)


    train_transf, val_transf = setup_transforms(config)

    train_ds = Dataset(
        data=df_to_dict_list(df),
        transform=val_transf
    )
    return train_ds
