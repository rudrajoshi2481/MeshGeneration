"""
dataset.py — ModelNet40 dataset with caching and contrastive augmentation.
"""

import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List

from preprocessing import (
    load_or_compute, augment_points, MODELNET40_CLASSES, CLASS_TO_IDX
)
from config import TrainConfig


class ModelNet40Dataset(Dataset):
    """
    Loads pre-cached processed ModelNet40 PLY files.
    Returns geometric features + query points + occupancy labels.
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: str,
        split: str = "train",             # "train" or "val"
        val_fraction: float = 0.15,
        num_surface: int = 2048,
        num_query: int = 2048,
        use_augmentation: bool = True,
        use_contrastive: bool = True,
        seed: int = 42,
    ):
        self.cache_dir = cache_dir
        self.num_surface = num_surface
        self.num_query = num_query
        self.use_augmentation = use_augmentation
        self.use_contrastive = use_contrastive

        # Gather all PLY files recursively (files live in class/split subdirs)
        all_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.ply"), recursive=True))
        if not all_files:
            raise FileNotFoundError(f"No .ply files found in {data_dir}")

        # Split by class (stratified)
        rng = np.random.RandomState(seed)
        train_files, val_files = [], []
        for cls in MODELNET40_CLASSES:
            cls_files = [f for f in all_files
                         if os.path.basename(f).startswith(cls + "_")]
            cls_files = sorted(cls_files)
            n_val = max(1, int(len(cls_files) * val_fraction))
            val_idx = set(rng.choice(len(cls_files), n_val, replace=False).tolist())
            for i, f in enumerate(cls_files):
                if i in val_idx:
                    val_files.append(f)
                else:
                    train_files.append(f)

        self.files = train_files if split == "train" else val_files
        print(f"[Dataset] {split}: {len(self.files)} samples from {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        data = load_or_compute(
            path, self.cache_dir,
            num_surface=self.num_surface,
            num_query=self.num_query,
        )
        if data is None:
            # Fall back to a random valid sample
            return self.__getitem__((idx + 1) % len(self.files))

        points = data["points"].numpy()     # [N, 3]
        normals = data["normals"].numpy()   # [N, 3]
        curvature = data["curvature"].numpy()  # [N, 1]
        query_pts = data["query_pts"]       # [M, 3] tensor
        occupancy = data["occupancy"]       # [M] tensor
        label = data["label"]

        if self.use_augmentation:
            points, normals, curvature = augment_points(points, normals, curvature)

        out = {
            "points": torch.from_numpy(points),       # [N, 3]
            "normals": torch.from_numpy(normals),     # [N, 3]
            "curvature": torch.from_numpy(curvature), # [N, 1]
            "query_pts": query_pts,                   # [M, 3]
            "occupancy": occupancy,                   # [M]
            "label": torch.tensor(label, dtype=torch.long),
        }

        # Contrastive augmentation: second view
        if self.use_contrastive:
            pts_aug, nrm_aug, cur_aug = augment_points(
                data["points"].numpy(),
                data["normals"].numpy(),
                data["curvature"].numpy(),
            )
            out["points_aug"] = torch.from_numpy(pts_aug)

        return out


def build_dataloaders(cfg: TrainConfig, num_gpus: int = 1):
    train_ds = ModelNet40Dataset(
        data_dir=cfg.data_dir,
        cache_dir=cfg.cache_dir,
        split="train",
        num_surface=cfg.num_surface_points,
        num_query=cfg.num_query_points,
        use_augmentation=True,
        use_contrastive=True,
    )
    val_ds = ModelNet40Dataset(
        data_dir=cfg.data_dir,
        cache_dir=cfg.cache_dir,
        split="val",
        num_surface=cfg.num_surface_points,
        num_query=cfg.num_query_points,
        use_augmentation=False,
        use_contrastive=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )
    return train_loader, val_loader
