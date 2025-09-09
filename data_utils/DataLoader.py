# -*- coding: utf-8 -*-
"""
Minimal, clean dataset loaders for tree-species point cloud classification.

- English comments
- Kept interfaces compatible with your train.py:
    * class ModelNetDataLoader (txt-based, with per-class subfolders)
    * class TreeSpeciesNpyDataset (npy-based split folders)
- Removes unused imports/variables and fixes typos (e.g., 'tvalidate' -> 'validate')
"""

import os
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset


def pc_normalize(pc: np.ndarray) -> np.ndarray:
    """
    Normalize a point cloud to zero-mean and unit sphere.
    Args:
        pc: (N, C) array, first 3 columns are xyz.
    Returns:
        normalized (N, C) array
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc[:, :3] ** 2, axis=1)))
    pc = pc / (m + 1e-8)
    return pc


def farthest_point_sample(point: np.ndarray, npoint: int) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) on CPU (NumPy version).

    Args:
        point: (N, D) array, first 3 columns are xyz.
        npoint: number of points to sample.

    Returns:
        sampled: (npoint, D) array
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,), dtype=np.int32)
    distance = np.ones((N,), dtype=np.float32) * 1e10
    farthest = np.random.randint(0, N)

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, axis=-1)

    return point[centroids]


class DataLoader(Dataset):
    """
    TXT-based dataset:
        root/
          ├─ Tree_names.txt      # class names, one per line (defines class IDs)
          ├─ train.txt           # file list (relative to class folder)
          ├─ validate.txt        # file list (relative to class folder)
          └─ <ClassName>/
              └─ <filename>.txt  # point cloud (N, C_total)

    Channel selection is controlled by args.geometric_channel + args.feature_channel.
    """

    def __init__(self, root: str, args, split: str = "train", process_data: bool = False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data  # kept for compatibility (unused here)
        self.uniform = args.use_uniform_sample
        self.num_category = args.num_category
        self.geometric_channel = args.geometric_channel
        self.feature_channel = args.feature_channel
        self.channel_idx = (self.geometric_channel or []) + (self.feature_channel or [])

        # class names and id mapping
        catfile = os.path.join(self.root, "Tree_names.txt")
        self.cat: List[str] = [line.rstrip() for line in open(catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        # split lists
        shape_ids = {
            "train": [line.rstrip() for line in open(os.path.join(self.root, "train.txt"))],
            "validate": [line.rstrip() for line in open(os.path.join(self.root, "validate.txt"))],
        }
        assert split in ("train", "validate"), "split must be 'train' or 'validate'"

        # species name is taken from filename prefix (before first underscore)
        shape_names = ["_".join(x.split("_")[0:1]) for x in shape_ids[split]]

        # full datapath: (class_name, /root/class_name/filename.txt)
        self.datapath = [
            (shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]))
            for i in range(len(shape_ids[split]))
        ]
        print(f"The size of {split} data is {len(self.datapath)}")

        # class weights (per-sample count, not per-point)
        self.labelweights = np.zeros(len(self.classes), dtype=np.float32)
        for shape_name, _ in self.datapath:
            cls_idx = self.classes[shape_name]
            self.labelweights[cls_idx] += 1

        if np.sum(self.labelweights) > 0:
            label_dist = self.labelweights / np.sum(self.labelweights)
            # smooth/flatten long-tail; same exponent as your original code
            self.labelweights = np.power(np.max(label_dist) / label_dist, 1.0 / 3.0)

    def __len__(self) -> int:
        return len(self.datapath)

    def _get_item(self, index: int) -> Tuple[np.ndarray, int]:
        """
        Load a single sample, apply FPS (or resampling) and channel selection.
        Returns:
            points: (npoints, C_sel)
            label:  int
        """
        cls_name, txt_path = self.datapath[index]
        label = int(self.classes[cls_name])

        point_set = np.loadtxt(txt_path, dtype=np.float32)  # (N, C_total)

        # sample/duplicate to npoints
        if point_set.shape[0] >= self.npoints:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            idx = np.random.choice(point_set.shape[0], self.npoints, replace=True)
            point_set = point_set[idx, :]

        # select channels
        if self.channel_idx:
            point_set = point_set[:, self.channel_idx]

        return point_set, label

    def __getitem__(self, index: int):
        return self._get_item(index)


class TreeSpeciesNpyDataset(Dataset):
    """
    NPY-based dataset with split subfolders:
        root/
          ├─ train/
          │   ├─ points/*.npy   # each (N, C_total)
          │   └─ labels.npy     # (num_samples,)
          └─ validate/
              ├─ points/*.npy
              └─ labels.npy
    """

    def __init__(self, root: str, args, split: str = "train"):
        self.root = root
        self.npoints = args.num_point
        self.num_category = args.num_category
        self.split = split

        self.data_dir = os.path.join(root, split)
        self.points_dir = os.path.join(self.data_dir, "points")
        self.labels_path = os.path.join(self.data_dir, "labels.npy")

        assert os.path.exists(self.labels_path), f"Missing labels.npy in {self.data_dir}"
        assert os.path.isdir(self.points_dir), f"Missing 'points' directory in {self.data_dir}"

        self.points_files = sorted(
            os.path.join(self.points_dir, fname)
            for fname in os.listdir(self.points_dir)
            if fname.endswith(".npy")
        )
        self.labels = np.load(self.labels_path).astype(np.int64)
        assert len(self.points_files) == len(self.labels), "Mismatch between samples and labels"

        print(f"The size of {split} data is {len(self.points_files)}")

        # class weights by sample count
        self.labelweights = np.zeros(self.num_category, dtype=np.float32)
        for lbl in self.labels:
            self.labelweights[int(lbl)] += 1

        if np.sum(self.labelweights) > 0:
            dist = self.labelweights / np.sum(self.labelweights)
            self.labelweights = np.power(np.max(dist) / dist, 1.0 / 3.0)

    def __len__(self) -> int:
        return len(self.points_files)

    def __getitem__(self, index: int):
        point_set = np.load(self.points_files[index]).astype(np.float32)  # (N, C_total)
        label = int(self.labels[index])

        # simple truncate/duplicate to npoints (keep first npoints if oversampled upstream)
        if point_set.shape[0] >= self.npoints:
            point_set = point_set[: self.npoints, :]
        else:
            idx = np.random.choice(point_set.shape[0], self.npoints, replace=True)
            point_set = point_set[idx, :]

        return torch.tensor(point_set, dtype=torch.float32), label

    def get_labelweights(self) -> np.ndarray:
        return self.labelweights
