#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference + Evaluation with voting on txt-listed dataset.


"""

import os
import sys
import csv
import argparse
import importlib
from pathlib import Path
from collections import Counter
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ErrorMatrix import ConfusionMatrix

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "models"))


# -----------------------------
# Helpers
# -----------------------------
def resolve_point_all(npoints: int) -> int:
    """Map num_point to the minimal loaded points (point_all) pool."""
    if npoints == 1024:
        return 1200
    elif npoints == 4096:
        return 4800
    elif npoints == 8192:
        return 8192
    else:
        # Default: a bit larger than target to enable diverse FPS picks
        return max(npoints, int(npoints * 1.1))


def farthest_point_sample_torch(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling (torch, CPU/GPU)
    Args:
        xyz: (B, N, 3)
        npoint: int
    Returns:
        idx: (B, npoint) long
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def gather_operation(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather points by indices.
    Args:
        x:   (B, C, N)
        idx: (B, S)
    Returns:
        out: (B, C, S)
    """
    B, C, N = x.shape
    S = idx.shape[1]
    idx_expanded = idx.unsqueeze(1).expand(-1, C, -1)  # (B, C, S)
    return torch.gather(x, 2, idx_expanded)


# -----------------------------
# Dataset: txt-listed + labels from class folder
# -----------------------------
class Dataset(Dataset):
    """
    Expected layout:
        data_path/
          ├─ Tree_names.txt
          ├─ train.txt | validate.txt | test.txt
          └─ <ClassName>/
               └─ <filename>.txt or .npy  (N, C_total)

    We read <split>.txt and return (points[N, C_total], label_id, tree_id).
    `label_id` is derived from <ClassName> using Tree_names.txt order.
    `tree_id` is filename stem (int if possible, else string).
    """

    def __init__(self, data_path: str, split: str, channel_idx: List[int]):
        self.root = data_path
        assert split in ("test"), "split must be test"
        self.split = split
        self.channel_idx = channel_idx or []

        # class names and mapping
        catfile = os.path.join(self.root, "Tree_names.txt")
        self.class_names: List[str] = [line.rstrip() for line in open(catfile)]
        self.classes = {name: i for i, name in enumerate(self.class_names)}

        # Read split list
        split_txt = os.path.join(self.root, f"{split}.txt")
        assert os.path.exists(split_txt), f"Missing {split}.txt at {split_txt}"
        shape_ids = [line.rstrip() for line in open(split_txt)]
        # Derive class folder name from filename prefix (before first '_')
        shape_names = ["_".join(x.split("_")[0:1]) for x in shape_ids]

        # Build full paths
        self.datapath: List[Tuple[str, str]] = [
            (shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[i])) for i in range(len(shape_ids))
        ]
        print(f"The size of {split} data is {len(self.datapath)}")

    def __len__(self) -> int:
        return len(self.datapath)

    def __getitem__(self, index: int):
        cls_name, fpath = self.datapath[index]
        label = int(self.classes[cls_name])

        # Load points
        if fpath.endswith(".npy"):
            pts = np.load(fpath).astype(np.float32)  # (N, C_total)
        else:
            pts = np.loadtxt(fpath, dtype=np.float32)

        # Select channels if provided
        if self.channel_idx:
            pts = pts[:, self.channel_idx]

        # tree id from filename stem
        stem = Path(fpath).stem
        try:
            tree_id = int(stem)
        except:
            tree_id = stem

        return pts, label, tree_id, fpath


# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Inference + Evaluation (voting)")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--use_cpu", action="store_true")

    # Data layout
    p.add_argument("--data_path", type=str, required=True, help="Dataset root (contains Tree_names.txt and split txt)")
    p.add_argument("--num_category", type=int, required=True)
    p.add_argument("--num_point", type=int, default=1024)

    # Channels
    p.add_argument("--geometric_channel", type=int, nargs="+", required=True)
    p.add_argument("--feature_channel", type=int, nargs="+", required=True)

    # Model
    p.add_argument("--model", type=str, required=True, help="e.g., GTN.3dgtn_cls")
    p.add_argument("--ckpt", type=str, required=True, help="checkpoint path (.pth)")
    p.add_argument("--sampling_rate", type=int, default=4)

    # Voting
    p.add_argument("--vote_times", type=int, default=10, help="number of FPS votes per sample")

    # Loader
    p.add_argument("--batch_size", type=int, default=1, help="keep 1 since samples can have variable lengths")
    p.add_argument("--num_workers", type=int, default=0)

    # Output
    p.add_argument("--output_csv", type=str, default="predict/pred_eval.csv")
    return p.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")

    # Build model
    chan_idx = (args.geometric_channel or []) + (args.feature_channel or [])
    model_module = importlib.import_module(args.model)
    classifier = model_module.get_model(
        args.num_category, args.geometric_channel, args.feature_channel, args.sampling_rate, args.num_point
    ).to(device).eval()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    classifier.load_state_dict(state_dict, strict=False)

    # Dataset & loader
    ds = Dataset(args.data_path, "test", chan_idx)
    class_names = ds.class_names
    assert len(class_names) == args.num_category, "num_category mismatch with Tree_names.txt"

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Evaluation setup
    cm = ConfusionMatrix(num_classes=args.num_category, labels=class_names)
    point_all = resolve_point_all(args.num_point)

    # CSV
    is_new = not os.path.exists(args.output_csv)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    outf = open(args.output_csv, "a", newline="")
    writer = csv.writer(outf)
    if is_new:
        writer.writerow(["treeID", "gt_species", "pred_species"])
        outf.flush()
        os.fsync(outf.fileno())

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), desc=f"Eval Test"):
            # batch is a tuple of tensors/lists from __getitem__ outputs (wrapped by default collate)
            # Since batch_size=1, unwrap them:
            pts_np, gt_label, tree_id, fpath = batch
            pts_np = pts_np[0].numpy() if isinstance(pts_np, torch.Tensor) else pts_np[0]  # (N, C_sel)
            gt_label = int(gt_label[0].item() if hasattr(gt_label[0], "item") else gt_label[0])

            # Ensure pool size = point_all
            N, Csel = pts_np.shape
            if N < point_all:
                # pad by random repeat
                extra_idx = np.random.choice(N, point_all - N, replace=True)
                pts_pool = np.concatenate([pts_np, pts_np[extra_idx, :]], axis=0)  # (point_all, Csel)
            elif N > point_all:
                keep = np.random.choice(N, point_all, replace=False)
                pts_pool = pts_np[keep, :]
            else:
                pts_pool = pts_np

            # Prepare torch tensors once per sample
            pts_pool_t = torch.from_numpy(pts_pool).float().to(device)          # (P, Csel)
            xyz_pool_t = pts_pool_t[:, :3].unsqueeze(0).contiguous()            # (1, P, 3)
            feat_pool_t = pts_pool_t.unsqueeze(0).transpose(2, 1).contiguous()  # (1, Csel, P)

            # Voting
            votes = []
            for _ in range(args.vote_times):
                if point_all < args.num_point:
                    # shouldn't happen with our mapping, but keep safe
                    idx = torch.arange(args.num_point, device=device) % point_all
                    idx = idx.unsqueeze(0)
                else:
                    idx = farthest_point_sample_torch(xyz_pool_t, args.num_point)  # (1, num_point)

                gathered = gather_operation(feat_pool_t, idx)   # (1, Csel, num_point)
                pts_input = gathered  # (1, Csel, num_point), model expects [B, C, N]

                logits, _ = classifier(pts_input)
                pred = int(logits.argmax(dim=-1).item())
                votes.append(pred)

            final_idx = Counter(votes).most_common(1)[0][0]
            cm.update(np.array([final_idx]), np.array([gt_label]))

            writer.writerow([tree_id[0], class_names[gt_label], class_names[final_idx]])
            outf.flush()
            os.fsync(outf.fileno())

    outf.close()

    # Summary
    OA, mrec, mF1, _, table = cm.summary()
    print("\n===== Evaluation Summary =====")
    print(f"Overall Accuracy (OA): {OA:.6f}")
    print(f"Mean Class Accuracy (mAcc): {mrec:.6f}")
    print(f"Macro F1: {mF1:.6f}")
    print(table)


if __name__ == "__main__":
    main()
