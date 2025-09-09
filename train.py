#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import gc
import shutil
import argparse
import datetime
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import provider
from ErrorMatrix import ConfusionMatrix
from data_utils.DataLoader import (
    DataLoader,
    TreeSpeciesNpyDataset,
    
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))


# ----------------------------- #
# Args
# ----------------------------- #
def build_args():
    parser = argparse.ArgumentParser("Training")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA device id(s), e.g. '0' or '0,1'")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--model", type=str, default="models.3DGTN.3dgtn_cls", help="Model module path, e.g. models.3DGTN.3dgtn_cls")
    parser.add_argument("--num_category", type=int, default=33, help="Number of classes")
    parser.add_argument("--epoch", type=int, default=200, help="Total epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--num_point", type=int, default=1024, help="Points per sample")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"], help="Optimizer type")
    parser.add_argument("--log_dir", type=str, default=None, help="Experiment name (folder under log/classification/)")
    parser.add_argument("--decay_rate", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--use_normals", action="store_true", help="(Reserved) Use normals if dataset has them")
    parser.add_argument("--process_data", action="store_true", help="(Reserved) Offline processing switch")
    parser.add_argument("--use_uniform_sample", action="store_true", help="Uniform sampling flag (ModelNetDataLoader)")
    parser.add_argument("--geometric_channel", type=int, nargs="+", help="Indices for geometric channels")
    parser.add_argument("--feature_channel", type=int, nargs="+", help="Indices for attribute channels")
    parser.add_argument("--dataset", type=str, default="TreeSpeciesCls",
                        choices=["TreeSpeciesNpy", "TreeSpeciesCls"],
                        help="Dataset type")
    parser.add_argument("--data_path", type=str, default=None, help="Dataset location (folder under data/...)")
    parser.add_argument("--weighted_loss", type=bool, default=True, help="Use class-weighted loss")
    parser.add_argument("--sampling_rate", type=int, default=4, help="Sampling rate (passed to model)")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers (set >0 if IO-bound)")
    return parser.parse_args()


# ----------------------------- #
# Utils
# ----------------------------- #
def inplace_relu(m: nn.Module):
    """Make nn.ReLU layers in-place (memory-friendly)"""
    if m.__class__.__name__.find("ReLU") != -1:
        m.inplace = True


def setup_logger(log_dir: Path, name: str = "Model") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(str(log_dir / "train.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def copy_training_artifacts(exp_dir: Path, model_path: str):
    """Copy the model file and training script for reproducibility."""
    model_file = os.path.join("models", *model_path.split(".")) + ".py"
    if os.path.exists(model_file):
        shutil.copy(model_file, str(exp_dir))
    if os.path.exists("train.py"):
        shutil.copy("train.py", str(exp_dir))


def get_device(args) -> torch.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    use_cuda = (not args.use_cpu) and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


# ----------------------------- #
# Data
# ----------------------------- #
def load_datasets(args):
    """
    Return:
        train_dataset, val_dataset, class_names(list[str])
    """
    if args.dataset == "TreeSpeciesNpy":
        root = args.data_path
        class_names = [line.rstrip() for line in open("./data/preprocessed/Tree_names.txt")]
        args.data_root = root
        print("Loading TreeSpeciesNpy training data ...")
        train_dataset = TreeSpeciesNpyDataset(root, args, split="train")
        print("Loading TreeSpeciesNpy validation data ...")
        val_dataset = TreeSpeciesNpyDataset(root, args, split="validate")

    elif args.dataset == "TreeSpeciesCls":
        data_path = args.data_path
        catfile = os.path.join(data_path, "Tree_names.txt")
        class_names = [line.rstrip() for line in open(catfile)]
        train_dataset = DataLoader(root=data_path, args=args, split="train", process_data=args.process_data)
        val_dataset = DataLoader(root=data_path, args=args, split="validate", process_data=args.process_data)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return train_dataset, val_dataset, class_names


def make_dataloaders(train_dataset, val_dataset, args):
    pin = torch.cuda.is_available() and (not args.use_cpu)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=pin,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader


# ----------------------------- #
# Evaluation
# ----------------------------- #
@torch.no_grad()
def evaluate(classifier, criterion, val_loader, class_names, args, weight, device, logger):
    classifier.eval()
    confusion = ConfusionMatrix(num_classes=args.num_category, labels=class_names)
    loss_sum = 0.0

    for points, target in tqdm(val_loader, total=len(val_loader), desc="Val", smoothing=0.9):
        points = points.float().to(device)            # [B, N, C]
        target = target.long().to(device)             # [B]
        points = points.transpose(2, 1)               # -> [B, C, N]

        logits, _ = classifier(points)                # [B, NUM_CLASSES]
        val_loss = criterion(logits, target, weight)  # keep original signature
        loss_sum += float(val_loss.item())

        pred_choice = logits.data.max(1)[1]           # [B]
        confusion.update(pred_choice.cpu().numpy(), target.cpu().numpy())

    OA, mrec, mF1, _, table = confusion.summary()
    logger.info("eval mean loss: %.6f", (loss_sum / float(len(val_loader))))
    logger.info("eval avg class accuracy (mAcc): %.6f", mrec)
    logger.info("eval overall accuracy (OA): %.6f", OA)
    logger.info("eval macro F1: %.6f", mF1)
    logger.info("\n%s", str(table))
    return OA, mrec, mF1


# ----------------------------- #
# Main
# ----------------------------- #
def main():
    args = build_args()

    # Device & cuDNN
    device = get_device(args)
    torch.backends.cudnn.benchmark = True

    # Exp folders
    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_root = Path("./log").joinpath("classification")
    exp_root.mkdir(parents=True, exist_ok=True)
    exp_dir = exp_root.joinpath(args.log_dir if args.log_dir else timestr)
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = exp_dir.joinpath("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath("logs")
    log_dir.mkdir(exist_ok=True)

    # Logger
    logger = setup_logger(log_dir)
    logger.info("PARAMETERS:")
    logger.info(args)

    # Data
    logger.info("Load dataset ...")
    train_dataset, val_dataset, class_names = load_datasets(args)
    train_loader, val_loader = make_dataloaders(train_dataset, val_dataset, args)

    # Model & Loss
    logger.info("Build model ...")
    import importlib
    model = importlib.import_module(args.model)
    copy_training_artifacts(exp_dir, args.model)

    classifier = model.get_model(
        args.num_category, args.geometric_channel, args.feature_channel, args.sampling_rate, args.num_point
    )
    criterion = model.get_loss_weighted().to(device) if args.weighted_loss else model.get_loss().to(device)
    classifier.apply(inplace_relu)
    classifier = classifier.to(device)

    # Class weights (for weighted loss signature: criterion(logits, target, weight))
    if args.weighted_loss:
        # Try to read label weights from dataset; default to ones if unavailable.
        weight_np = getattr(train_dataset, "labelweights", None)
        if weight_np is None:
            logger.warning("No labelweights found in train_dataset; fallback to uniform weights.")
            weight = torch.ones(args.num_category, dtype=torch.float32, device=device)
        else:
            logger.info("Label weights: %s", str(weight_np))
            weight = torch.tensor(weight_np, dtype=torch.float32, device=device)
    else:
        weight = torch.ones(args.num_category, dtype=torch.float32, device=device)

    # Optimizer & Scheduler
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.decay_rate
        )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.learning_rate / 100.0)

    # Resume (if last.pth exists)
    best_OA = 0.0
    best_mAcc = 0.0
    best_mF1 = 0.0
    start_epoch = 0
    last_ckpt = ckpt_dir / "last.pth"

    if last_ckpt.is_file():
        ckpt = torch.load(str(last_ckpt), map_location="cpu")
        classifier.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_OA = ckpt.get("best_acc", best_OA)
        best_mAcc = ckpt.get("best_macc", best_mAcc)
        best_mF1 = ckpt.get("best_ave_F1_score", best_mF1)
        logger.info("Resumed from epoch %d", start_epoch)
    else:
        logger.info("No existing checkpoint, start from scratch.")

    # Train
    logger.info("Start training ...")
    logger.info("Total parameters: %d", sum(p.numel() for p in classifier.parameters()))

    for epoch in range(start_epoch, args.epoch):
        logger.info("Epoch %d/%d", epoch + 1, args.epoch)
        classifier.train()
        
        logger.info("Learning rate: %.6f", optimizer.param_groups[0]["lr"])

        loss_sum = 0.0
        confusion_train = ConfusionMatrix(num_classes=args.num_category, labels=class_names)

        for _, (points, target) in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9, desc="Train"):
            optimizer.zero_grad()

            # NumPy-based augmentation (same as your original)
            points_np = points.numpy()
            points_np = provider.random_point_dropout(points_np)
            points_np[:, :, :3] = provider.shift_point_cloud(points_np[:, :, :3], shift_range=1)
            points_np[:, :, :3] = provider.rotate_point_cloud_z(points_np[:, :, :3])
            points_np[:, :, :3] = provider.jitter_point_cloud(points_np[:, :, :3])

            points = torch.tensor(points_np, dtype=torch.float32).transpose(2, 1)  # [B, C, N]
            target = target.long()

            points = points.to(device)
            target = target.to(device)

            logits, _ = classifier(points)
            loss = criterion(logits, target, weight)  # keep original signature

            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())

            pred_choice = logits.data.max(1)[1]
            confusion_train.update(pred_choice.cpu().numpy(), target.cpu().numpy())
        
        scheduler.step()
        
        train_OA = float(confusion_train.OA())
        logger.info("Training mean loss: %.6f", loss_sum / float(len(train_loader)))
        logger.info("Train Overall Accuracy (OA): %.6f", train_OA)

        # ---- Validation ----
        OA, mAcc, mF1 = evaluate(classifier, criterion, val_loader, class_names, args, weight, device, logger)

        # Save "last"
        state = {
            "epoch": epoch,
            "model_state_dict": classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "OA": OA,
            "mAcc": mAcc,
            "mF1": mF1,
            "best_acc": best_OA,
            "best_macc": best_mAcc,
            "best_ave_F1_score": best_mF1,
        }
        torch.save(state, str(ckpt_dir / "last.pth"))
        logger.info("Saved last.pth (best OA so far = %.4f)", best_OA)

        # Save best by different metrics
        if mAcc >= best_mAcc:
            best_mAcc = mAcc
            torch.save(state, str(ckpt_dir / "best_macc.pth"))
            logger.info("Saved best_macc.pth")

        if mF1 >= best_mF1:
            best_mF1 = mF1
            torch.save(state, str(ckpt_dir / "best_f1.pth"))
            logger.info("Saved best_f1.pth")

        if OA >= best_OA:
            best_OA = OA
            torch.save(state, str(ckpt_dir / "best_oa.pth"))
            logger.info("Saved best_oa.pth")

        logger.info("Best_OA: %.6f | Best_mAcc: %.6f | Best_mF1: %.6f", best_OA, best_mAcc, best_mF1)

        # Housekeeping
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("End of training.")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
