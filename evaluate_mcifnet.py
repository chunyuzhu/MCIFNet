"""Evaluate the MCIFNet fusion model and export metrics.

OA/AA/Kappa require a separate classifier's predicted class map and a ground-truth
class map. MCIFNet itself outputs a continuous hyperspectral cube, not classes.
"""
import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.io as sio
import torch
from torch import nn

from data_loader import build_datasets
from metrics import calc_ergas, calc_psnr, calc_rmse, calc_sam
from models.MCIFNet import MCIFNet

BANDS = {"PaviaU": 103, "Pavia": 102, "Chikusei": 128,
         "IEEE2018": 48, "Botswana": 145}


def load_labels(path, key):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        labels = np.load(path)
    elif path.suffix.lower() == ".mat":
        data = sio.loadmat(path)
        if key not in data:
            raise KeyError(f"{path}: missing MATLAB variable {key!r}")
        labels = data[key]
    else:
        raise ValueError("Label maps must be .npy or .mat files")
    labels = np.squeeze(labels)
    if labels.shape != (192, 192):
        raise ValueError(f"{path}: expected 192x192 test-crop label map; got {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.isfinite(labels)) or not np.all(labels == np.rint(labels)):
            raise ValueError(f"{path}: class IDs must be finite integers")
        labels = labels.astype(np.int64)
    return labels


def classification_scores(truth, prediction, ignore_label=0):
    """Return OA, AA, Kappa in [0, 1], ignoring unlabeled truth pixels."""
    valid = truth != ignore_label
    if not np.any(valid):
        raise ValueError("Ground truth has no labeled pixels")
    y_true = truth[valid].ravel()
    y_pred = prediction[valid].ravel()
    classes = np.unique(y_true)
    oa = float(np.mean(y_true == y_pred))
    aa = float(np.mean([np.mean(y_pred[y_true == c] == c) for c in classes]))
    all_classes = np.union1d(y_true, y_pred)
    true_hist = np.array([(y_true == c).sum() for c in all_classes], dtype=float)
    pred_hist = np.array([(y_pred == c).sum() for c in all_classes], dtype=float)
    pe = float(np.dot(true_hist, pred_hist) / y_true.size**2)
    kappa = float((oa - pe) / (1.0 - pe)) if pe < 1.0 else (1.0 if oa == 1.0 else 0.0)
    return oa, aa, kappa, int(y_true.size), int(classes.size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data")
    parser.add_argument("--dataset", choices=sorted(BANDS), default="IEEE2018")
    parser.add_argument("--scale-ratio", type=int, default=4)
    parser.add_argument("--model-path", default="./checkpoints/{dataset}_MCIFNet_4.pkl")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--gt-labels", help="Ground-truth class map for the 192x192 test crop")
    parser.add_argument("--pred-labels", help="Classifier prediction class map for the same crop")
    parser.add_argument("--gt-key", default="labels")
    parser.add_argument("--pred-key", default="labels")
    parser.add_argument("--ignore-label", type=int, default=0)
    args = parser.parse_args()

    if bool(args.gt_labels) != bool(args.pred_labels):
        parser.error("--gt-labels and --pred-labels must be supplied together")
    if not torch.cuda.is_available():
        parser.error("MCIFNet.forward calls .cuda(); a CUDA GPU is required")
    model_path = Path(args.model_path.format(dataset=args.dataset))
    if not model_path.is_file():
        parser.error(f"Checkpoint not found: {model_path}")

    _, test_list = build_datasets(args.root, args.dataset, 128, 4, args.scale_ratio)
    ref, lr, hr = (x.float() for x in test_list)
    model = MCIFNet(
        img_size=64, patch_size=1, in_chans_MSI=4,
        in_chans_HSI=BANDS[args.dataset], embed_dim=96,
        depths=(1,), mlp_dim=[256, 128], drop_rate=0.,
        d_state=16, mlp_ratio=2., drop_path_rate=0.1,
        norm_layer=nn.LayerNorm, patch_norm=True,
        use_checkpoint=False, upscale=2, img_range=1.,
        upsampler="", resi_connection="1conv",
    ).cuda()
    model.load_state_dict(torch.load(model_path, map_location="cuda"), strict=True)
    model.eval()

    with torch.inference_mode():
        lr, hr = lr.cuda(), hr.cuda()
        torch.cuda.synchronize()
        start = perf_counter()
        out = model(lr, hr)[0]
        torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - start) * 1000.0

    reference = ref.cpu().numpy()
    prediction = out.cpu().numpy()
    row = {
        "dataset": args.dataset, "model": "MCIFNet",
        "scale_ratio": args.scale_ratio, "checkpoint": str(model_path),
        "RMSE": float(calc_rmse(reference, prediction)),
        "PSNR": float(calc_psnr(reference, prediction)),
        "ERGAS": float(calc_ergas(reference, prediction)),
        "SAM": float(calc_sam(reference, prediction)),
        "inference_ms": elapsed_ms,
        "OA": "", "AA": "", "Kappa": "", "labeled_pixels": "", "classes": "",
    }
    if args.gt_labels:
        truth = load_labels(args.gt_labels, args.gt_key)
        labels = load_labels(args.pred_labels, args.pred_key)
        oa, aa, kappa, count, classes = classification_scores(
            truth, labels, args.ignore_label
        )
        row.update(OA=oa, AA=aa, Kappa=kappa,
                   labeled_pixels=count, classes=classes)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_MCIFNet_x{args.scale_ratio}"
    csv_path = output_dir / f"{stem}_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    sio.savemat(output_dir / f"{stem}_prediction.mat",
                {"Out": np.squeeze(prediction).transpose(1, 2, 0)})
    print(f"Metrics: {csv_path}")
    for name in ("RMSE", "PSNR", "ERGAS", "SAM", "OA", "AA", "Kappa"):
        if row[name] != "":
            print(f"{name}: {row[name]:.6f}")
    if not args.gt_labels:
        print("OA/AA/Kappa are empty: supply aligned ground-truth and classifier label maps.")


if __name__ == "__main__":
    main()
