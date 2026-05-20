#!/usr/bin/env python3
"""Evaluate raw LASA SFP checkpoints without obstacle shaping.

This is a checkpoint sanity-check script: for each selected task/fold it loads
the SFP checkpoint, rolls out the raw policy on the held-out LASA demo, writes a
live quantitative CSV, and saves a trajectory plot against the ground truth.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_FILE = Path(__file__).resolve()
CASF_ROOT = THIS_FILE.parent
PROJECT_ROOT_LOCAL = CASF_ROOT.parent
for path in (CASF_ROOT, PROJECT_ROOT_LOCAL):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from lasa_test import (
    DEFAULT_DATA_DIR,
    DEFAULT_TASKS,
    PROJECT_ROOT,
    arc_length_parameter,
    checkpoint_path_for_task,
    default_device,
    import_legacy_modules,
    import_torch,
    load_policy,
    make_loo_test_dataset,
    mse_and_final_dist,
    parse_csv_list,
    parse_int_selection,
    resample_by_arclength,
    rollout_policy,
)


DEFAULT_RAW_OUT_DIR = PROJECT_ROOT / "outputs" / "lasa_raw_eval"
DEFAULT_LOO_CHECKPOINT_TEMPLATE = (
    PROJECT_ROOT
    / "models"
    / "CASF_lasaTask_ah8_{task}_fold{fold}_sfpdObs_1000ep_lr0.0001_obsDim4_demo6-1_norm.pth"
)
RAW_FIELDNAMES = (
    "task",
    "fold",
    "demo_idx",
    "valid",
    "checkpoint",
    "plot_path",
    "mse",
    "rmse",
    "mean_l2",
    "max_l2",
    "final_dist",
    "endpoint_dist",
    "steps_taken",
    "gt_len",
    "pred_len",
    "path_length_gt",
    "path_length_pred",
    "path_length_ratio",
    "discrete_frechet",
)


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(points[1:] - points[:-1], axis=-1)))


def discrete_frechet(pred: np.ndarray, reference: np.ndarray, *, n_interp: int = 300) -> float:
    if len(pred) < 3 or len(reference) < 3:
        return float("nan")
    pred_interp = resample_by_arclength(pred, n_interp)
    ref_interp = resample_by_arclength(reference, n_interp)
    n_pred, n_ref = len(pred_interp), len(ref_interp)
    ca = np.full((n_pred, n_ref), np.inf, dtype=np.float64)
    ca[0, 0] = float(np.linalg.norm(pred_interp[0] - ref_interp[0]))
    for i in range(1, n_pred):
        ca[i, 0] = max(ca[i - 1, 0], float(np.linalg.norm(pred_interp[i] - ref_interp[0])))
    for j in range(1, n_ref):
        ca[0, j] = max(ca[0, j - 1], float(np.linalg.norm(pred_interp[0] - ref_interp[j])))
    for i in range(1, n_pred):
        for j in range(1, n_ref):
            dist = float(np.linalg.norm(pred_interp[i] - ref_interp[j]))
            ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), dist)
    return float(ca[-1, -1])


def trajectory_metrics(pred: np.ndarray, gt: np.ndarray, rollout: dict[str, Any]) -> dict[str, float | int]:
    n = min(len(pred), len(gt))
    mse, final_dist = mse_and_final_dist(pred, gt)
    if n > 0:
        l2 = np.linalg.norm(pred[:n] - gt[:n], axis=-1)
        mean_l2 = float(np.mean(l2))
        max_l2 = float(np.max(l2))
    else:
        mean_l2 = max_l2 = float("nan")
    gt_path = path_length(gt)
    pred_path = path_length(pred)
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)) if math.isfinite(mse) else float("nan"),
        "mean_l2": mean_l2,
        "max_l2": max_l2,
        "final_dist": final_dist,
        "endpoint_dist": float(rollout["endpoint_dist"]),
        "steps_taken": int(rollout["steps_taken"]),
        "gt_len": int(len(gt)),
        "pred_len": int(len(pred)),
        "path_length_gt": gt_path,
        "path_length_pred": pred_path,
        "path_length_ratio": pred_path / gt_path if gt_path > 1e-12 else float("nan"),
        "discrete_frechet": discrete_frechet(pred, gt),
    }


def save_plot(path: Path, *, task: str, fold: int, demo_idx: int, pred: np.ndarray, gt: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(gt[:, 0], gt[:, 1], color="black", linewidth=2.0, label="GT held-out demo")
    ax.plot(pred[:, 0], pred[:, 1], color="#1f77b4", linewidth=2.0, label="Raw SFP rollout")
    ax.scatter(gt[0, 0], gt[0, 1], color="black", marker="o", s=36)
    ax.scatter(gt[-1, 0], gt[-1, 1], color="black", marker="x", s=48)
    ax.scatter(pred[0, 0], pred[0, 1], color="#1f77b4", marker="o", s=30)
    ax.scatter(pred[-1, 0], pred[-1, 1], color="#1f77b4", marker="x", s=44)
    ax.set_title(f"{task} fold {fold} demo {demo_idx}")
    ax.set_xlabel("x normalized")
    ax.set_ylabel("y normalized")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_FIELDNAMES)
        writer.writeheader()


def append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_FIELDNAMES, extrasaction="ignore")
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RAW_OUT_DIR)
    parser.add_argument("--checkpoint-template", default=str(DEFAULT_LOO_CHECKPOINT_TEMPLATE))
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--folds", default="all")
    parser.add_argument("--num-demos", type=int, default=7)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--skip-missing-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--pred-horizon", type=int, default=16)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--obs-dim", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--rollout-factor", type=float, default=1.2)
    parser.add_argument("--min-completion-frac", type=float, default=0.9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    tasks = parse_csv_list(args.tasks)
    folds = parse_int_selection(args.folds, max_count=args.num_demos)
    csv_path = out_dir / "lasa_raw_trials.csv"
    plot_dir = out_dir / "plots"

    if not data_dir.is_dir():
        raise FileNotFoundError(f"LASA data directory not found: {data_dir}")

    if args.dry_run:
        print(f"Data dir: {data_dir}")
        print(f"Output CSV: {csv_path}")
        print(f"Plot dir: {plot_dir}")
        for task in tasks:
            mat_path = data_dir / f"{task}.mat"
            print(f"  {task:10s} data={'OK' if mat_path.is_file() else 'MISSING'}")
            for fold in folds:
                ckpt_path = checkpoint_path_for_task(
                    args.checkpoint_template,
                    task,
                    fold=fold,
                    heldout_demo=fold,
                    train_demos=args.num_demos - 1,
                    test_demos=1,
                )
                print(f"    fold={fold}: checkpoint={'OK' if ckpt_path.is_file() else 'MISSING'} {ckpt_path}")
        return 0

    torch = import_torch()
    device = torch.device(args.device)
    (
        ConditionalUnet1D,
        _DatasetCls,
        StreamingFlowPolicyDeterministic,
        _shape_velocity_batch_metric_CASF,
        _shape_velocity_batch_hardBarrier,
        _shape_velocity_batch_CBF,
    ) = import_legacy_modules()

    write_header(csv_path)
    print(f"[live] writing raw SFP rows to {csv_path}")
    num_rows = 0
    for task in tasks:
        mat_path = data_dir / f"{task}.mat"
        if not mat_path.is_file():
            raise FileNotFoundError(f"Missing LASA task file: {mat_path}")
        for fold in folds:
            ckpt_path = checkpoint_path_for_task(
                args.checkpoint_template,
                task,
                fold=fold,
                heldout_demo=fold,
                train_demos=args.num_demos - 1,
                test_demos=1,
            )
            if not ckpt_path.is_file():
                message = f"Missing checkpoint for {task} fold {fold}: {ckpt_path}"
                if args.skip_missing_checkpoints:
                    print(f"[skip] {message}")
                    continue
                raise FileNotFoundError(message)

            print(f"[task] raw SFP {task} fold={fold}: loading {ckpt_path}")
            policy = load_policy(
                ckpt_path,
                device=device,
                ConditionalUnet1D=ConditionalUnet1D,
                StreamingFlowPolicyDeterministic=StreamingFlowPolicyDeterministic,
                pred_horizon=args.pred_horizon,
                obs_horizon=args.obs_horizon,
                obs_dim=args.obs_dim,
                action_dim=args.action_dim,
            )
            _train_indices, test_ds = make_loo_test_dataset(
                data_dir=data_dir,
                task=task,
                heldout_demo=fold,
                num_demos=args.num_demos,
                pred_horizon=args.pred_horizon,
                obs_horizon=args.obs_horizon,
                obs_dim=args.obs_dim,
            )
            for demo in test_ds:
                demo_idx = int(demo["demo_idx"])
                rollout = rollout_policy(
                    policy,
                    demo,
                    device=device,
                    action_horizon=args.action_horizon,
                    obs_horizon=args.obs_horizon,
                    rollout_factor=args.rollout_factor,
                    min_completion_frac=args.min_completion_frac,
                    post_shaping=None,
                    shaping_config=None,
                )
                pred = np.asarray(rollout["trajectory"], dtype=np.float32)
                gt = np.asarray(demo["action"], dtype=np.float32)
                plot_path = plot_dir / f"{task}_fold{fold}_demo{demo_idx}_raw_sfp.png"
                save_plot(plot_path, task=task, fold=fold, demo_idx=demo_idx, pred=pred, gt=gt)
                row = {
                    "task": task,
                    "fold": fold,
                    "demo_idx": demo_idx,
                    "valid": int(bool(rollout["valid"])),
                    "checkpoint": str(ckpt_path),
                    "plot_path": str(plot_path),
                    **trajectory_metrics(pred, gt, rollout),
                }
                append_row(csv_path, row)
                num_rows += 1
                print(
                    f"[row] {task} fold={fold} demo={demo_idx} valid={row['valid']} "
                    f"mse={row['mse']:.6f} final={row['final_dist']:.4f} "
                    f"frechet={row['discrete_frechet']:.4f}"
                )

    print(f"[done] wrote {num_rows} rows to {csv_path}")
    print(f"[done] wrote plots to {plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
