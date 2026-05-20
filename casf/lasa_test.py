#!/usr/bin/env python3
"""Train/evaluate LASA obstacle-avoidance rollouts and export quantitative tables.

This script is a cleaned-up, reproducible version of
scripts_old/casf_lasaSWEEP_test.ipynb. By default it evaluates existing
checkpoints. With ``--train-first`` it first trains the missing LASA SFP
checkpoints from ``--data-dir`` and then evaluates the nominal SFP rollout plus
constraint-handling variants on LASA tasks, then writes:

  - lasa_trials.csv: one row per task/demo/method/config.
  - lasa_summary_by_config.csv: mean/std/SEM/95% CI for each config.
  - lasa_table_selected.csv: table-facing rows after selecting one config.
  - lasa_table_wide_mean.csv: compact table of means.
  - lasa_table_wide_mean_std.csv: compact table of mean +/- std.
  - lasa_table_wide_mean_ci95.csv: compact table of mean [95% CI].

The table metrics are:
  - Masked F.D.: masked discrete Frechet distance to the nominal SFP rollout.
  - M.P.D.: maximum penetration depth into the circular obstacle.
  - IntViolation: integrated penetration depth over rollout steps.

Use ``--fold-mode loo`` for seven-fold leave-one-demo-out evaluation. Std-dev
and confidence intervals are computed across the evaluated demos/trials within
each selected task/method/config group. With a single held-out LASA demo,
std-dev and CI are not statistically defined and are written as NaN.
"""

from __future__ import annotations

import argparse
import collections
import csv
import itertools
import math
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


THIS_FILE = Path(__file__).resolve()
CASF_ROOT = THIS_FILE.parent
PROJECT_ROOT = CASF_ROOT.parent
LEGACY_MODULE_ROOT = CASF_ROOT

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "lasa"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "lasa_quant"
DEFAULT_CHECKPOINT_TEMPLATE = (
    PROJECT_ROOT
    / "models"
    / "CASF_lasaTask_ah8_{task}_sfpdObs_1000ep_lr0.0001_obsDim4_demo6-1_norm.pth"
)
DEFAULT_TASKS = ("Line", "Khamesh", "NShape", "Sine", "RShape", "Sshape", "WShape", "Worm", "Zshape")
DEFAULT_METHODS = ("sfp", "projection", "cbf", "casf")
TABLE_METHOD_LABELS = {
    "sfp": "SFP",
    "projection": "Projection",
    "cbf": "CBF",
    "casf": "CASF",
}
TABLE_METRICS = {
    "Masked F.D.": "masked_frechet_vs_sfp",
    "M.P.D.": "max_pen_depth",
    "IntViolation": "int_violation",
}
SUMMARY_METRICS = (
    "masked_frechet_vs_sfp",
    "masked_frechet_vs_gt",
    "mse",
    "final_dist",
    "min_clearance",
    "max_pen_depth",
    "int_violation",
    "steps_taken",
    "lipschitz_max",
    "lipschitz_mean",
)
TRIAL_FIELDNAMES = (
    "task",
    "fold",
    "demo_idx",
    "heldout_demo",
    "train_demo_indices",
    "method",
    "method_label",
    "post_shaping",
    "alpha",
    "beta",
    "w_scale",
    "drift_gain",
    "eps",
    "valid",
    "steps_taken_sfp",
    "center_x",
    "center_y",
    "radius",
    "center_scale_x",
    "center_scale_y",
    "radius_scale",
    "checkpoint",
    *SUMMARY_METRICS,
)


@dataclass(frozen=True)
class MethodConfig:
    method: str
    post_shaping: str | None
    alpha: float
    beta: float
    w_scale: float
    drift_gain: float
    eps: float

    @property
    def label(self) -> str:
        return TABLE_METHOD_LABELS[self.method]


@dataclass(frozen=True)
class ObstacleConfig:
    center_norm: np.ndarray
    radius_norm: float
    center_scale: tuple[float, float]
    radius_scale: float

    def shaping_dict(self, method: MethodConfig) -> dict[str, Any]:
        return {
            "box_bounds": (-1, -1, 1, 1),
            "center_norm": self.center_norm,
            "radius_norm": self.radius_norm,
            "alpha": method.alpha,
            "beta": method.beta,
            "w_scale": method.w_scale,
            "drift_gain": method.drift_gain,
            "eps": method.eps,
        }


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(item) for item in parse_csv_list(raw)]


def finite_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def format_float(value: float, digits: int) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def parse_int_selection(raw: str, *, max_count: int | None = None) -> list[int]:
    if raw.strip().lower() == "all":
        if max_count is None:
            raise ValueError("'all' requires max_count")
        return list(range(max_count))
    selected: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            selected.extend(range(int(left), int(right) + 1))
        else:
            selected.append(int(part))
    return sorted(dict.fromkeys(selected))


def checkpoint_template_has_fold_field(template: str) -> bool:
    return any(field in template for field in ("{fold}", "{heldout}", "{heldout_demo}", "{demo}"))


def checkpoint_path_for_task(
    template: str,
    task: str,
    *,
    fold: int | None = None,
    heldout_demo: int | None = None,
    train_demos: int | None = None,
    test_demos: int | None = None,
) -> Path:
    value = template.format(
        task=task,
        fold="" if fold is None else fold,
        heldout="" if heldout_demo is None else heldout_demo,
        heldout_demo="" if heldout_demo is None else heldout_demo,
        demo="" if heldout_demo is None else heldout_demo,
        train_demos="" if train_demos is None else train_demos,
        test_demos="" if test_demos is None else test_demos,
    )
    return Path(value).expanduser().resolve()


def import_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - CLI dependency error.
        raise RuntimeError(
            "Torch is required for LASA rollout evaluation. Run this script in the "
            "same environment used for the old CASF notebook."
        ) from exc
    return torch


def default_device() -> str:
    try:
        torch = import_torch()
    except RuntimeError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def install_pydrake_fallback_if_needed() -> None:
    try:
        import pydrake.all  # noqa: F401
        return
    except Exception:
        pass

    class PiecewisePolynomial:
        def __init__(self, times: np.ndarray, values: np.ndarray):
            self.times = np.asarray(times, dtype=np.float64)
            self.values = np.asarray(values, dtype=np.float64)
            if self.values.ndim != 2:
                raise ValueError("PiecewisePolynomial values must be 2-D")
            if self.values.shape[1] != self.times.shape[0]:
                raise ValueError("PiecewisePolynomial expects values shaped (dim, num_times)")

        @classmethod
        def FirstOrderHold(cls, times: np.ndarray, values: np.ndarray) -> "PiecewisePolynomial":
            return cls(times, values)

        def value(self, t: float) -> np.ndarray:
            t_float = float(np.asarray(t))
            out = [np.interp(t_float, self.times, self.values[dim]) for dim in range(self.values.shape[0])]
            return np.asarray(out, dtype=np.float64)[:, None]

        def EvalDerivative(self, t: float) -> np.ndarray:
            t_float = float(np.asarray(t))
            idx = int(np.searchsorted(self.times, t_float, side="right") - 1)
            idx = max(0, min(idx, len(self.times) - 2))
            dt = max(float(self.times[idx + 1] - self.times[idx]), 1e-12)
            deriv = (self.values[:, idx + 1] - self.values[:, idx]) / dt
            return np.asarray(deriv, dtype=np.float64)[:, None]

    pydrake_module = types.ModuleType("pydrake")
    pydrake_all_module = types.ModuleType("pydrake.all")
    pydrake_all_module.PiecewisePolynomial = PiecewisePolynomial
    pydrake_module.all = pydrake_all_module
    sys.modules.setdefault("pydrake", pydrake_module)
    sys.modules.setdefault("pydrake.all", pydrake_all_module)


def import_legacy_modules() -> tuple[Any, ...]:
    install_pydrake_fallback_if_needed()
    for path in (LEGACY_MODULE_ROOT, PROJECT_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    try:
        from streaming_flow_policy.pusht.dp_state_notebook.network import ConditionalUnet1D
        from streaming_flow_policy.casf_lasa.dataset import MatSequenceDatasetWithNextObsAsAction_Task
        from streaming_flow_policy.casf_lasa.sfpd import (
            StreamingFlowPolicyDeterministic,
            shape_velocity_batch_CBF,
            shape_velocity_batch_hardBarrier,
            shape_velocity_batch_metric_CASF,
        )
    except Exception as exc:  # pragma: no cover - meant to give a useful CLI error.
        raise RuntimeError(
            "Could not import the legacy LASA SFP/CASF modules from "
            f"{LEGACY_MODULE_ROOT}. Make sure this environment has the notebook "
            "dependencies installed, including torch, pydrake, torchdyn, and torchdiffeq."
        ) from exc
    return (
        ConditionalUnet1D,
        MatSequenceDatasetWithNextObsAsAction_Task,
        StreamingFlowPolicyDeterministic,
        shape_velocity_batch_metric_CASF,
        shape_velocity_batch_hardBarrier,
        shape_velocity_batch_CBF,
    )


def get_data_stats(data: np.ndarray) -> dict[str, np.ndarray]:
    flat = data.reshape(-1, data.shape[-1])
    return {
        "min": np.min(flat, axis=0),
        "max": np.max(flat, axis=0),
    }


def normalize_data(data: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    denom = np.maximum(stats["max"] - stats["min"], 1e-8)
    return ((data - stats["min"]) / denom) * 2.0 - 1.0


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    *,
    pad_before: int = 0,
    pad_after: int = 0,
) -> np.ndarray:
    indices = []
    for episode_idx in range(len(episode_ends)):
        start_idx = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
        end_idx = int(episode_ends[episode_idx])
        episode_length = end_idx - start_idx
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            indices.append([buffer_start_idx, buffer_end_idx, start_offset, sequence_length - end_offset])
    return np.asarray(indices, dtype=np.int64)


def sample_sequence(
    data: dict[str, np.ndarray],
    sequence_length: int,
    buffer_start_idx: int,
    buffer_end_idx: int,
    sample_start_idx: int,
    sample_end_idx: int,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, arr in data.items():
        sample = arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0 or sample_end_idx < sequence_length:
            padded = np.zeros((sequence_length,) + arr.shape[1:], dtype=arr.dtype)
            if sample_start_idx > 0:
                padded[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                padded[sample_end_idx:] = sample[-1]
            padded[sample_start_idx:sample_end_idx] = sample
            sample = padded
        result[key] = sample
    return result


def load_lasa_demos(mat_path: Path) -> list[dict[str, np.ndarray]]:
    try:
        import scipy.io as sio
    except Exception as exc:  # pragma: no cover - CLI dependency error.
        raise RuntimeError("scipy is required to load LASA .mat files.") from exc

    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    demos_raw = np.ravel(data["demos"])
    demos: list[dict[str, np.ndarray]] = []
    for demo in demos_raw:
        pos = np.asarray(demo.pos, dtype=np.float32).T
        vel = np.asarray(demo.vel, dtype=np.float32).T
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError(f"{mat_path.name}: expected demo.pos to become (T, 2), got {pos.shape}")
        if vel.shape != pos.shape:
            raise ValueError(f"{mat_path.name}: expected demo.vel shape {pos.shape}, got {vel.shape}")
        demos.append({"pos": pos, "vel": vel})
    return demos


def next_position_actions(pos: np.ndarray) -> np.ndarray:
    return np.concatenate([pos[1:], pos[-1:]], axis=0).astype(np.float32)


def compute_fold_stats(demos: list[dict[str, np.ndarray]], train_indices: list[int]) -> dict[str, Any]:
    train_pos = np.concatenate([demos[idx]["pos"] for idx in train_indices], axis=0)
    pos_stats = get_data_stats(train_pos)
    xmin, ymin = pos_stats["min"]
    xmax, ymax = pos_stats["max"]
    vel_scale = 2.0 / np.maximum(np.array([xmax - xmin, ymax - ymin], dtype=np.float32), 1e-8)
    return {
        "pos": pos_stats,
        # The legacy dataset computes action stats before replacing actions with
        # next-position targets, so action stats are effectively position stats.
        "action": {"min": pos_stats["min"].copy(), "max": pos_stats["max"].copy()},
        "vel": vel_scale.astype(np.float32),
    }


def make_initial_obs(pos_norm: np.ndarray, *, pred_horizon: int, obs_horizon: int, obs_dim: int) -> np.ndarray:
    if len(pos_norm) < pred_horizon:
        pad = np.repeat(pos_norm[-1:], pred_horizon - len(pos_norm), axis=0)
        traj_pos = np.concatenate([pos_norm, pad], axis=0)
    else:
        traj_pos = pos_norm[:pred_horizon]
    traj_time = np.linspace(0.0, 1.0, pred_horizon)

    # Match the legacy dataset's PiecewisePolynomial.FirstOrderHold obs builder.
    try:
        from pydrake.all import PiecewisePolynomial

        segment = PiecewisePolynomial.FirstOrderHold(traj_time, traj_pos.T)
        xs = np.concatenate([segment.value(t).T for t in traj_time], axis=0)
        vs = np.concatenate([segment.EvalDerivative(t).T for t in traj_time], axis=0)
    except Exception:
        xs = traj_pos
        dt = 1.0 / max(pred_horizon - 1, 1)
        vs = np.gradient(traj_pos, dt, axis=0)

    if obs_dim == 4:
        obs = np.concatenate([xs, vs], axis=-1)
    elif obs_dim == 2:
        obs = xs
    else:
        raise ValueError(f"Unsupported LASA obs_dim={obs_dim}; expected 2 or 4.")
    return obs[:obs_horizon].astype(np.float32)


def make_training_transform(*, pred_horizon: int, sigma: float) -> Any:
    """Create a worker-safe LASA SFP training transform.

    The legacy ``policy.TransformTrainingDatum`` reads ``self.pred_horizon``
    from a torch buffer. If the policy is on CUDA and DataLoader workers are
    enabled, that buffer access initializes CUDA in the worker process. Keeping
    this transform NumPy-only avoids that failure while matching the same
    sampled continuous-trajectory target used by the legacy policy.
    """

    def transform(datum: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        obs = np.asarray(datum["obs"], dtype=np.float32)
        action = np.asarray(datum["action"], dtype=np.float32)
        obs_horizon, _ = obs.shape
        action_horizon, _ = action.shape
        if action_horizon != pred_horizon:
            raise AssertionError(f"Expected action horizon {pred_horizon}, got {action_horizon}")
        if obs_horizon != 2:
            raise AssertionError(f"Expected obs horizon 2, got {obs_horizon}")
        if not np.all(action[0] == obs[-1, :2]):
            action = action.copy()
            action[0] = obs[-1, :2]

        traj_times = np.linspace(0.0, 1.0, pred_horizon)
        try:
            install_pydrake_fallback_if_needed()
            from pydrake.all import PiecewisePolynomial

            traj = PiecewisePolynomial.FirstOrderHold(traj_times, action.T)
            time_sample = np.float32(np.random.rand())
            x = traj.value(time_sample).T.astype(np.float32)
            v = traj.EvalDerivative(time_sample).T.astype(np.float32)
            dt = np.float32(1.0 / (pred_horizon - 1))
            x_next = traj.value(min(float(time_sample + dt), 1.0)).T.astype(np.float32)
        except Exception:
            time_sample = np.float32(np.random.rand())
            interp = np.array(
                [np.interp(float(time_sample), traj_times, action[:, dim]) for dim in range(action.shape[1])],
                dtype=np.float32,
            )[None, :]
            gradients = np.gradient(action, traj_times, axis=0).astype(np.float32)
            grad = np.array(
                [np.interp(float(time_sample), traj_times, gradients[:, dim]) for dim in range(action.shape[1])],
                dtype=np.float32,
            )[None, :]
            dt = np.float32(1.0 / (pred_horizon - 1))
            t_next = min(float(time_sample + dt), 1.0)
            x_next = np.array(
                [np.interp(t_next, traj_times, action[:, dim]) for dim in range(action.shape[1])],
                dtype=np.float32,
            )[None, :]
            x = interp
            v = grad

        x = (x + float(sigma) * np.random.randn(*x.shape)).astype(np.float32)
        return {
            "obs": obs,
            "action": action,
            "x": x.astype(np.float32),
            "v": v.astype(np.float32),
            "t": time_sample,
            "dt": dt,
            "x_next": x_next,
        }

    return transform


class LasaFoldTestDataset:
    """Minimal fold-aware LASA test dataset matching the legacy test samples."""

    def __init__(
        self,
        demos: list[dict[str, np.ndarray]],
        *,
        test_indices: list[int],
        stats: dict[str, Any],
        pred_horizon: int,
        obs_horizon: int,
        obs_dim: int,
    ):
        self.demos = demos
        self.test_indices = test_indices
        self.stats = stats
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.obs_dim = obs_dim

    def __len__(self) -> int:
        return len(self.test_indices)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | int]:
        demo_idx = self.test_indices[idx]
        demo = self.demos[demo_idx]
        pos_norm = normalize_data(demo["pos"], self.stats["pos"]).astype(np.float32)
        vel_norm = (demo["vel"] * self.stats["vel"]).astype(np.float32)
        action_norm = normalize_data(next_position_actions(demo["pos"]), self.stats["action"]).astype(np.float32)
        return {
            "demo_idx": demo_idx,
            "pos": pos_norm,
            "vel": vel_norm,
            "action": action_norm,
            "obs": make_initial_obs(
                pos_norm,
                pred_horizon=self.pred_horizon,
                obs_horizon=self.obs_horizon,
                obs_dim=self.obs_dim,
            ),
        }


class LasaFoldTrainDataset:
    """Fold-aware LASA train dataset with the same samples as the legacy trainer."""

    def __init__(
        self,
        demos: list[dict[str, np.ndarray]],
        *,
        train_indices: list[int],
        stats: dict[str, Any],
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        obs_dim: int,
        transform_datum_fn: Any,
    ):
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.obs_dim = obs_dim
        self.transform_datum_fn = transform_datum_fn
        self.stats = stats

        all_pos: list[np.ndarray] = []
        all_vel: list[np.ndarray] = []
        all_action: list[np.ndarray] = []
        episode_ends: list[int] = []
        offset = 0
        for demo_idx in train_indices:
            pos = demos[demo_idx]["pos"]
            vel = demos[demo_idx]["vel"]
            all_pos.append(pos)
            all_vel.append(vel)
            all_action.append(next_position_actions(pos))
            offset += len(pos)
            episode_ends.append(offset)

        pos_arr = np.concatenate(all_pos, axis=0).astype(np.float32)
        vel_arr = np.concatenate(all_vel, axis=0).astype(np.float32)
        action_arr = np.concatenate(all_action, axis=0).astype(np.float32)
        self.episode_ends = np.asarray(episode_ends, dtype=np.int64)
        self.normalized_data = {
            "pos": normalize_data(pos_arr, stats["pos"]).astype(np.float32),
            "vel": (vel_arr * stats["vel"]).astype(np.float32),
            "action": normalize_data(action_arr, stats["action"]).astype(np.float32),
        }
        self.indices = create_sample_indices(
            self.episode_ends,
            pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        sample = sample_sequence(
            self.normalized_data,
            self.pred_horizon,
            int(buffer_start_idx),
            int(buffer_end_idx),
            int(sample_start_idx),
            int(sample_end_idx),
        )
        sample["obs"] = make_initial_obs(
            sample["pos"],
            pred_horizon=self.pred_horizon,
            obs_horizon=self.obs_horizon,
            obs_dim=self.obs_dim,
        )
        if self.transform_datum_fn is not None:
            sample = self.transform_datum_fn(sample)
        return sample


def make_loo_test_dataset(
    *,
    data_dir: Path,
    task: str,
    heldout_demo: int,
    num_demos: int,
    pred_horizon: int,
    obs_horizon: int,
    obs_dim: int,
) -> tuple[list[int], LasaFoldTestDataset]:
    demos = load_lasa_demos(data_dir / f"{task}.mat")
    if num_demos > len(demos):
        raise ValueError(f"{task}: requested {num_demos} demos, but file contains {len(demos)}")
    if not 0 <= heldout_demo < num_demos:
        raise ValueError(f"{task}: heldout demo must be in [0, {num_demos - 1}], got {heldout_demo}")
    train_indices = [idx for idx in range(num_demos) if idx != heldout_demo]
    stats = compute_fold_stats(demos, train_indices)
    test_ds = LasaFoldTestDataset(
        demos[:num_demos],
        test_indices=[heldout_demo],
        stats=stats,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        obs_dim=obs_dim,
    )
    return train_indices, test_ds


def make_train_eval_dataset(
    *,
    data_dir: Path,
    task: str,
    num_demos: int,
    pred_horizon: int,
    obs_horizon: int,
    obs_dim: int,
) -> tuple[list[int], LasaFoldTestDataset]:
    demos = load_lasa_demos(data_dir / f"{task}.mat")
    if num_demos > len(demos):
        raise ValueError(f"{task}: requested {num_demos} demos, but file contains {len(demos)}")
    train_indices = list(range(num_demos))
    stats = compute_fold_stats(demos, train_indices)
    test_ds = LasaFoldTestDataset(
        demos[:num_demos],
        test_indices=train_indices,
        stats=stats,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        obs_dim=obs_dim,
    )
    return train_indices, test_ds


def make_loo_train_dataset(
    *,
    data_dir: Path,
    task: str,
    heldout_demo: int,
    num_demos: int,
    pred_horizon: int,
    obs_horizon: int,
    action_horizon: int,
    obs_dim: int,
    transform_datum_fn: Any,
) -> tuple[list[int], LasaFoldTrainDataset]:
    demos = load_lasa_demos(data_dir / f"{task}.mat")
    if num_demos > len(demos):
        raise ValueError(f"{task}: requested {num_demos} demos, but file contains {len(demos)}")
    if not 0 <= heldout_demo < num_demos:
        raise ValueError(f"{task}: heldout demo must be in [0, {num_demos - 1}], got {heldout_demo}")
    train_indices = [idx for idx in range(num_demos) if idx != heldout_demo]
    stats = compute_fold_stats(demos, train_indices)
    train_ds = LasaFoldTrainDataset(
        demos[:num_demos],
        train_indices=train_indices,
        stats=stats,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        obs_dim=obs_dim,
        transform_datum_fn=transform_datum_fn,
    )
    return train_indices, train_ds


def circle_sdf(traj: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    diff = traj - center[None, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1) + 1e-8)
    return dist - radius


def arc_length_parameter(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros((len(points),), dtype=np.float64)
    d = np.sqrt(np.sum((points[1:] - points[:-1]) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 1e-12:
        return np.linspace(0.0, 1.0, len(points))
    return s / s[-1]


def resample_by_arclength(points: np.ndarray, n_interp: int) -> np.ndarray:
    s = arc_length_parameter(points)
    s_uniform = np.linspace(0.0, 1.0, n_interp)
    return np.vstack([np.interp(s_uniform, s, points[:, dim]) for dim in range(points.shape[1])]).T


def masked_discrete_frechet(
    pred: np.ndarray,
    reference: np.ndarray,
    center: np.ndarray,
    radius: float,
    *,
    delta_clearance: float = 0.02,
    n_interp: int = 300,
) -> float:
    """Discrete Frechet distance after masking reference points near obstacle."""
    if len(pred) < 3 or len(reference) < 3:
        return float("nan")
    if np.sum(circle_sdf(reference, center, radius) > delta_clearance) < 3:
        return float("nan")

    pred_interp = resample_by_arclength(pred, n_interp)
    ref_interp = resample_by_arclength(reference, n_interp)
    valid_mask = circle_sdf(ref_interp, center, radius) > delta_clearance
    pred_masked = pred_interp[valid_mask]
    ref_masked = ref_interp[valid_mask]
    if len(pred_masked) < 3 or len(ref_masked) < 3:
        return float("nan")

    n_pred, n_ref = len(pred_masked), len(ref_masked)
    ca = np.full((n_pred, n_ref), np.inf, dtype=np.float64)
    d00 = float(np.linalg.norm(pred_masked[0] - ref_masked[0]))
    ca[0, 0] = d00
    for i in range(1, n_pred):
        ca[i, 0] = max(ca[i - 1, 0], float(np.linalg.norm(pred_masked[i] - ref_masked[0])))
    for j in range(1, n_ref):
        ca[0, j] = max(ca[0, j - 1], float(np.linalg.norm(pred_masked[0] - ref_masked[j])))
    for i in range(1, n_pred):
        for j in range(1, n_ref):
            d = float(np.linalg.norm(pred_masked[i] - ref_masked[j]))
            ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d)
    return float(ca[-1, -1])


def collision_metrics(traj: np.ndarray, center: np.ndarray, radius: float, *, dt: float = 1.0) -> dict[str, float]:
    phi = circle_sdf(traj, center, radius)
    penetration = np.maximum(0.0, -phi)
    return {
        "min_clearance": float(np.min(phi)),
        "max_pen_depth": float(np.max(penetration)),
        "int_violation": float(np.sum(penetration) * dt),
    }


def mse_and_final_dist(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    n = min(len(pred), len(gt))
    if n == 0:
        return float("nan"), float("nan")
    mse = float(np.mean((pred[:n] - gt[:n]) ** 2))
    final_dist = float(np.linalg.norm(pred[-1] - gt[-1]))
    return mse, final_dist


def make_obstacle(task: str, radius_scale: float) -> ObstacleConfig:
    if task in {"Khamesh"}:
        center_scale = (0.6, 0.5)
    elif task in {"NShape", "PShape", "RShape"}:
        center_scale = (0.25, 0.2)
    elif task in {"Sshape", "Line", "WShape", "Sine", "Worm", "Zshape", "ZShape"}:
        center_scale = (0.35, 0.5)
    elif task in {"Saeghe", "Snake", "heee"}:
        center_scale = (0.3, 0.75)
    elif task == "Trapezoid":
        center_scale = (0.3, 0.15)
    elif task == "BendedLine":
        center_scale = (0.15, 0.15)
    elif task == "DoubleBendedLine":
        center_scale = (0.15, 0.75)
    elif task == "Spoon":
        center_scale = (0.5, 0.5)
    else:
        center_scale = (0.5, 0.5)

    xmin_norm, ymin_norm = -1.0, -1.0
    xmax_norm, ymax_norm = 1.0, 1.0
    max_span_norm = max(xmax_norm - xmin_norm, ymax_norm - ymin_norm)
    workspace_norm_size = np.array([max_span_norm, max_span_norm], dtype=np.float32)
    center_arr = np.array(center_scale, dtype=np.float32)
    center_norm = np.array(
        [
            xmin_norm + center_arr[0] * workspace_norm_size[0],
            ymax_norm - center_arr[1] * workspace_norm_size[1],
        ],
        dtype=np.float32,
    )
    radius_norm = float(radius_scale * np.mean(workspace_norm_size))
    return ObstacleConfig(
        center_norm=center_norm,
        radius_norm=radius_norm,
        center_scale=center_scale,
        radius_scale=radius_scale,
    )


def build_method_configs(args: argparse.Namespace) -> list[MethodConfig]:
    requested = set(parse_csv_list(args.methods))
    configs: list[MethodConfig] = []
    eps = args.eps
    if "sfp" in requested:
        configs.append(MethodConfig("sfp", None, float("nan"), float("nan"), float("nan"), float("nan"), eps))
    if "projection" in requested:
        configs.append(MethodConfig("projection", "obstacle-hardB", 1.0, 0.0, 0.0, 0.0, eps))
    if "cbf" in requested:
        for alpha in parse_float_list(args.cbf_alphas):
            configs.append(MethodConfig("cbf", "obstacle-cbf", alpha, 0.0, 0.0, 0.0, eps))
    if "casf" in requested:
        for alpha, w_scale, drift_gain in itertools.product(
            parse_float_list(args.casf_alphas),
            parse_float_list(args.casf_w_scales),
            parse_float_list(args.casf_drift_gains),
        ):
            configs.append(MethodConfig("casf", "obstacle", alpha, args.casf_beta, w_scale, drift_gain, eps))
    unknown = requested.difference(DEFAULT_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(sorted(unknown))}")
    return configs


def load_policy(
    checkpoint_path: Path,
    *,
    device: torch.device,
    ConditionalUnet1D: Any,
    StreamingFlowPolicyDeterministic: Any,
    pred_horizon: int,
    obs_horizon: int,
    obs_dim: int,
    action_dim: int,
) -> Any:
    torch = import_torch()
    velocity_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
        fc_timesteps=1,
    )
    policy = StreamingFlowPolicyDeterministic(
        velocity_net=velocity_net,
        action_dim=action_dim,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        device=str(device),
    )
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "policy_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint does not contain a state dict: {checkpoint_path}")
    if payload and all(str(key).startswith("module.") for key in payload):
        payload = {str(key)[7:]: value for key, value in payload.items()}
    policy.load_state_dict(payload)
    policy.to(device)
    policy.eval()
    return policy


def dict_to_device(batch: Any, device: Any) -> Any:
    if isinstance(batch, dict):
        return {key: dict_to_device(value, device) for key, value in batch.items()}
    if hasattr(batch, "to"):
        return batch.to(device, non_blocking=True)
    return batch


class SimpleEMA:
    """Small EMA helper so training does not depend on diffusers."""

    def __init__(self, parameters: Iterable[Any], decay: float):
        self.decay = decay
        self.shadow = [param.detach().clone() for param in parameters if param.requires_grad]

    def step(self, parameters: Iterable[Any]) -> None:
        shadow_idx = 0
        for param in parameters:
            if not param.requires_grad:
                continue
            self.shadow[shadow_idx].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
            shadow_idx += 1

    def copy_to(self, parameters: Iterable[Any]) -> None:
        shadow_idx = 0
        for param in parameters:
            if not param.requires_grad:
                continue
            param.data.copy_(self.shadow[shadow_idx].data)
            shadow_idx += 1


def make_warmup_cosine_scheduler(optimizer: Any, *, warmup_steps: int, total_steps: int) -> Any:
    torch = import_torch()
    warmup_steps = max(0, min(warmup_steps, total_steps))
    total_steps = max(1, total_steps)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_fresh_policy(
    *,
    device: Any,
    ConditionalUnet1D: Any,
    StreamingFlowPolicyDeterministic: Any,
    pred_horizon: int,
    obs_horizon: int,
    obs_dim: int,
    action_dim: int,
    sigma: float,
) -> Any:
    velocity_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
        fc_timesteps=1,
    )
    policy = StreamingFlowPolicyDeterministic(
        velocity_net=velocity_net,
        action_dim=action_dim,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        sigma=sigma,
        device=device,
    )
    policy.to(device)
    return policy


def train_policy_checkpoint(
    *,
    args: argparse.Namespace,
    task: str,
    checkpoint_path: Path,
    data_dir: Path,
    device: Any,
    DatasetCls: Any,
    ConditionalUnet1D: Any,
    StreamingFlowPolicyDeterministic: Any,
    fold: int | None = None,
    heldout_demo: int | None = None,
    train_indices_override: list[int] | None = None,
) -> Path:
    torch = import_torch()
    policy = build_fresh_policy(
        device=device,
        ConditionalUnet1D=ConditionalUnet1D,
        StreamingFlowPolicyDeterministic=StreamingFlowPolicyDeterministic,
        pred_horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        sigma=args.train_sigma,
    )
    train_transform = make_training_transform(pred_horizon=args.pred_horizon, sigma=args.train_sigma)
    if train_indices_override is not None:
        demos = load_lasa_demos(data_dir / f"{task}.mat")
        stats = compute_fold_stats(demos, train_indices_override)
        train_ds = LasaFoldTrainDataset(
            demos[: args.num_demos],
            train_indices=train_indices_override,
            stats=stats,
            pred_horizon=args.pred_horizon,
            obs_horizon=args.obs_horizon,
            action_horizon=args.action_horizon,
            obs_dim=args.obs_dim,
            transform_datum_fn=train_transform,
        )
        train_indices = list(train_indices_override)
    elif heldout_demo is None:
        train_ds = DatasetCls(
            dataset_dir=str(data_dir),
            pred_horizon=args.pred_horizon,
            obs_horizon=args.obs_horizon,
            action_horizon=args.action_horizon,
            obs_dim=args.obs_dim,
            task=task,
            split="train",
            split_config={"train_demos": args.train_demos, "test_demos": args.test_demos},
            transform_datum_fn=train_transform,
        )
        train_indices = list(range(args.train_demos))
    else:
        train_indices, train_ds = make_loo_train_dataset(
            data_dir=data_dir,
            task=task,
            heldout_demo=heldout_demo,
            num_demos=args.num_demos,
            pred_horizon=args.pred_horizon,
            obs_horizon=args.obs_horizon,
            action_horizon=args.action_horizon,
            obs_dim=args.obs_dim,
            transform_datum_fn=train_transform,
        )

    num_workers = max(0, args.train_num_workers)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=num_workers > 0,
    )
    if len(train_loader) == 0:
        raise RuntimeError(f"{task}: training dataloader is empty")

    optimizer = torch.optim.AdamW(
        policy.velocity_net.parameters(),
        lr=args.train_lr,
        weight_decay=args.train_weight_decay,
    )
    total_steps = len(train_loader) * args.train_epochs
    lr_scheduler = make_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=args.train_warmup_steps,
        total_steps=total_steps,
    )
    ema = None if args.train_ema_decay <= 0 else SimpleEMA(policy.velocity_net.parameters(), args.train_ema_decay)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    losses_path = checkpoint_path.with_suffix(".loss.csv")
    start = time.time()
    print(
        f"[train] {task} fold={fold if fold is not None else 'last'} "
        f"train_demos={','.join(str(i) for i in train_indices)} samples={len(train_ds)} "
        f"epochs={args.train_epochs} batch={args.train_batch_size} -> {checkpoint_path}"
    )
    with losses_path.open("w", newline="") as loss_fh:
        writer = csv.writer(loss_fh)
        writer.writerow(["epoch", "mean_loss", "lr"])
        for epoch_idx in range(args.train_epochs):
            policy.train()
            epoch_losses: list[float] = []
            for batch in train_loader:
                batch = dict_to_device(batch, device)
                loss = policy.Loss(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                if ema is not None:
                    ema.step(policy.velocity_net.parameters())
                epoch_losses.append(float(loss.detach().cpu().item()))
            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            lr = float(optimizer.param_groups[0]["lr"])
            writer.writerow([epoch_idx + 1, mean_loss, lr])
            if (epoch_idx + 1) == 1 or (epoch_idx + 1) % args.train_log_every == 0 or (epoch_idx + 1) == args.train_epochs:
                elapsed_min = (time.time() - start) / 60.0
                print(
                    f"[train] {task} fold={fold if fold is not None else 'last'} "
                    f"epoch={epoch_idx + 1}/{args.train_epochs} loss={mean_loss:.6f} "
                    f"lr={lr:.3g} elapsed={elapsed_min:.1f}m"
                )

    if ema is not None:
        ema.copy_to(policy.velocity_net.parameters())
    torch.save(policy.state_dict(), checkpoint_path)
    print(f"[train] saved {checkpoint_path}")
    return checkpoint_path


def ensure_checkpoint(
    *,
    args: argparse.Namespace,
    task: str,
    checkpoint_path: Path,
    data_dir: Path,
    device: Any,
    DatasetCls: Any,
    ConditionalUnet1D: Any,
    StreamingFlowPolicyDeterministic: Any,
    fold: int | None = None,
    heldout_demo: int | None = None,
    train_indices_override: list[int] | None = None,
) -> None:
    if not args.train_first:
        return
    if checkpoint_path.is_file() and not args.force_train:
        print(f"[train-skip] existing checkpoint: {checkpoint_path}")
        return
    train_policy_checkpoint(
        args=args,
        task=task,
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        device=device,
        DatasetCls=DatasetCls,
        ConditionalUnet1D=ConditionalUnet1D,
        StreamingFlowPolicyDeterministic=StreamingFlowPolicyDeterministic,
        fold=fold,
        heldout_demo=heldout_demo,
        train_indices_override=train_indices_override,
    )


def make_datasets(
    *,
    dataset_cls: Any,
    policy: Any,
    data_dir: Path,
    task: str,
    pred_horizon: int,
    obs_horizon: int,
    action_horizon: int,
    obs_dim: int,
    train_demos: int,
    test_demos: int,
) -> tuple[Any, Any]:
    split_config = {"train_demos": train_demos, "test_demos": test_demos}
    train_ds = dataset_cls(
        dataset_dir=str(data_dir),
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        obs_dim=obs_dim,
        task=task,
        split="train",
        split_config=split_config,
        transform_datum_fn=policy.TransformTrainingDatum,
    )
    test_ds = dataset_cls(
        dataset_dir=str(data_dir),
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        obs_dim=obs_dim,
        task=task,
        split="test",
        split_config=split_config,
        stats=train_ds.stats,
        demos_point_norm=train_ds.demos_point_norm,
    )
    return train_ds, test_ds


def rollout_policy(
    policy: Any,
    demo: dict[str, np.ndarray],
    *,
    device: torch.device,
    action_horizon: int,
    obs_horizon: int,
    rollout_factor: float,
    min_completion_frac: float,
    post_shaping: str | None,
    shaping_config: dict[str, Any] | None,
) -> dict[str, Any]:
    torch = import_torch()
    with torch.no_grad():
        gt = np.asarray(demo["action"], dtype=np.float32)
        obs_deque: collections.deque[np.ndarray] = collections.deque(
            np.asarray(demo["obs"], dtype=np.float32),
            maxlen=obs_horizon,
        )
        pred_points: list[np.ndarray] = []
        pred_vels: list[np.ndarray] = []
        step = 0
        target_steps = max(1, int(math.ceil(len(gt) * rollout_factor)))
        while step < target_steps:
            obs_seq = np.stack(obs_deque)
            nobs = torch.from_numpy(obs_seq).to(device=device, dtype=torch.float32).unsqueeze(0)
            naction, nvel = policy(
                nobs,
                num_actions=action_horizon,
                postShaping=post_shaping,
                shapingConfig=shaping_config,
            )
            naction_np = naction.detach().cpu().numpy()[0]
            nvel_np = nvel.detach().cpu().numpy()[0]
            if not pred_points:
                pred_points.extend(list(naction_np))
                pred_vels.extend(list(nvel_np))
            else:
                pred_points.extend(list(naction_np[1:]))
                pred_vels.extend(list(nvel_np[1:]))

            next_obs = np.concatenate((naction_np, nvel_np), axis=-1)
            for next_obs_t in next_obs:
                obs_deque.append(next_obs_t.astype(np.float32))
            step += action_horizon - 1

        pred = np.asarray(pred_points, dtype=np.float32)
        pred_vel = np.asarray(pred_vels, dtype=np.float32)
        dists = np.linalg.norm(pred - gt[-1], axis=1)
        cut_idx = int(np.argmin(dists))
        valid = bool(cut_idx > len(gt) * min_completion_frac)
        if valid:
            pred = pred[: cut_idx + 1]
            pred_vel = pred_vel[: cut_idx + 1]
        return {
            "trajectory": pred,
            "velocity": pred_vel,
            "steps_taken": cut_idx,
            "endpoint_dist": float(dists[cut_idx]),
            "valid": valid,
        }


def velocity_grid_for_lipschitz(
    pred_traj: np.ndarray,
    *,
    device: torch.device,
    grid_size: int,
    method: MethodConfig,
    shaping_config: dict[str, Any] | None,
    shape_velocity_batch_metric_CASF: Any,
    shape_velocity_batch_hardBarrier: Any,
    shape_velocity_batch_CBF: Any,
) -> tuple[np.ndarray, np.ndarray]:
    torch = import_torch()
    xg = torch.linspace(-1.0, 1.0, grid_size, device=device)
    yg = torch.linspace(-1.0, 1.0, grid_size, device=device)
    x_mesh, y_mesh = torch.meshgrid(xg, yg, indexing="xy")
    pos_grid = torch.stack((x_mesh, y_mesh), dim=-1).reshape(-1, 2)
    pos_np = pos_grid.detach().cpu().numpy()

    traj_t = torch.from_numpy(pred_traj).to(device=device, dtype=torch.float32)
    vel_t = torch.zeros_like(traj_t)
    vel_t[1:] = traj_t[1:] - traj_t[:-1]
    vel_t[0] = vel_t[1] if len(vel_t) > 1 else 0.0
    vel_np = vel_t.detach().cpu().numpy()
    nearest_idx = np.linalg.norm(pos_np[:, None, :] - pred_traj[None, :, :], axis=-1).argmin(axis=1)
    v_grid = torch.tensor(vel_np[nearest_idx], dtype=torch.float32, device=device)

    if method.post_shaping == "obstacle":
        assert shaping_config is not None
        v_grid = shape_velocity_batch_metric_CASF(
            pos_grid,
            v_grid,
            shaping_config["center_norm"],
            shaping_config["radius_norm"],
            alpha=shaping_config["alpha"],
            beta=shaping_config["beta"],
            w_scale=shaping_config["w_scale"],
            eps=shaping_config["eps"],
        )
    elif method.post_shaping == "obstacle-hardB":
        assert shaping_config is not None
        v_grid = shape_velocity_batch_hardBarrier(
            pos_grid,
            v_grid,
            shaping_config["center_norm"],
            shaping_config["radius_norm"],
            alpha=shaping_config["alpha"],
            beta=shaping_config["beta"],
            w_scale=shaping_config["w_scale"],
            eps=shaping_config["eps"],
        )
    elif method.post_shaping == "obstacle-cbf":
        assert shaping_config is not None
        v_grid = shape_velocity_batch_CBF(
            pos_grid,
            v_grid,
            shaping_config["center_norm"],
            shaping_config["radius_norm"],
            alpha=shaping_config["alpha"],
            beta=shaping_config["beta"],
            w_scale=shaping_config["w_scale"],
            eps=shaping_config["eps"],
        )
    return pos_np, v_grid.detach().cpu().numpy()


def lipschitz_estimate(pos_grid: np.ndarray, v_field: np.ndarray, grid_size: int) -> tuple[float, float]:
    x = pos_grid.reshape(grid_size, grid_size, 2)
    v = v_field.reshape(grid_size, grid_size, 2)
    dvx = v[1:, :, :] - v[:-1, :, :]
    dvy = v[:, 1:, :] - v[:, :-1, :]
    dxx = x[1:, :, :] - x[:-1, :, :]
    dxy = x[:, 1:, :] - x[:, :-1, :]
    lx = np.linalg.norm(dvx, axis=-1) / (np.linalg.norm(dxx, axis=-1) + 1e-8)
    ly = np.linalg.norm(dvy, axis=-1) / (np.linalg.norm(dxy, axis=-1) + 1e-8)
    all_l = np.concatenate([lx.ravel(), ly.ravel()])
    return float(np.max(all_l)), float(np.mean(all_l))


def evaluate_trajectory(
    *,
    task: str,
    fold: int | None,
    demo_idx: int,
    train_demo_indices: list[int] | None,
    method: MethodConfig,
    checkpoint_path: Path,
    gt: np.ndarray,
    pred: np.ndarray,
    raw_pred: np.ndarray,
    rollout_valid: bool,
    steps_taken: int,
    raw_steps_taken: int,
    obstacle: ObstacleConfig,
    lipschitz: tuple[float, float] | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task": task,
        "fold": "" if fold is None else fold,
        "demo_idx": demo_idx,
        "heldout_demo": demo_idx,
        "train_demo_indices": "" if train_demo_indices is None else ",".join(str(idx) for idx in train_demo_indices),
        "method": method.method,
        "method_label": method.label,
        "post_shaping": method.post_shaping or "none",
        "alpha": method.alpha,
        "beta": method.beta,
        "w_scale": method.w_scale,
        "drift_gain": method.drift_gain,
        "eps": method.eps,
        "valid": int(rollout_valid),
        "steps_taken": steps_taken,
        "steps_taken_sfp": raw_steps_taken,
        "center_x": float(obstacle.center_norm[0]),
        "center_y": float(obstacle.center_norm[1]),
        "radius": obstacle.radius_norm,
        "center_scale_x": obstacle.center_scale[0],
        "center_scale_y": obstacle.center_scale[1],
        "radius_scale": obstacle.radius_scale,
        "checkpoint": str(checkpoint_path),
    }
    if not rollout_valid:
        for metric in SUMMARY_METRICS:
            row[metric] = float("nan")
        return row

    mse, final_dist = mse_and_final_dist(pred, gt)
    collisions = collision_metrics(pred, obstacle.center_norm, obstacle.radius_norm)
    row.update(
        {
            "mse": mse,
            "final_dist": final_dist,
            "masked_frechet_vs_gt": masked_discrete_frechet(
                pred,
                gt,
                obstacle.center_norm,
                obstacle.radius_norm,
            ),
            "masked_frechet_vs_sfp": (
                float("nan")
                if method.method == "sfp"
                else masked_discrete_frechet(
                    pred,
                    raw_pred,
                    obstacle.center_norm,
                    obstacle.radius_norm,
                )
            ),
            **collisions,
            "lipschitz_max": float("nan") if lipschitz is None else lipschitz[0],
            "lipschitz_mean": float("nan") if lipschitz is None else lipschitz[1],
        }
    )
    return row


def evaluate_demo(
    *,
    args: argparse.Namespace,
    task: str,
    fold: int | None,
    demo_idx: int,
    train_demo_indices: list[int] | None,
    demo: dict[str, Any],
    policy: Any,
    methods: list[MethodConfig],
    checkpoint_path: Path,
    obstacle: ObstacleConfig,
    device: Any,
    shape_velocity_batch_metric_CASF: Any,
    shape_velocity_batch_hardBarrier: Any,
    shape_velocity_batch_CBF: Any,
) -> list[dict[str, Any]]:
    gt = np.asarray(demo["action"], dtype=np.float32)
    raw = rollout_policy(
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
    raw_pred = raw["trajectory"]

    rows: list[dict[str, Any]] = []
    for method in methods:
        if method.method == "sfp":
            pred = raw_pred
            valid = bool(raw["valid"])
            steps_taken = int(raw["steps_taken"])
            shaping_config = None
        else:
            shaping_config = obstacle.shaping_dict(method)
            shaped = rollout_policy(
                policy,
                demo,
                device=device,
                action_horizon=args.action_horizon,
                obs_horizon=args.obs_horizon,
                rollout_factor=args.rollout_factor,
                min_completion_frac=args.min_completion_frac,
                post_shaping=method.post_shaping,
                shaping_config=shaping_config,
            )
            pred = shaped["trajectory"]
            valid = bool(shaped["valid"])
            steps_taken = int(shaped["steps_taken"])

        lipschitz = None
        if args.compute_lipschitz and valid:
            pos_grid, v_grid = velocity_grid_for_lipschitz(
                pred,
                device=device,
                grid_size=args.lipschitz_grid_size,
                method=method,
                shaping_config=shaping_config,
                shape_velocity_batch_metric_CASF=shape_velocity_batch_metric_CASF,
                shape_velocity_batch_hardBarrier=shape_velocity_batch_hardBarrier,
                shape_velocity_batch_CBF=shape_velocity_batch_CBF,
            )
            lipschitz = lipschitz_estimate(pos_grid, v_grid, args.lipschitz_grid_size)

        row = evaluate_trajectory(
            task=task,
            fold=fold,
            demo_idx=demo_idx,
            train_demo_indices=train_demo_indices,
            method=method,
            checkpoint_path=checkpoint_path,
            gt=gt,
            pred=pred,
            raw_pred=raw_pred,
            rollout_valid=valid,
            steps_taken=steps_taken,
            raw_steps_taken=int(raw["steps_taken"]),
            obstacle=obstacle,
            lipschitz=lipschitz,
        )
        rows.append(row)
        print(
            "[row] "
            f"{task} fold={fold if fold is not None else 'last'} demo={demo_idx} method={method.label} "
            f"alpha={format_float(method.alpha, 3)} w={format_float(method.w_scale, 3)} "
            f"MPD={format_float(row.get('max_pen_depth', float('nan')), 4)} "
            f"IntV={format_float(row.get('int_violation', float('nan')), 4)}"
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        for row in rows:
            writer.writerow(row)


def t_critical_95(n: int) -> float:
    if n <= 1:
        return float("nan")
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, n - 1))
    except Exception:
        return 1.96


def summarize_values(values: Iterable[float]) -> dict[str, float | int]:
    arr = np.array([finite_or_nan(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "ci95_half_width": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    mean = float(np.mean(arr))
    if n == 1:
        std = sem = half = low = high = float("nan")
    else:
        std = float(np.std(arr, ddof=1))
        sem = float(std / math.sqrt(n))
        half = float(t_critical_95(n) * sem)
        low = mean - half
        high = mean + half
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_half_width": half,
        "ci95_low": low,
        "ci95_high": high,
    }


def group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        grouped.setdefault(key, []).append(row)
    return grouped


def summarize_by_config(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key_fields = ("task", "method", "method_label", "post_shaping", "alpha", "beta", "w_scale", "drift_gain", "eps")
    summary_rows: list[dict[str, Any]] = []
    for key, group in sorted(group_rows(rows, key_fields).items()):
        out = {field: value for field, value in zip(key_fields, key)}
        out["num_rows"] = len(group)
        out["num_valid"] = int(sum(int(row.get("valid", 0)) for row in group))
        for metric in SUMMARY_METRICS:
            stats = summarize_values(row.get(metric) for row in group if int(row.get("valid", 0)))
            for stat_name, stat_value in stats.items():
                out[f"{metric}_{stat_name}"] = stat_value
        summary_rows.append(out)
    return summary_rows


def sort_value(row: dict[str, Any], metric: str) -> float:
    value = finite_or_nan(row.get(f"{metric}_mean"))
    return value if math.isfinite(value) else float("inf")


def select_table_configs(summary_rows: list[dict[str, Any]], selection_metrics: list[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for key, group in sorted(group_rows(summary_rows, ("task", "method")).items()):
        task, method = key
        if method == "sfp":
            candidates = group
        else:
            candidates = [row for row in group if int(row.get("num_valid", 0)) > 0]
        if not candidates:
            continue
        best = sorted(candidates, key=lambda row: tuple(sort_value(row, metric) for metric in selection_metrics))[0]
        selected.append(best)
    return selected


def make_table_rows(selected_configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in selected_configs:
        for table_metric, metric_key in TABLE_METRICS.items():
            out = {
                "task": config["task"],
                "method": config["method"],
                "method_label": config["method_label"],
                "metric": table_metric,
                "metric_key": metric_key,
                "alpha": config["alpha"],
                "beta": config["beta"],
                "w_scale": config["w_scale"],
                "drift_gain": config["drift_gain"],
                "eps": config["eps"],
                "n": config.get(f"{metric_key}_n"),
                "mean": config.get(f"{metric_key}_mean"),
                "std": config.get(f"{metric_key}_std"),
                "sem": config.get(f"{metric_key}_sem"),
                "ci95_half_width": config.get(f"{metric_key}_ci95_half_width"),
                "ci95_low": config.get(f"{metric_key}_ci95_low"),
                "ci95_high": config.get(f"{metric_key}_ci95_high"),
            }
            if config["method"] == "sfp" and metric_key == "masked_frechet_vs_sfp":
                out.update(
                    {
                        "n": 0,
                        "mean": float("nan"),
                        "std": float("nan"),
                        "sem": float("nan"),
                        "ci95_half_width": float("nan"),
                        "ci95_low": float("nan"),
                        "ci95_high": float("nan"),
                    }
                )
            rows.append(out)
    return rows


def write_wide_tables(
    out_dir: Path,
    table_rows: list[dict[str, Any]],
    *,
    tasks: list[str],
    digits: int,
) -> None:
    row_lookup = {
        (row["method"], row["metric"], row["task"]): row
        for row in table_rows
    }
    method_order = [method for method in DEFAULT_METHODS if method in {row["method"] for row in table_rows}]
    metric_order = list(TABLE_METRICS.keys())

    variants = {
        "mean": lambda row: format_float(finite_or_nan(row.get("mean")), digits),
        "mean_std": lambda row: (
            "NA"
            if not math.isfinite(finite_or_nan(row.get("mean")))
            else f"{format_float(finite_or_nan(row.get('mean')), digits)} +/- {format_float(finite_or_nan(row.get('std')), digits)}"
        ),
        "mean_ci95": lambda row: (
            "NA"
            if not math.isfinite(finite_or_nan(row.get("mean")))
            else (
                f"{format_float(finite_or_nan(row.get('mean')), digits)} "
                f"[{format_float(finite_or_nan(row.get('ci95_low')), digits)}, "
                f"{format_float(finite_or_nan(row.get('ci95_high')), digits)}]"
            )
        ),
    }
    for variant, formatter in variants.items():
        path = out_dir / f"lasa_table_wide_{variant}.csv"
        with path.open("w", newline="") as fh:
            fieldnames = ["method", "metric", *tasks]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for method in method_order:
                for metric in metric_order:
                    row_out = {"method": TABLE_METHOD_LABELS[method], "metric": metric}
                    for task in tasks:
                        row = row_lookup.get((method, metric, task))
                        row_out[task] = "NA" if row is None else formatter(row)
                    writer.writerow(row_out)


def csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def dry_run(args: argparse.Namespace) -> int:
    tasks = parse_csv_list(args.tasks)
    data_dir = Path(args.data_dir).expanduser().resolve()
    print(f"Data dir: {Path(args.data_dir).expanduser().resolve()}")
    print(f"Output dir: {Path(args.out_dir).expanduser().resolve()}")
    print(f"Fold mode: {args.fold_mode}")
    print(f"Train first: {args.train_first} force_train={args.force_train}")
    if args.train_first:
        print(
            "Training: "
            f"epochs={args.train_epochs} batch={args.train_batch_size} lr={args.train_lr} "
            f"sigma={args.train_sigma} workers={args.train_num_workers}"
        )
    print("Tasks:")
    if args.fold_mode == "loo":
        folds = parse_int_selection(args.folds, max_count=args.num_demos)
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
                status = "OK" if ckpt_path.is_file() else ("TRAIN" if args.train_first else "MISSING")
                print(f"    fold={fold} heldout={fold}: checkpoint={status} {ckpt_path}")
    elif args.fold_mode == "train":
        for task in tasks:
            mat_path = data_dir / f"{task}.mat"
            ckpt_path = checkpoint_path_for_task(
                args.checkpoint_template,
                task,
                train_demos=args.num_demos,
                test_demos=0,
            )
            status = "OK" if ckpt_path.is_file() else ("TRAIN" if args.train_first else "MISSING")
            print(
                f"  {task:10s} data={'OK' if mat_path.is_file() else 'MISSING'} "
                f"train/eval_demos=0-{args.num_demos - 1} checkpoint={status} {ckpt_path}"
            )
    else:
        for task in tasks:
            mat_path = data_dir / f"{task}.mat"
            ckpt_path = checkpoint_path_for_task(
                args.checkpoint_template,
                task,
                train_demos=args.train_demos,
                test_demos=args.test_demos,
            )
            status = "OK" if ckpt_path.is_file() else ("TRAIN" if args.train_first else "MISSING")
            print(
                f"  {task:10s} data={'OK' if mat_path.is_file() else 'MISSING'} "
                f"checkpoint={status}"
            )
    print(f"Methods/configs: {len(build_method_configs(args))}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing LASA .mat files.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for CSV outputs.")
    parser.add_argument(
        "--checkpoint-template",
        default=str(DEFAULT_CHECKPOINT_TEMPLATE),
        help=(
            "Checkpoint path template. Supported fields: {task}, {fold}, "
            "{heldout_demo}/{heldout}/{demo}, {train_demos}, {test_demos}."
        ),
    )
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="Comma-separated LASA task names.")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="Comma-separated methods: sfp,projection,cbf,casf.")
    parser.add_argument("--device", default=default_device(), help="Torch device.")
    parser.add_argument("--skip-missing-checkpoints", action="store_true", help="Skip tasks with missing checkpoints instead of failing.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned inputs/configs and exit before importing legacy modules.")
    parser.add_argument("--train-first", action="store_true", help="Train missing LASA SFP checkpoints before evaluation.")
    parser.add_argument("--force-train", action="store_true", help="Retrain checkpoints even if the target checkpoint path already exists.")
    parser.add_argument(
        "--fold-mode",
        choices=("last", "loo", "train"),
        default="last",
        help=(
            "last keeps the legacy first-N train/final-M test split; loo evaluates one held-out LASA demo per fold; "
            "train trains on all selected demos and evaluates those same training demos."
        ),
    )
    parser.add_argument("--num-demos", type=int, default=7, help="Number of demos per LASA task for --fold-mode loo.")
    parser.add_argument("--folds", default="all", help="Fold ids for --fold-mode loo, e.g. all, 0-6, or 0,3,6.")

    parser.add_argument("--pred-horizon", type=int, default=16)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--obs-dim", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--train-demos", type=int, default=6, help="Number of LASA demos used for normalization stats.")
    parser.add_argument("--test-demos", type=int, default=1, help="Number of final LASA demos evaluated.")
    parser.add_argument("--train-epochs", type=int, default=1000, help="Training epochs used with --train-first.")
    parser.add_argument("--train-batch-size", type=int, default=1024, help="Training batch size used with --train-first.")
    parser.add_argument("--train-lr", type=float, default=1e-4, help="Training learning rate used with --train-first.")
    parser.add_argument("--train-sigma", type=float, default=0.1, help="SFP training noise sigma used with --train-first.")
    parser.add_argument("--train-weight-decay", type=float, default=1e-6, help="Training AdamW weight decay.")
    parser.add_argument("--train-warmup-steps", type=int, default=500, help="Training LR warmup steps.")
    parser.add_argument("--train-ema-decay", type=float, default=0.999, help="EMA decay for inference weights; set <=0 to disable.")
    parser.add_argument("--train-num-workers", type=int, default=1, help="Training dataloader workers.")
    parser.add_argument("--train-log-every", type=int, default=50, help="Print training loss every N epochs.")
    parser.add_argument("--rollout-factor", type=float, default=1.2, help="Rollout budget as a multiplier of GT length.")
    parser.add_argument("--min-completion-frac", type=float, default=0.9, help="Reject rollout if endpoint is reached before this GT fraction.")

    parser.add_argument("--radius-scale", type=float, default=0.15)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--cbf-alphas", default="0.5,1,5,10,20,50", help="Comma-separated CBF alpha sweep.")
    parser.add_argument("--casf-alphas", default="1,5,10,20,30,50", help="Comma-separated CASF alpha sweep.")
    parser.add_argument("--casf-beta", type=float, default=0.0)
    parser.add_argument("--casf-w-scales", default="10,20,30,40,50", help="Comma-separated CASF w_scale sweep.")
    parser.add_argument("--casf-drift-gains", default="0.02,0.1", help="Comma-separated CASF drift_gain sweep.")
    parser.add_argument(
        "--selection-metrics",
        default="max_pen_depth,int_violation,masked_frechet_vs_sfp",
        help="Comma-separated summary metrics used lexicographically to pick one config for the table.",
    )
    parser.add_argument("--digits", type=int, default=3, help="Digits for wide formatted tables.")

    parser.add_argument("--compute-lipschitz", action="store_true", help="Also estimate velocity-field Lipschitz metrics.")
    parser.add_argument("--lipschitz-grid-size", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        return dry_run(args)

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    tasks = parse_csv_list(args.tasks)
    methods = build_method_configs(args)
    torch = import_torch()
    device = torch.device(args.device)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"LASA data directory not found: {data_dir}")
    if args.train_first and args.fold_mode == "loo" and not checkpoint_template_has_fold_field(args.checkpoint_template):
        raise ValueError(
            "--train-first with --fold-mode loo needs a fold-specific --checkpoint-template, "
            "for example: "
            f"{PROJECT_ROOT}/models/"
            "CASF_lasaTask_ah8_{task}_fold{fold}_sfpdObs_1000ep_lr0.0001_obsDim4_demo6-1_norm.pth"
        )

    (
        ConditionalUnet1D,
        DatasetCls,
        StreamingFlowPolicyDeterministic,
        shape_velocity_batch_metric_CASF,
        shape_velocity_batch_hardBarrier,
        shape_velocity_batch_CBF,
    ) = import_legacy_modules()

    all_rows: list[dict[str, Any]] = []
    trials_path = out_dir / "lasa_trials.csv"
    write_csv(trials_path, [], list(TRIAL_FIELDNAMES))
    print(f"[live] writing trial rows to {trials_path}")
    for task in tasks:
        mat_path = data_dir / f"{task}.mat"
        if not mat_path.is_file():
            raise FileNotFoundError(f"Missing LASA task file: {mat_path}")
        obstacle = make_obstacle(task, args.radius_scale)

        if args.fold_mode == "loo":
            folds = parse_int_selection(args.folds, max_count=args.num_demos)
            for fold in folds:
                heldout_demo = fold
                ckpt_path = checkpoint_path_for_task(
                    args.checkpoint_template,
                    task,
                    fold=fold,
                    heldout_demo=heldout_demo,
                    train_demos=args.num_demos - 1,
                    test_demos=1,
                )
                ensure_checkpoint(
                    args=args,
                    task=task,
                    checkpoint_path=ckpt_path,
                    data_dir=data_dir,
                    device=device,
                    DatasetCls=DatasetCls,
                    ConditionalUnet1D=ConditionalUnet1D,
                    StreamingFlowPolicyDeterministic=StreamingFlowPolicyDeterministic,
                    fold=fold,
                    heldout_demo=heldout_demo,
                )
                if not ckpt_path.is_file():
                    message = f"Missing checkpoint for {task} fold {fold}: {ckpt_path}"
                    if args.skip_missing_checkpoints:
                        print(f"[skip] {message}")
                        continue
                    raise FileNotFoundError(message)

                print(f"[task] {task} fold={fold} heldout_demo={heldout_demo}: loading {ckpt_path}")
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
                train_demo_indices, test_ds = make_loo_test_dataset(
                    data_dir=data_dir,
                    task=task,
                    heldout_demo=heldout_demo,
                    num_demos=args.num_demos,
                    pred_horizon=args.pred_horizon,
                    obs_horizon=args.obs_horizon,
                    obs_dim=args.obs_dim,
                )
                for demo in test_ds:
                    rows = evaluate_demo(
                        args=args,
                        task=task,
                        fold=fold,
                        demo_idx=int(demo["demo_idx"]),
                        train_demo_indices=train_demo_indices,
                        demo=demo,
                        policy=policy,
                        methods=methods,
                        checkpoint_path=ckpt_path,
                        obstacle=obstacle,
                        device=device,
                        shape_velocity_batch_metric_CASF=shape_velocity_batch_metric_CASF,
                        shape_velocity_batch_hardBarrier=shape_velocity_batch_hardBarrier,
                        shape_velocity_batch_CBF=shape_velocity_batch_CBF,
                    )
                    all_rows.extend(rows)
                    append_csv_rows(trials_path, rows, list(TRIAL_FIELDNAMES))
            continue

        if args.fold_mode == "train":
            train_demo_indices = list(range(args.num_demos))
            ckpt_path = checkpoint_path_for_task(
                args.checkpoint_template,
                task,
                train_demos=args.num_demos,
                test_demos=0,
            )
            ensure_checkpoint(
                args=args,
                task=task,
                checkpoint_path=ckpt_path,
                data_dir=data_dir,
                device=device,
                DatasetCls=DatasetCls,
                ConditionalUnet1D=ConditionalUnet1D,
                StreamingFlowPolicyDeterministic=StreamingFlowPolicyDeterministic,
                train_indices_override=train_demo_indices,
            )
            if not ckpt_path.is_file():
                message = f"Missing train-set checkpoint for {task}: {ckpt_path}"
                if args.skip_missing_checkpoints:
                    print(f"[skip] {message}")
                    continue
                raise FileNotFoundError(message)

            print(f"[task] {task} train-set eval demos={train_demo_indices}: loading {ckpt_path}")
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
            train_demo_indices, test_ds = make_train_eval_dataset(
                data_dir=data_dir,
                task=task,
                num_demos=args.num_demos,
                pred_horizon=args.pred_horizon,
                obs_horizon=args.obs_horizon,
                obs_dim=args.obs_dim,
            )
            for demo in test_ds:
                rows = evaluate_demo(
                    args=args,
                    task=task,
                    fold=None,
                    demo_idx=int(demo["demo_idx"]),
                    train_demo_indices=train_demo_indices,
                    demo=demo,
                    policy=policy,
                    methods=methods,
                    checkpoint_path=ckpt_path,
                    obstacle=obstacle,
                    device=device,
                    shape_velocity_batch_metric_CASF=shape_velocity_batch_metric_CASF,
                    shape_velocity_batch_hardBarrier=shape_velocity_batch_hardBarrier,
                    shape_velocity_batch_CBF=shape_velocity_batch_CBF,
                )
                all_rows.extend(rows)
                append_csv_rows(trials_path, rows, list(TRIAL_FIELDNAMES))
            continue

        ckpt_path = checkpoint_path_for_task(
            args.checkpoint_template,
            task,
            train_demos=args.train_demos,
            test_demos=args.test_demos,
        )
        ensure_checkpoint(
            args=args,
            task=task,
            checkpoint_path=ckpt_path,
            data_dir=data_dir,
            device=device,
            DatasetCls=DatasetCls,
            ConditionalUnet1D=ConditionalUnet1D,
            StreamingFlowPolicyDeterministic=StreamingFlowPolicyDeterministic,
        )
        if not ckpt_path.is_file():
            message = f"Missing checkpoint for {task}: {ckpt_path}"
            if args.skip_missing_checkpoints:
                print(f"[skip] {message}")
                continue
            raise FileNotFoundError(message)

        print(f"[task] {task}: loading {ckpt_path}")
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
        _, test_ds = make_datasets(
            dataset_cls=DatasetCls,
            policy=policy,
            data_dir=data_dir,
            task=task,
            pred_horizon=args.pred_horizon,
            obs_horizon=args.obs_horizon,
            action_horizon=args.action_horizon,
            obs_dim=args.obs_dim,
            train_demos=args.train_demos,
            test_demos=args.test_demos,
        )

        for local_demo_idx, demo in enumerate(test_ds):
            # LASA legacy test split uses the final `test_demos` demos.
            demo_idx = args.train_demos + local_demo_idx
            rows = evaluate_demo(
                args=args,
                task=task,
                fold=None,
                demo_idx=demo_idx,
                train_demo_indices=list(range(args.train_demos)),
                demo=demo,
                policy=policy,
                methods=methods,
                checkpoint_path=ckpt_path,
                obstacle=obstacle,
                device=device,
                shape_velocity_batch_metric_CASF=shape_velocity_batch_metric_CASF,
                shape_velocity_batch_hardBarrier=shape_velocity_batch_hardBarrier,
                shape_velocity_batch_CBF=shape_velocity_batch_CBF,
            )
            all_rows.extend(rows)
            append_csv_rows(trials_path, rows, list(TRIAL_FIELDNAMES))

    if not all_rows:
        raise RuntimeError("No rows were evaluated. Check tasks, checkpoint paths, and --skip-missing-checkpoints.")

    summary_rows = summarize_by_config(all_rows)
    summary_path = out_dir / "lasa_summary_by_config.csv"
    write_csv(summary_path, summary_rows, csv_fieldnames(summary_rows))

    selected_configs = select_table_configs(summary_rows, parse_csv_list(args.selection_metrics))
    table_rows = make_table_rows(selected_configs)
    table_path = out_dir / "lasa_table_selected.csv"
    write_csv(table_path, table_rows, csv_fieldnames(table_rows))
    write_wide_tables(out_dir, table_rows, tasks=tasks, digits=args.digits)

    print(f"[done] wrote {trials_path}")
    print(f"[done] wrote {summary_path}")
    print(f"[done] wrote {table_path}")
    print(f"[done] wrote wide tables in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
