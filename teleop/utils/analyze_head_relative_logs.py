#!/usr/bin/env python3
"""Analyze head-relative teleoperation monitor logs.

By default this script finds the newest ``head_relative.csv`` below the
standard log directory, compares it with the previous compatible run, prints a
short report, and writes ``analysis_summary.json`` plus ``analysis.png`` into
an ``analysis`` subdirectory beside the source CSV.

Examples:
    python utils/analyze_head_relative_logs.py
    python utils/analyze_head_relative_logs.py path/to/run_dir
    python utils/analyze_head_relative_logs.py latest --baseline none
    python utils/analyze_head_relative_logs.py latest --baseline path/to/older/run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_LOGS_DIR = Path(__file__).resolve().with_name("head_relative_logs")
ARMS = ("left", "right")
AXES = ("x", "y", "z")
COMPATIBILITY_KEYS = ("arm", "ee", "input_mode", "arm_reference_mode", "sim")

CSV_COLUMNS = ["seq", "t_wall", "t_rel", "ik_ok"]
for _arm in ARMS:
    CSV_COLUMNS += [f"{_arm}_human_d{axis}" for axis in AXES]
    CSV_COLUMNS += [f"{_arm}_human_distance"]
    CSV_COLUMNS += [f"{_arm}_robot_d{axis}" for axis in AXES]
    CSV_COLUMNS += [f"{_arm}_robot_distance"]
    CSV_COLUMNS += [f"{_arm}_error_d{axis}" for axis in AXES]
    CSV_COLUMNS += [f"{_arm}_error_distance"]


def _finite_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _percentile(values, percentile):
    return _finite_float(np.percentile(values, percentile))


def _xyz(data, arm, kind):
    return np.column_stack([data[f"{arm}_{kind}_d{axis}"] for axis in AXES])


def _sample_speed(data, arm):
    position = _xyz(data, arm, "human")
    speed = np.zeros(len(data), dtype=float)
    dt = np.diff(data["t_rel"])
    valid = dt > 0.0
    differences = np.linalg.norm(np.diff(position, axis=0), axis=1)
    increments = speed[1:]
    increments[valid] = differences[valid] / dt[valid]
    increments[~valid] = np.nan
    return speed


def resolve_csv(value, logs_dir=DEFAULT_LOGS_DIR):
    """Resolve ``latest``, a run directory, or a CSV path to a source CSV."""
    logs_dir = Path(logs_dir).expanduser().resolve()
    if value in (None, "latest"):
        candidates = list(logs_dir.glob("**/head_relative.csv"))
        if not candidates:
            raise FileNotFoundError(f"No head_relative.csv files found below {logs_dir}")
        return max(candidates, key=lambda path: path.stat().st_mtime).resolve()

    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "head_relative.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Head-relative CSV not found: {path}")
    return path


def load_metadata(csv_path):
    meta_path = Path(csv_path).with_name("run_meta.json")
    if not meta_path.is_file():
        return {}, None
    try:
        with meta_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read metadata {meta_path}: {exc}") from exc
    return metadata, meta_path


def load_csv(csv_path):
    """Load and validate the stable monitor CSV schema."""
    csv_path = Path(csv_path)
    try:
        data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read {csv_path}: {exc}") from exc
    data = np.atleast_1d(data)
    if data.dtype.names is None:
        raise ValueError(f"CSV has no named header: {csv_path}")
    missing = [column for column in CSV_COLUMNS if column not in data.dtype.names]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    if len(data) < 2:
        raise ValueError("At least two samples are required for timing and error analysis")
    nonfinite = sum(int((~np.isfinite(data[name])).sum()) for name in data.dtype.names)
    if nonfinite:
        raise ValueError(f"CSV contains {nonfinite} non-finite numeric cells")
    duration = float(data["t_rel"][-1] - data["t_rel"][0])
    if duration <= 0.0:
        raise ValueError("CSV elapsed time must increase from the first sample to the last")
    return data


def compatible_metadata(candidate, target):
    """Return true when two runs describe the same teleoperation configuration."""
    if not target:
        return True
    return all(candidate.get(key) == target.get(key) for key in COMPATIBILITY_KEYS)


def find_previous_compatible(csv_path, logs_dir, target_metadata):
    """Find the newest older run with matching robot/input/reference metadata."""
    csv_path = Path(csv_path).resolve()
    target_mtime = csv_path.stat().st_mtime
    candidates = sorted(
        Path(logs_dir).expanduser().resolve().glob("**/head_relative.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate == csv_path or candidate.stat().st_mtime >= target_mtime:
            continue
        candidate_metadata, _ = load_metadata(candidate)
        if compatible_metadata(candidate_metadata, target_metadata):
            return candidate
    return None


def integrity_metrics(data):
    dt = np.diff(data["t_rel"])
    seq_delta = np.diff(data["seq"])
    nonfinite = sum(int((~np.isfinite(data[name])).sum()) for name in data.dtype.names)
    consistency = {}
    for arm in ARMS:
        human = _xyz(data, arm, "human")
        robot = _xyz(data, arm, "robot")
        recorded_error = _xyz(data, arm, "error")
        calculated_error = robot - human
        vector_difference = np.linalg.norm(recorded_error - calculated_error, axis=1)
        recorded_distance = data[f"{arm}_error_distance"]
        calculated_distance = np.linalg.norm(calculated_error, axis=1)
        consistency[arm] = {
            "max_error_vector_difference_m": _finite_float(vector_difference.max()),
            "max_error_distance_difference_m": _finite_float(
                np.abs(recorded_distance - calculated_distance).max()
            ),
        }
    return {
        "rows": int(len(data)),
        "duration_s": _finite_float(data["t_rel"][-1] - data["t_rel"][0]),
        "effective_rate_hz": _finite_float((len(data) - 1) / (data["t_rel"][-1] - data["t_rel"][0])),
        "dt_median_ms": _finite_float(np.median(dt) * 1000.0),
        "dt_p95_ms": _percentile(dt * 1000.0, 95),
        "dt_max_ms": _finite_float(dt.max() * 1000.0),
        "nonpositive_time_steps": int((dt <= 0.0).sum()),
        "missing_sequences": int(np.maximum(seq_delta - 1.0, 0.0).sum()),
        "duplicate_or_reversed_sequences": int((seq_delta <= 0.0).sum()),
        "ik_failures": int((data["ik_ok"] == 0.0).sum()),
        "nonfinite_cells": int(nonfinite),
        "schema_columns": list(data.dtype.names),
        "error_consistency": consistency,
    }


def _correlation(left, right):
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return _finite_float(np.corrcoef(left, right)[0, 1])


def estimate_lag(data, arm, mask, displacement_samples=5, max_lag_samples=10):
    human = _xyz(data, arm, "human")[mask]
    robot = _xyz(data, arm, "robot")[mask]
    if len(human) <= displacement_samples + max_lag_samples + 2:
        return {"samples": None, "milliseconds": None, "correlation": None}
    human_displacement = human[displacement_samples:] - human[:-displacement_samples]
    robot_displacement = robot[displacement_samples:] - robot[:-displacement_samples]
    scores = []
    for lag in range(max_lag_samples + 1):
        if lag:
            current_human = human_displacement[:-lag]
            current_robot = robot_displacement[lag:]
        else:
            current_human = human_displacement
            current_robot = robot_displacement
        moving = (
            (np.linalg.norm(current_human, axis=1) > 1e-4)
            & (np.linalg.norm(current_robot, axis=1) > 1e-4)
        )
        score = _correlation(current_human[moving].ravel(), current_robot[moving].ravel())
        scores.append(float("-inf") if score is None else score)
    lag = int(np.argmax(scores))
    median_dt = float(np.median(np.diff(data["t_rel"])))
    return {
        "samples": lag,
        "milliseconds": _finite_float(lag * median_dt * 1000.0),
        "correlation": _finite_float(scores[lag]),
    }


def _contiguous_runs(mask):
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    boundaries = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, boundaries + 1]
    ends = np.r_[boundaries, len(indices) - 1]
    return [(int(indices[start]), int(indices[end])) for start, end in zip(starts, ends)]


def _band_metrics(error, selected):
    if not np.any(selected):
        return None
    values = error[selected]
    return {
        "samples": int(len(values)),
        "mean_error_cm": _finite_float(values.mean() * 100.0),
        "median_error_cm": _finite_float(np.median(values) * 100.0),
        "p95_error_cm": _percentile(values * 100.0, 95),
    }


def arm_metrics(
    data,
    arm,
    startup_seconds=3.0,
    error_threshold_cm=5.0,
    near_x_cm=25.0,
    reach_x_cm=35.0,
    low_z_cm=-40.0,
    max_lag_samples=10,
):
    steady = data["t_rel"] >= startup_seconds
    if not np.any(steady):
        raise ValueError(f"No samples remain after the {startup_seconds:g} s startup exclusion")

    human = _xyz(data, arm, "human")
    robot = _xyz(data, arm, "robot")
    error_xyz = robot - human
    error = np.linalg.norm(error_xyz, axis=1)
    human_steady = human[steady]
    robot_steady = robot[steady]
    error_xyz_steady = error_xyz[steady]
    error_steady = error[steady]
    threshold_m = error_threshold_cm / 100.0
    near_x_m = near_x_cm / 100.0
    reach_x_m = reach_x_cm / 100.0
    low_z_m = low_z_cm / 100.0

    bands = {
        f"x_at_or_below_{near_x_cm:g}_cm": _band_metrics(
            error_steady, human_steady[:, 0] <= near_x_m
        ),
        f"x_{near_x_cm:g}_to_{reach_x_cm:g}_cm": _band_metrics(
            error_steady,
            (human_steady[:, 0] > near_x_m) & (human_steady[:, 0] <= reach_x_m),
        ),
        f"x_above_{reach_x_cm:g}_cm": _band_metrics(
            error_steady, human_steady[:, 0] > reach_x_m
        ),
        f"z_below_{low_z_cm:g}_cm": _band_metrics(
            error_steady, human_steady[:, 2] < low_z_m
        ),
    }

    lag = estimate_lag(data, arm, steady, max_lag_samples=max_lag_samples)
    lag_samples = lag["samples"] or 0
    if lag_samples:
        aligned = data["t_rel"][:-lag_samples] >= startup_seconds
        lag_corrected = np.linalg.norm(
            robot[lag_samples:] - human[:-lag_samples], axis=1
        )[aligned]
        lag_uncorrected = error[:-lag_samples][aligned]
    else:
        lag_corrected = error_steady
        lag_uncorrected = error_steady

    sample_speed = _sample_speed(data, arm)
    dt = float(np.median(np.diff(data["t_rel"])))
    excursions = []
    for start, end in _contiguous_runs(steady & (error > threshold_m)):
        peak = start + int(np.argmax(error[start:end + 1]))
        excursions.append({
            "start_s": _finite_float(data["t_rel"][start]),
            "end_s": _finite_float(data["t_rel"][end]),
            "duration_s": _finite_float(data["t_rel"][end] - data["t_rel"][start] + dt),
            "peak_s": _finite_float(data["t_rel"][peak]),
            "peak_error_cm": _finite_float(error[peak] * 100.0),
            "peak_error_xyz_cm": (error_xyz[peak] * 100.0).tolist(),
            "human_xyz_cm": (human[peak] * 100.0).tolist(),
            "robot_xyz_cm": (robot[peak] * 100.0).tolist(),
            "human_speed_cm_s": _finite_float(sample_speed[peak] * 100.0),
        })
    excursions.sort(key=lambda item: item["peak_error_cm"], reverse=True)

    low_motion = steady & np.isfinite(sample_speed) & (sample_speed <= 0.20)
    fast_motion = steady & np.isfinite(sample_speed) & (sample_speed > 0.20)
    result = {
        "samples": int(steady.sum()),
        "mean_error_cm": _finite_float(error_steady.mean() * 100.0),
        "median_error_cm": _finite_float(np.median(error_steady) * 100.0),
        "p95_error_cm": _percentile(error_steady * 100.0, 95),
        "p99_error_cm": _percentile(error_steady * 100.0, 99),
        "max_error_cm": _finite_float(error_steady.max() * 100.0),
        f"over_{error_threshold_cm:g}_cm_percent": _finite_float(
            (error_steady > threshold_m).mean() * 100.0
        ),
        "over_10_cm_percent": _finite_float((error_steady > 0.10).mean() * 100.0),
        "bias_xyz_cm": (error_xyz_steady.mean(axis=0) * 100.0).tolist(),
        "mean_absolute_error_xyz_cm": (
            np.abs(error_xyz_steady).mean(axis=0) * 100.0
        ).tolist(),
        "radial_bias_cm": _finite_float(
            (
                np.linalg.norm(robot_steady, axis=1)
                - np.linalg.norm(human_steady, axis=1)
            ).mean()
            * 100.0
        ),
        "correlation_xyz": [
            _correlation(human_steady[:, index], robot_steady[:, index])
            for index in range(3)
        ],
        "human_min_xyz_cm": (human_steady.min(axis=0) * 100.0).tolist(),
        "human_max_xyz_cm": (human_steady.max(axis=0) * 100.0).tolist(),
        "robot_min_xyz_cm": (robot_steady.min(axis=0) * 100.0).tolist(),
        "robot_max_xyz_cm": (robot_steady.max(axis=0) * 100.0).tolist(),
        "low_motion": _band_metrics(error, low_motion),
        "fast_motion": _band_metrics(error, fast_motion),
        "workspace_bands": {key: value for key, value in bands.items() if value},
        "lag_estimate": lag,
        "lag_corrected_mean_error_cm": _finite_float(lag_corrected.mean() * 100.0),
        "lag_correction_change_percent": (
            _finite_float((lag_corrected.mean() / lag_uncorrected.mean() - 1.0) * 100.0)
            if lag_uncorrected.mean() != 0.0
            else None
        ),
        "largest_excursions": excursions[:10],
    }
    return result


def analyze_session(data, options):
    return {
        "integrity": integrity_metrics(data),
        "steady_state": {
            arm: arm_metrics(
                data,
                arm,
                startup_seconds=options.startup_seconds,
                error_threshold_cm=options.error_threshold_cm,
                near_x_cm=options.near_x_cm,
                reach_x_cm=options.reach_x_cm,
                low_z_cm=options.low_z_cm,
                max_lag_samples=options.max_lag_samples,
            )
            for arm in ARMS
        },
    }


def _nearest_neighbors(reference, query, count):
    """Return nearest-neighbor distances/indices, with a NumPy fallback."""
    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(reference).query(query, k=count)
        if np.ndim(distances) == 1:
            distances = np.asarray(distances)[:, None]
            indices = np.asarray(indices)[:, None]
        return distances, indices
    except ImportError:
        all_distances = []
        all_indices = []
        for start in range(0, len(query), 256):
            block = query[start:start + 256]
            squared = np.sum((block[:, None, :] - reference[None, :, :]) ** 2, axis=2)
            indices = np.argpartition(squared, count - 1, axis=1)[:, :count]
            distances = np.sqrt(np.take_along_axis(squared, indices, axis=1))
            order = np.argsort(distances, axis=1)
            all_indices.append(np.take_along_axis(indices, order, axis=1))
            all_distances.append(np.take_along_axis(distances, order, axis=1))
        return np.vstack(all_distances), np.vstack(all_indices)


def target_matched_comparison(reference, target, arm, options):
    """Compare low-motion samples at nearby human wrist positions.

    This controls for xyz target location, but not wrist orientation because the
    monitor CSV intentionally stores translation only.
    """
    speed_limit_m_s = options.match_max_speed_cm_s / 100.0
    reference_speed = _sample_speed(reference, arm)
    target_speed = _sample_speed(target, arm)
    reference_mask = (
        (reference["t_rel"] >= options.startup_seconds)
        & np.isfinite(reference_speed)
        & (reference_speed <= speed_limit_m_s)
    )
    target_mask = (
        (target["t_rel"] >= options.startup_seconds)
        & np.isfinite(target_speed)
        & (target_speed <= speed_limit_m_s)
    )
    reference_position = _xyz(reference, arm, "human")[reference_mask]
    target_position = _xyz(target, arm, "human")[target_mask]
    if len(reference_position) == 0 or len(target_position) == 0:
        return {"available": False, "reason": "No low-motion samples available"}

    reference_error = reference[f"{arm}_error_distance"][reference_mask]
    target_error = target[f"{arm}_error_distance"][target_mask]
    neighbor_count = min(options.match_neighbors, len(reference_position))
    distances, indices = _nearest_neighbors(
        reference_position, target_position, neighbor_count
    )
    matched = distances[:, 0] <= options.match_radius_cm / 100.0
    if not np.any(matched):
        return {
            "available": False,
            "reason": f"No target samples matched within {options.match_radius_cm:g} cm",
            "target_low_motion_samples": int(len(target_position)),
        }
    expected_reference_error = np.mean(reference_error[indices], axis=1)
    change = target_error[matched] - expected_reference_error[matched]
    return {
        "available": True,
        "method": "nearest human xyz among low-motion samples; wrist orientation is not matched",
        "target_low_motion_samples": int(len(target_position)),
        "matched_samples": int(matched.sum()),
        "coverage_percent": _finite_float(matched.mean() * 100.0),
        "target_mean_error_cm": _finite_float(target_error[matched].mean() * 100.0),
        "reference_mean_error_cm": _finite_float(
            expected_reference_error[matched].mean() * 100.0
        ),
        "mean_change_cm": _finite_float(change.mean() * 100.0),
        "relative_change_percent": _finite_float(
            change.mean() / expected_reference_error[matched].mean() * 100.0
        ),
        "target_samples_better_percent": _finite_float((change < 0.0).mean() * 100.0),
        "nearest_distance_median_cm": _finite_float(
            np.median(distances[matched, 0]) * 100.0
        ),
    }


def metadata_differences(reference, target):
    keys = sorted(set(reference) | set(target))
    ignored = {"csv_columns", "started_at_unix"}
    return {
        key: {"reference": reference.get(key), "target": target.get(key)}
        for key in keys
        if key not in ignored and reference.get(key) != target.get(key)
    }


def comparison_metrics(reference_data, target_data, reference_meta, target_meta, options):
    result = {
        "metadata_differences": metadata_differences(reference_meta, target_meta),
        "target_matched_low_motion": {
            arm: target_matched_comparison(reference_data, target_data, arm, options)
            for arm in ARMS
        },
        "steady_state_change": {},
    }
    reference_analysis = analyze_session(reference_data, options)["steady_state"]
    target_analysis = analyze_session(target_data, options)["steady_state"]
    for arm in ARMS:
        arm_change = {}
        for metric in ("mean_error_cm", "median_error_cm", "p95_error_cm", "max_error_cm"):
            reference_value = reference_analysis[arm][metric]
            target_value = target_analysis[arm][metric]
            arm_change[metric] = {
                "reference": reference_value,
                "target": target_value,
                "change": _finite_float(target_value - reference_value),
                "change_percent": _finite_float(
                    (target_value / reference_value - 1.0) * 100.0
                ) if reference_value else None,
            }
        result["steady_state_change"][arm] = arm_change
    return result


def save_plot(target_data, target_analysis, output_path, options, reference_data=None):
    matplotlib_config = output_path.parent / ".matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None, "Matplotlib is unavailable; skipped plot generation"

    steady = target_data["t_rel"] >= options.startup_seconds
    figure, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    colors = {"left": "#2563eb", "right": "#dc2626"}
    for arm in ARMS:
        axes[0, 0].plot(
            target_data["t_rel"],
            target_data[f"{arm}_error_distance"] * 100.0,
            color=colors[arm],
            linewidth=1.0,
            label=arm.title(),
        )
    axes[0, 0].axvspan(
        target_data["t_rel"][0],
        options.startup_seconds,
        color="#d1d5db",
        alpha=0.45,
        label="startup excluded",
    )
    axes[0, 0].axhline(options.error_threshold_cm, color="#f59e0b", linestyle="--")
    axes[0, 0].axhline(10.0, color="#991b1b", linestyle="--")
    axes[0, 0].set(
        title="Target session error",
        xlabel="Time (s)",
        ylabel="3D error (cm)",
    )
    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[0, 0].grid(alpha=0.2)

    for arm in ARMS:
        target_error = np.sort(target_data[f"{arm}_error_distance"][steady] * 100.0)
        target_cdf = np.arange(1, len(target_error) + 1) / len(target_error)
        axes[0, 1].plot(
            target_error,
            target_cdf,
            color=colors[arm],
            linewidth=1.8,
            label=f"Target {arm}",
        )
        if reference_data is not None:
            reference_steady = reference_data["t_rel"] >= options.startup_seconds
            reference_error = np.sort(
                reference_data[f"{arm}_error_distance"][reference_steady] * 100.0
            )
            reference_cdf = np.arange(1, len(reference_error) + 1) / len(reference_error)
            axes[0, 1].plot(
                reference_error,
                reference_cdf,
                color=colors[arm],
                linewidth=1.0,
                linestyle="--",
                alpha=0.65,
                label=f"Reference {arm}",
            )
    axes[0, 1].set(
        title="Steady-state error CDF",
        xlabel="3D error (cm)",
        ylabel="Fraction at or below error",
    )
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.2)

    for index, arm in enumerate(ARMS):
        axis = axes[1, index]
        human = _xyz(target_data, arm, "human")
        robot = _xyz(target_data, arm, "robot")
        axis.scatter(
            human[steady, 0] * 100.0,
            robot[steady, 0] * 100.0,
            s=6,
            alpha=0.3,
            color=colors[arm],
        )
        low = min(human[steady, 0].min(), robot[steady, 0].min()) * 100.0
        high = max(human[steady, 0].max(), robot[steady, 0].max()) * 100.0
        axis.plot([low, high], [low, high], color="#111827", linestyle="--")
        axis.axvline(options.reach_x_cm, color="#f59e0b", linestyle=":")
        axis.set(
            title=f"{arm.title()} forward response",
            xlabel="Human target X (cm)",
            ylabel="Robot wrist X (cm)",
        )
        axis.grid(alpha=0.2)

    metadata = target_analysis.get("metadata", {})
    configuration = " ".join(
        str(metadata.get(key))
        for key in ("arm", "ee", "input_mode", "arm_reference_mode")
        if metadata.get(key) is not None
    )
    figure.suptitle(f"Head-relative teleoperation analysis: {configuration}".strip(), fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path, None


def print_report(summary):
    integrity = summary["target"]["analysis"]["integrity"]
    metadata = summary["target"]["metadata"]
    print(f"Run: {summary['target']['csv_path']}")
    print(
        "Config: "
        f"{metadata.get('arm', 'unknown')} / {metadata.get('ee', 'no_ee')} / "
        f"{metadata.get('input_mode', 'unknown')} / "
        f"{metadata.get('arm_reference_mode', 'unknown')} / "
        f"{metadata.get('head_origin_calibration', 'unspecified')}"
    )
    print(
        f"Data: {integrity['rows']} rows, {integrity['duration_s']:.2f} s, "
        f"{integrity['effective_rate_hz']:.2f} Hz; "
        f"gaps={integrity['missing_sequences']}, "
        f"IK failures={integrity['ik_failures']}, "
        f"non-finite cells={integrity['nonfinite_cells']}"
    )
    print(f"Steady state (after {summary['parameters']['startup_seconds']:g} s):")
    for arm in ARMS:
        values = summary["target"]["analysis"]["steady_state"][arm]
        lag = values["lag_estimate"]
        lag_text = (
            f"{lag['milliseconds']:.0f} ms"
            if lag["milliseconds"] is not None
            else "unavailable"
        )
        print(
            f"  {arm:>5}: mean={values['mean_error_cm']:.2f} cm, "
            f"median={values['median_error_cm']:.2f} cm, "
            f"p95={values['p95_error_cm']:.2f} cm, "
            f"max={values['max_error_cm']:.2f} cm, "
            f"lag={lag_text}"
        )
        for label, band in values["workspace_bands"].items():
            print(
                f"         {label}: n={band['samples']}, "
                f"mean={band['mean_error_cm']:.2f} cm, p95={band['p95_error_cm']:.2f} cm"
            )

    reference = summary.get("reference")
    comparison = summary.get("comparison")
    if reference and comparison:
        print(f"Reference: {reference['csv_path']}")
        for arm in ARMS:
            matched = comparison["target_matched_low_motion"][arm]
            if matched.get("available"):
                coverage_warning = (
                    " [LOW OVERLAP: treat comparison as indicative only]"
                    if matched["coverage_percent"] < 20.0
                    else ""
                )
                print(
                    f"  {arm:>5} target-matched: "
                    f"{matched['reference_mean_error_cm']:.2f} -> "
                    f"{matched['target_mean_error_cm']:.2f} cm "
                    f"({matched['relative_change_percent']:+.1f}%, "
                    f"coverage {matched['coverage_percent']:.1f}%)"
                    f"{coverage_warning}"
                )
            else:
                print(f"  {arm:>5} target-matched: unavailable ({matched['reason']})")

    outputs = summary.get("outputs", {})
    if outputs.get("json"):
        print(f"JSON: {outputs['json']}")
    if outputs.get("plot"):
        print(f"Plot: {outputs['plot']}")
    if outputs.get("plot_warning"):
        print(f"Plot warning: {outputs['plot_warning']}", file=sys.stderr)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a head_relative.csv run, compare it with a prior compatible "
            "session, and write JSON plus a diagnostic plot."
        )
    )
    parser.add_argument(
        "run",
        nargs="?",
        default="latest",
        help="Run directory, head_relative.csv path, or 'latest' (default).",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help=f"Directory searched for runs (default: {DEFAULT_LOGS_DIR}).",
    )
    parser.add_argument(
        "--baseline",
        default="auto",
        help="Reference run directory/CSV, 'auto' for previous compatible run, or 'none'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: ANALYZED_RUN/analysis).",
    )
    parser.add_argument("--startup-seconds", type=float, default=3.0)
    parser.add_argument("--error-threshold-cm", type=float, default=5.0)
    parser.add_argument("--near-x-cm", type=float, default=25.0)
    parser.add_argument("--reach-x-cm", type=float, default=35.0)
    parser.add_argument("--low-z-cm", type=float, default=-40.0)
    parser.add_argument("--max-lag-samples", type=int, default=10)
    parser.add_argument("--match-radius-cm", type=float, default=2.0)
    parser.add_argument("--match-max-speed-cm-s", type=float, default=20.0)
    parser.add_argument("--match-neighbors", type=int, default=20)
    parser.add_argument("--no-plot", action="store_true", help="Write JSON only.")
    return parser


def validate_options(options):
    if options.startup_seconds < 0.0:
        raise ValueError("--startup-seconds cannot be negative")
    if options.error_threshold_cm <= 0.0:
        raise ValueError("--error-threshold-cm must be positive")
    if options.near_x_cm >= options.reach_x_cm:
        raise ValueError("--near-x-cm must be less than --reach-x-cm")
    if options.max_lag_samples < 0:
        raise ValueError("--max-lag-samples cannot be negative")
    if options.match_radius_cm <= 0.0:
        raise ValueError("--match-radius-cm must be positive")
    if options.match_max_speed_cm_s <= 0.0:
        raise ValueError("--match-max-speed-cm-s must be positive")
    if options.match_neighbors < 1:
        raise ValueError("--match-neighbors must be at least 1")


def run_analysis(options):
    validate_options(options)
    target_csv = resolve_csv(options.run, options.logs_dir)
    target_metadata, target_meta_path = load_metadata(target_csv)
    target_data = load_csv(target_csv)
    target_analysis = analyze_session(target_data, options)

    reference_csv = None
    if options.baseline != "none":
        if options.baseline == "auto":
            reference_csv = find_previous_compatible(
                target_csv, options.logs_dir, target_metadata
            )
        else:
            reference_csv = resolve_csv(options.baseline, options.logs_dir)

    reference_data = None
    reference_metadata = {}
    reference_meta_path = None
    comparison = None
    if reference_csv is not None:
        reference_metadata, reference_meta_path = load_metadata(reference_csv)
        reference_data = load_csv(reference_csv)
        comparison = comparison_metrics(
            reference_data,
            target_data,
            reference_metadata,
            target_metadata,
            options,
        )

    output_dir = (
        options.output_dir.expanduser().resolve()
        if options.output_dir
        else target_csv.parent / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "analysis_summary.json"
    plot_path = output_dir / "analysis.png"

    summary = {
        "analysis_version": 1,
        "generated_at_unix": time.time(),
        "parameters": {
            "startup_seconds": options.startup_seconds,
            "error_threshold_cm": options.error_threshold_cm,
            "near_x_cm": options.near_x_cm,
            "reach_x_cm": options.reach_x_cm,
            "low_z_cm": options.low_z_cm,
            "max_lag_samples": options.max_lag_samples,
            "match_radius_cm": options.match_radius_cm,
            "match_max_speed_cm_s": options.match_max_speed_cm_s,
            "match_neighbors": options.match_neighbors,
        },
        "target": {
            "csv_path": str(target_csv),
            "metadata_path": str(target_meta_path) if target_meta_path else None,
            "metadata": target_metadata,
            "analysis": target_analysis,
        },
        "reference": (
            {
                "csv_path": str(reference_csv),
                "metadata_path": str(reference_meta_path) if reference_meta_path else None,
                "metadata": reference_metadata,
            }
            if reference_csv is not None
            else None
        ),
        "comparison": comparison,
        "outputs": {
            "json": str(json_path),
            "plot": None,
            "plot_warning": None,
        },
    }

    if not options.no_plot:
        written_plot, warning = save_plot(
            target_data,
            {"metadata": target_metadata, **target_analysis},
            plot_path,
            options,
            reference_data=reference_data,
        )
        summary["outputs"]["plot"] = str(written_plot) if written_plot else None
        summary["outputs"]["plot_warning"] = warning

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return summary


def main(argv: Iterable[str] | None = None):
    parser = build_parser()
    options = parser.parse_args(argv)
    try:
        summary = run_analysis(options)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print_report(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
