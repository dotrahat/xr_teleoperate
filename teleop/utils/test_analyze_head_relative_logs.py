"""Offline tests for the reusable head-relative log analyzer."""

import csv
import json
from pathlib import Path
import tempfile

import numpy as np

from teleop.utils.analyze_head_relative_logs import (
    CSV_COLUMNS,
    analyze_session,
    build_parser,
    find_previous_compatible,
    load_csv,
    resolve_csv,
    run_analysis,
    target_matched_comparison,
)
from teleop.utils.head_relative_monitor import _CSV_HEADER, csv_row, relative_measurement


def _write_run(root, name, error_m, started_at, calibration="legacy_virtual_head"):
    run_dir = Path(root) / name
    run_dir.mkdir()
    csv_path = run_dir / "head_relative.csv"
    rows = []
    for seq in range(150):
        t_rel = seq / 30.0
        human_x = 0.20 + seq * 0.0005
        human_left = np.array([human_x, 0.15, -0.25])
        human_right = np.array([human_x, -0.15, -0.25])
        robot_left = human_left + [error_m, 0.0, 0.0]
        robot_right = human_right + [error_m, 0.0, 0.0]
        rows.append(csv_row(
            seq,
            started_at + t_rel,
            t_rel,
            True,
            relative_measurement(human_left, robot_left, np.zeros(3)),
            relative_measurement(human_right, robot_right, np.zeros(3)),
        ))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_CSV_HEADER)
        writer.writerows(rows)
    metadata = {
        "arm": "G1_23",
        "ee": "brainco",
        "input_mode": "hand",
        "arm_reference_mode": "head_yaw",
        "sim": False,
        "head_origin_calibration": calibration,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    return csv_path


def test_analyzer_schema_matches_monitor_schema():
    assert CSV_COLUMNS == _CSV_HEADER


def test_latest_resolution_and_previous_compatible_discovery():
    with tempfile.TemporaryDirectory() as directory:
        older = _write_run(directory, "older", 0.05, 1000.0)
        newer = _write_run(directory, "newer", 0.01, 2000.0, "g1_urdf_camera_midline")
        older.touch()
        newer.touch()
        older_mtime = older.stat().st_mtime
        newer_mtime = older_mtime + 10.0
        import os
        os.utime(newer, (newer_mtime, newer_mtime))

        assert resolve_csv("latest", directory) == newer.resolve()
        metadata = json.loads(newer.with_name("run_meta.json").read_text())
        assert find_previous_compatible(newer, directory, metadata) == older.resolve()


def test_metrics_and_target_matching_detect_improvement():
    with tempfile.TemporaryDirectory() as directory:
        reference_path = _write_run(directory, "reference", 0.05, 1000.0)
        target_path = _write_run(directory, "target", 0.01, 2000.0)
        reference = load_csv(reference_path)
        target = load_csv(target_path)
        options = build_parser().parse_args([
            str(target_path),
            "--startup-seconds", "0",
            "--match-radius-cm", "2",
        ])
        analysis = analyze_session(target, options)
        assert np.isclose(analysis["steady_state"]["left"]["mean_error_cm"], 1.0)
        assert analysis["integrity"]["missing_sequences"] == 0

        matched = target_matched_comparison(reference, target, "left", options)
        assert matched["available"]
        assert matched["coverage_percent"] == 100.0
        assert np.isclose(matched["relative_change_percent"], -80.0)


def test_end_to_end_json_only_analysis():
    with tempfile.TemporaryDirectory() as directory:
        reference = _write_run(directory, "reference", 0.05, 1000.0)
        target = _write_run(directory, "target", 0.01, 2000.0)
        output_dir = Path(directory) / "outputs"
        options = build_parser().parse_args([
            str(target),
            "--baseline", str(reference),
            "--output-dir", str(output_dir),
            "--startup-seconds", "0",
            "--no-plot",
        ])
        summary = run_analysis(options)
        json_path = output_dir / "analysis_summary.json"
        assert json_path.is_file()
        saved = json.loads(json_path.read_text())
        assert saved["target"]["csv_path"] == str(target.resolve())
        assert saved["comparison"]["target_matched_low_motion"]["left"]["available"]
        assert summary["outputs"]["plot"] is None


def test_nonfinite_source_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        source = _write_run(directory, "invalid", 0.01, 2000.0)
        rows = source.read_text(encoding="utf-8").splitlines()
        cells = rows[-1].split(",")
        cells[4] = "nan"
        rows[-1] = ",".join(cells)
        source.write_text("\n".join(rows) + "\n", encoding="utf-8")
        try:
            load_csv(source)
        except ValueError as exc:
            assert "non-finite" in str(exc)
        else:
            raise AssertionError("non-finite source data should be rejected")


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} head-relative analyzer tests passed")
