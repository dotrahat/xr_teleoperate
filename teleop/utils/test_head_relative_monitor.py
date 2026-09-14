"""Offline tests for the standalone head-relative wrist monitor."""

import csv
from pathlib import Path
import tempfile
import numpy as np

from teleop.utils.head_relative_monitor import (
    ROBOT_HEAD_IN_WAIST_M,
    HeadRelativeMonitor,
    _CSV_HEADER,
    csv_row,
    relative_measurement,
)


def test_matching_head_relative_vectors_overlap():
    human_wrist = ROBOT_HEAD_IN_WAIST_M + np.array([0.40, 0.12, -0.08])
    robot_wrist = ROBOT_HEAD_IN_WAIST_M + np.array([0.40, 0.12, -0.08])
    measurement = relative_measurement(human_wrist, robot_wrist)

    assert np.allclose(measurement.human, [0.40, 0.12, -0.08])
    assert np.allclose(measurement.robot, measurement.human)
    assert np.allclose(measurement.error, np.zeros(3))
    assert measurement.error_distance == 0.0


def test_equal_distances_do_not_hide_axis_error():
    measurement = relative_measurement(
        ROBOT_HEAD_IN_WAIST_M + np.array([0.4, 0.0, 0.0]),
        ROBOT_HEAD_IN_WAIST_M + np.array([0.0, 0.4, 0.0]),
    )

    assert np.isclose(measurement.human_distance, measurement.robot_distance)
    assert not np.allclose(measurement.human, measurement.robot)
    assert measurement.error_distance > 0.0


def test_body_offset_is_subtracted_without_scaling():
    vector = np.array([0.48, -0.21, 0.13])
    measurement = relative_measurement(
        ROBOT_HEAD_IN_WAIST_M + vector,
        ROBOT_HEAD_IN_WAIST_M,
    )

    assert np.allclose(measurement.human, vector)
    assert np.array_equal(ROBOT_HEAD_IN_WAIST_M, [0.15, 0.0, 0.45])


def test_csv_schema_and_row_stay_aligned():
    left = relative_measurement([0.55, 0.1, 0.45], [0.50, 0.1, 0.45])
    right = relative_measurement([0.55, -0.1, 0.45], [0.50, -0.1, 0.45])
    row = csv_row(7, 100.0, 1.5, True, left, right)

    assert len(row) == len(_CSV_HEADER)
    assert row[:4] == [7, 100.0, 1.5, 1]


def test_monitor_source_is_episode_and_rerun_independent():
    source = (Path(__file__).with_name("head_relative_monitor.py")).read_text(encoding="utf-8")
    assert "episode_writer" not in source.lower()
    assert "rerun" not in source.lower()


def test_main_baseline_provenance_is_logged():
    source = (Path(__file__).parents[1] / "teleop_hand_and_arm.py").read_text(
        encoding="utf-8"
    )
    assert '"comparison_role": "main_behavior_baseline"' in source
    assert '"behavior_base_branch": "main"' in source
    assert '"instrumentation_only": True' in source
    assert '"baseline_behavior_unchanged": True' in source
    assert '"head_origin_calibration": "legacy_virtual_head"' in source
    assert "arm_ctrl = None" in source
    assert "if arm_ctrl is not None:" in source


def test_main_ik_endpoint_is_unchanged():
    source = (
        Path(__file__).parents[1] / "robot_control" / "robot_arm_ik.py"
    ).read_text(encoding="utf-8")
    assert 'self.cache_path = "g1_23_mode10_model_cache.pkl"' in source
    assert source.count("np.array([0.20,0,0]).T") >= 2


def test_spawned_csv_monitor_uses_actual_fk_without_a_viewer():
    import pinocchio as pin

    model = pin.Model()
    left_id = model.addFrame(pin.Frame(
        "test_left_wrist", 0, pin.SE3(np.eye(3), np.array([0.55, 0.10, 0.45])),
        pin.FrameType.OP_FRAME,
    ))
    right_id = model.addFrame(pin.Frame(
        "test_right_wrist", 0, pin.SE3(np.eye(3), np.array([0.55, -0.10, 0.45])),
        pin.FrameType.OP_FRAME,
    ))

    with tempfile.TemporaryDirectory() as out_dir:
        monitor = HeadRelativeMonitor(
            model=model,
            left_frame_id=left_id,
            right_frame_id=right_id,
            out_dir=out_dir,
            metadata={"test": True},
            show_plot=False,
        )
        desired_left = np.eye(4)
        desired_left[:3, 3] = [0.60, 0.10, 0.45]
        desired_right = np.eye(4)
        desired_right[:3, 3] = [0.60, -0.10, 0.45]
        monitor.log(0, 0.0, desired_left, desired_right, np.zeros(model.nq), True)
        monitor.close(timeout=10.0)

        csv_path = Path(out_dir) / "head_relative.csv"
        rows = list(csv.reader(csv_path.open(newline="", encoding="utf-8")))
        assert rows[0] == _CSV_HEADER
        assert len(rows) == 2
        assert len(rows[1]) == len(_CSV_HEADER)


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} head-relative-monitor tests passed")
