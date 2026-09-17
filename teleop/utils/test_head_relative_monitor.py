"""Offline tests for the standalone head-relative wrist monitor."""

import ast
import csv
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from teleop.utils.g1_23_geometry import G1_23_BRAINCO_WRIST_OFFSET_M
from teleop.utils.head_relative_monitor import (
    G1_23_CALIBRATED_LEFT_WRIST_RESIDUAL_M,
    G1_23_CALIBRATED_RIGHT_WRIST_RESIDUAL_M,
    G1_CALIBRATED_HEAD_IN_WAIST_M,
    ROBOT_HEAD_IN_WAIST_M,
    SUPPORTED_G1_ARMS,
    HeadRelativeMonitor,
    _CSV_HEADER,
    apply_wrist_target_residual,
    csv_row,
    g1_wrist_target_residuals,
    head_relative_monitor_run_name,
    relative_measurement,
    retarget_wrist_pose_head_origin,
    robot_head_in_waist,
    validate_head_relative_monitor_config,
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


def test_g1_head_origin_calibration_is_opt_in_and_g1_only():
    assert np.array_equal(robot_head_in_waist("G1_23"), ROBOT_HEAD_IN_WAIST_M)
    for arm in SUPPORTED_G1_ARMS:
        selected = robot_head_in_waist(arm, use_g1_calibration=True)
        assert np.array_equal(selected, G1_CALIBRATED_HEAD_IN_WAIST_M)
        selected[0] = 999.0
        assert G1_CALIBRATED_HEAD_IN_WAIST_M[0] != 999.0

    try:
        robot_head_in_waist("H2", use_g1_calibration=True)
    except ValueError as exc:
        assert "G1 teleoperation only" in str(exc)
    else:
        raise AssertionError("G1 head calibration unexpectedly accepted H2")


def test_wrist_retarget_preserves_relative_vector_and_orientation():
    wrist = np.eye(4)
    wrist[:3, 3] = ROBOT_HEAD_IN_WAIST_M + [0.30, -0.20, -0.25]
    wrist[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    calibrated = retarget_wrist_pose_head_origin(
        wrist, G1_CALIBRATED_HEAD_IN_WAIST_M
    )

    assert np.allclose(
        calibrated[:3, 3] - G1_CALIBRATED_HEAD_IN_WAIST_M,
        wrist[:3, 3] - ROBOT_HEAD_IN_WAIST_M,
    )
    assert np.array_equal(calibrated[:3, :3], wrist[:3, :3])
    assert np.allclose(wrist[:3, 3], [0.45, -0.20, 0.20])


def test_calibrated_head_origin_matches_cad_dimensions():
    expected_from_cad_mm = np.array([47.64571478, 0.0, 462.68178553])
    assert np.allclose(
        G1_CALIBRATED_HEAD_IN_WAIST_M,
        expected_from_cad_mm / 1000.0,
        atol=1e-12,
    )


def test_g1_23_wrist_residuals_are_opt_in_and_do_not_change_x():
    zero_left, zero_right = g1_wrist_target_residuals(
        "G1_23", use_g1_calibration=False
    )
    assert np.array_equal(zero_left, np.zeros(3))
    assert np.array_equal(zero_right, np.zeros(3))

    left, right = g1_wrist_target_residuals(
        "G1_23", use_g1_calibration=True
    )
    assert np.array_equal(left, G1_23_CALIBRATED_LEFT_WRIST_RESIDUAL_M)
    assert np.array_equal(right, G1_23_CALIBRATED_RIGHT_WRIST_RESIDUAL_M)
    assert left[0] == right[0] == 0.0
    assert left[1] > 0.0 and right[1] < 0.0
    assert left[2] == right[2] == -0.0075

    g1_29_left, g1_29_right = g1_wrist_target_residuals(
        "G1_29", use_g1_calibration=True
    )
    assert np.array_equal(g1_29_left, np.zeros(3))
    assert np.array_equal(g1_29_right, np.zeros(3))


def test_wrist_target_residual_preserves_orientation_and_input():
    wrist = np.eye(4)
    wrist[:3, 3] = [0.3, 0.2, -0.1]
    wrist[:3, :3] = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    original = wrist.copy()

    corrected = apply_wrist_target_residual(
        wrist, G1_23_CALIBRATED_LEFT_WRIST_RESIDUAL_M
    )

    assert np.array_equal(wrist, original)
    assert np.array_equal(corrected[:3, :3], wrist[:3, :3])
    assert np.allclose(
        corrected[:3, 3],
        wrist[:3, 3] + G1_23_CALIBRATED_LEFT_WRIST_RESIDUAL_M,
    )


def test_csv_schema_and_row_stay_aligned():
    left = relative_measurement([0.55, 0.1, 0.45], [0.50, 0.1, 0.45])
    right = relative_measurement([0.55, -0.1, 0.45], [0.50, -0.1, 0.45])
    row = csv_row(7, 100.0, 1.5, True, left, right)

    assert len(row) == len(_CSV_HEADER)
    assert row[:4] == [7, 100.0, 1.5, 1]


def test_brainco_wrist_offset_matches_custom_urdf():
    repo_root = Path(__file__).resolve().parents[2]
    urdf = ET.parse(
        repo_root / "assets/g1/mode10/g1_23dof_mode_10_with_brainco.urdf"
    ).getroot()
    joints = {joint.attrib["name"]: joint for joint in urdf.findall("joint")}

    for side in ("left", "right"):
        xyz = np.fromstring(joints[f"{side}_base_joint"].find("origin").attrib["xyz"], sep=" ")
        assert np.array_equal(xyz, [G1_23_BRAINCO_WRIST_OFFSET_M, 0.0, 0.0])


def test_monitor_source_is_episode_and_rerun_independent():
    source = (Path(__file__).with_name("head_relative_monitor.py")).read_text(encoding="utf-8")
    assert "episode_writer" not in source.lower()
    assert "rerun" not in source.lower()


def test_monitor_config_accepts_every_g1_reference_mode():
    assert SUPPORTED_G1_ARMS == {"G1_23", "G1_29"}
    for arm in SUPPORTED_G1_ARMS:
        for reference_mode in ("head_yaw", "head_position"):
            validate_head_relative_monitor_config(arm, reference_mode, 20.0, 10.0)


def test_monitor_config_rejects_non_g1_arms():
    for arm in ("H1_2", "H1", "H2", "R1_A5", "R1_A7"):
        try:
            validate_head_relative_monitor_config(arm, "head_yaw", 20.0, 10.0)
        except ValueError as exc:
            assert "G1 teleoperation only" in str(exc)
        else:
            raise AssertionError(f"monitor unexpectedly accepted {arm}")


def test_monitor_config_rejects_invalid_rates():
    for window_seconds, plot_rate_hz, expected in (
        (0.0, 10.0, "window"),
        (20.0, 0.0, "rate"),
    ):
        try:
            validate_head_relative_monitor_config(
                "G1_29", "head_position", window_seconds, plot_rate_hz
            )
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("monitor unexpectedly accepted an invalid rate")


def test_monitor_run_name_describes_the_full_configuration():
    assert head_relative_monitor_run_name(
        True, "G1_29", "dex3", "hand", "head_yaw", "20260914_120000"
    ) == "sim_G1_29_dex3_hand_head_yaw_20260914_120000"
    assert head_relative_monitor_run_name(
        False, "G1_23", None, "controller", "head_position", "20260914_120001"
    ) == "real_G1_23_no_ee_controller_head_position_20260914_120001"


def test_g1_29_solver_updates_convergence_status_on_both_paths():
    repo_root = Path(__file__).resolve().parents[2]
    tree = ast.parse(
        (repo_root / "teleop/robot_control/robot_arm_ik.py").read_text(encoding="utf-8")
    )
    g1_29 = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "G1_29_ArmIK"
    )
    init = next(
        node for node in g1_29.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    solve = next(
        node for node in g1_29.body
        if isinstance(node, ast.FunctionDef) and node.name == "solve_ik"
    )

    def assigned_status_values(nodes):
        values = []
        for node in nodes:
            for child in ast.walk(node):
                if not isinstance(child, ast.Assign) or len(child.targets) != 1:
                    continue
                target = child.targets[0]
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr == "last_solve_ok"
                    and isinstance(child.value, ast.Constant)
                ):
                    values.append(child.value.value)
        return values

    assert True in assigned_status_values(init.body)
    solve_try = next(node for node in solve.body if isinstance(node, ast.Try))
    assert True in assigned_status_values(solve_try.body)
    assert False in assigned_status_values(solve_try.handlers)


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

    desired_left = np.eye(4)
    desired_left[:3, 3] = [0.60, 0.10, 0.45]
    desired_right = np.eye(4)
    desired_right[:3, 3] = [0.60, -0.10, 0.45]

    with tempfile.TemporaryDirectory() as out_dir:
        for arm, reference_mode in (
            ("G1_23", "head_yaw"),
            ("G1_29", "head_position"),
        ):
            run_dir = Path(out_dir) / arm
            metadata = {
                "arm": arm,
                "ee": None,
                "input_mode": "controller",
                "arm_reference_mode": reference_mode,
                "sim": True,
            }
            monitor = HeadRelativeMonitor(
                model=model,
                left_frame_id=left_id,
                right_frame_id=right_id,
                out_dir=run_dir,
                metadata=metadata,
                show_plot=False,
                robot_head_in_waist=G1_CALIBRATED_HEAD_IN_WAIST_M,
            )
            monitor.log(0, 0.0, desired_left, desired_right, np.zeros(model.nq), True)
            monitor.close(timeout=10.0)

            rows = list(csv.reader(
                (run_dir / "head_relative.csv").open(newline="", encoding="utf-8")
            ))
            assert rows[0] == _CSV_HEADER
            assert len(rows) == 2
            assert len(rows[1]) == len(_CSV_HEADER)
            run_meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
            for key, value in metadata.items():
                assert run_meta[key] == value
            assert np.allclose(
                run_meta["robot_head_in_waist_m"], G1_CALIBRATED_HEAD_IN_WAIST_M
            )


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} head-relative-monitor tests passed")
