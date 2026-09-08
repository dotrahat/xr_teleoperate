"""Offline tests for G1-23 + BrainCo shoulder-relative wrist retargeting."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from teleop.utils.arm_retarget import (
    DEFAULT_LEFT_HUMAN_SHOULDER_HEAD_M,
    DEFAULT_RIGHT_HUMAN_SHOULDER_HEAD_M,
    G1_23_BRAINCO_WRIST_OFFSET_M,
    G1_23_FORWARD_SHOULDER_TO_WRIST_M,
    G1_23_LEFT_SHOULDER_M,
    G1_23_RIGHT_SHOULDER_M,
    HEAD_TO_WAIST_BODY_OFFSET_M,
    G1_23_BraincoArmRetargeter,
    parse_xyz,
)


def _pose(translation, rotation=None):
    pose = np.eye(4)
    pose[:3, 3] = translation
    if rotation is not None:
        pose[:3, :3] = rotation
    return pose


def test_measured_arm_length_produces_expected_scale():
    retargeter = G1_23_BraincoArmRetargeter(human_arm_length_m=0.48)
    assert np.isclose(retargeter.scale, 0.871504603302)


def test_full_human_extension_maps_to_full_robot_forward_reach():
    retargeter = G1_23_BraincoArmRetargeter(human_arm_length_m=0.48)
    left_source_shoulder = DEFAULT_LEFT_HUMAN_SHOULDER_HEAD_M + HEAD_TO_WAIST_BODY_OFFSET_M
    right_source_shoulder = DEFAULT_RIGHT_HUMAN_SHOULDER_HEAD_M + HEAD_TO_WAIST_BODY_OFFSET_M
    left = _pose(left_source_shoulder + np.array([0.48, 0.0, 0.0]))
    right = _pose(right_source_shoulder + np.array([0.48, 0.0, 0.0]))

    left_out, right_out = retargeter.retarget(left, right)

    expected_reach = np.array([G1_23_FORWARD_SHOULDER_TO_WRIST_M, 0.0, 0.0])
    assert np.allclose(left_out[:3, 3], G1_23_LEFT_SHOULDER_M + expected_reach)
    assert np.allclose(right_out[:3, 3], G1_23_RIGHT_SHOULDER_M + expected_reach)


def test_rotation_and_input_poses_are_unchanged():
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    left = _pose([0.4, 0.2, 0.3], rotation)
    right = _pose([0.4, -0.2, 0.3], rotation)
    left_before = left.copy()
    right_before = right.copy()

    left_out, right_out = G1_23_BraincoArmRetargeter().retarget(left, right)

    assert np.array_equal(left, left_before)
    assert np.array_equal(right, right_before)
    assert np.array_equal(left_out[:3, :3], rotation)
    assert np.array_equal(right_out[:3, :3], rotation)


def test_default_anchors_preserve_verified_body_offset():
    assert np.allclose(
        DEFAULT_LEFT_HUMAN_SHOULDER_HEAD_M + HEAD_TO_WAIST_BODY_OFFSET_M,
        G1_23_LEFT_SHOULDER_M,
    )
    assert np.allclose(
        DEFAULT_RIGHT_HUMAN_SHOULDER_HEAD_M + HEAD_TO_WAIST_BODY_OFFSET_M,
        G1_23_RIGHT_SHOULDER_M,
    )
    assert np.array_equal(HEAD_TO_WAIST_BODY_OFFSET_M, [0.15, 0.0, 0.45])


def test_wrist_mount_offset_matches_custom_urdf():
    repo_root = Path(__file__).resolve().parents[2]
    urdf = ET.parse(
        repo_root / "assets/g1/mode10/g1_23dof_mode_10_with_brainco.urdf"
    ).getroot()
    joints = {joint.attrib["name"]: joint for joint in urdf.findall("joint")}

    for side in ("left", "right"):
        xyz = parse_xyz(joints[f"{side}_base_joint"].find("origin").attrib["xyz"].split())
        assert np.array_equal(xyz, [G1_23_BRAINCO_WRIST_OFFSET_M, 0.0, 0.0])


def test_robot_shoulder_constants_match_custom_urdf():
    repo_root = Path(__file__).resolve().parents[2]
    urdf = ET.parse(
        repo_root / "assets/g1/mode10/g1_23dof_mode_10_with_brainco.urdf"
    ).getroot()
    joints = {joint.attrib["name"]: joint for joint in urdf.findall("joint")}
    waist = parse_xyz(joints["waist_yaw_joint"].find("origin").attrib["xyz"].split())
    left = waist + parse_xyz(
        joints["left_shoulder_pitch_joint"].find("origin").attrib["xyz"].split()
    )
    right = waist + parse_xyz(
        joints["right_shoulder_pitch_joint"].find("origin").attrib["xyz"].split()
    )

    assert np.allclose(left, G1_23_LEFT_SHOULDER_M)
    assert np.allclose(right, G1_23_RIGHT_SHOULDER_M)


def test_parse_xyz_rejects_bad_values():
    assert np.array_equal(parse_xyz("1,2,3"), [1.0, 2.0, 3.0])
    for bad in ("1,2", "1,2,3,4", "1,nope,3", "1,nan,3"):
        try:
            parse_xyz(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"parse_xyz should reject {bad!r}")


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} arm-retarget tests passed")
