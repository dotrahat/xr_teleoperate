"""Offline tests for camera selection and XR canvas composition."""

import numpy as np
import pytest

from teleop.utils.camera_display import (
    AlternatingFistGesture,
    CameraDisplay,
    DualRingPinchGesture,
    TripleDoublePinchGesture,
    camera_configs,
    resolve_camera_names,
)


CONFIG = {
    "webrtc": {"bitrate": {"max": 12_000_000}},
    "head_camera": {
        "enable_zmq": True,
        "enable_webrtc": True,
        "image_shape": [60, 80],
        "binocular": False,
    },
    "left_wrist_camera": {
        "enable_zmq": True,
        "enable_webrtc": False,
        "image_shape": [40, 40],
        "binocular": False,
    },
    "disabled_camera": {
        "enable_zmq": False,
        "enable_webrtc": False,
        "image_shape": [40, 40],
        "binocular": False,
    },
}


def test_camera_config_discovery_and_selection():
    assert list(camera_configs(CONFIG)) == [
        "head_camera",
        "left_wrist_camera",
        "disabled_camera",
    ]
    assert resolve_camera_names(CONFIG, "auto") == ["head_camera"]
    assert resolve_camera_names(CONFIG, "all") == ["head_camera", "left_wrist_camera"]
    assert resolve_camera_names(CONFIG, "left_wrist_camera,head_camera") == [
        "left_wrist_camera",
        "head_camera",
    ]


def test_invalid_or_disabled_camera_selection_is_rejected():
    with pytest.raises(ValueError, match="Unknown camera"):
        resolve_camera_names(CONFIG, "elbow_camera")
    with pytest.raises(ValueError, match="disabled"):
        resolve_camera_names(CONFIG, "disabled_camera")


def test_single_layout_cycles_selected_cameras_and_preserves_canvas_shape():
    display = CameraDisplay(CONFIG, ["head_camera", "left_wrist_camera"], "single")
    frames = {
        "head_camera": np.full((60, 80, 3), 10, dtype=np.uint8),
        "left_wrist_camera": np.full((40, 40, 3), 200, dtype=np.uint8),
    }
    assert display.active_camera == "head_camera"
    assert display.compose(frames).shape == (60, 80, 3)
    assert display.cycle() == "left_wrist_camera"
    wrist_canvas = display.compose(frames)
    assert wrist_canvas.shape == (60, 80, 3)
    assert wrist_canvas[30, 40, 0] == 200


def test_side_by_side_layout_supports_missing_frames():
    display = CameraDisplay(CONFIG, ["head_camera", "left_wrist_camera"], "side-by-side")
    canvas = display.compose({"head_camera": np.full((60, 80, 3), 25, dtype=np.uint8)})
    assert canvas.shape == (60, 80, 3)
    assert canvas[40, 20, 0] == 25
    assert canvas[40, 60, 0] == 0


def test_binocular_reference_duplicates_multi_camera_dashboard_for_both_eyes():
    config = dict(CONFIG)
    config["head_camera"] = dict(CONFIG["head_camera"], image_shape=[60, 160], binocular=True)
    display = CameraDisplay(config, ["head_camera", "left_wrist_camera"], "side-by-side")
    head = np.concatenate(
        (
            np.full((60, 80, 3), 20, dtype=np.uint8),
            np.full((60, 80, 3), 40, dtype=np.uint8),
        ),
        axis=1,
    )
    canvas = display.compose({"head_camera": head})
    assert canvas.shape == (60, 160, 3)
    np.testing.assert_array_equal(canvas[:, :80], canvas[:, 80:])


def _ring_pinch_hand():
    hand = np.zeros((25, 3), dtype=float)
    # Straight index, middle, and little fingers. The WebXR joint groups are
    # metacarpal -> proximal -> intermediate -> distal -> tip.
    for indices, y in (((5, 6, 7, 8, 9), -0.02),
                       ((10, 11, 12, 13, 14), 0.0),
                       ((20, 21, 22, 23, 24), 0.04)):
        for offset, index in enumerate(indices):
            hand[index] = [0.04 + 0.015 * offset, y, 0.0]
    hand[4] = [0.09, 0.02, 0.0]     # thumb tip
    hand[19] = [0.091, 0.02, 0.0]   # ring tip touching thumb
    return hand


def _wrist_pose(side):
    pose = np.eye(4)
    # With identity rotation, left palm normal is -Y and right is +Y.
    pose[:3, 3] = [0.15, 0.30 if side == "left" else -0.30, 0.45]
    return pose


def test_dual_ring_pinch_matches_requested_hand_geometry():
    gesture = DualRingPinchGesture()
    hand = _ring_pinch_hand()
    assert gesture.hand_matches(hand, _wrist_pose("left"), "left")
    assert gesture.hand_matches(hand, _wrist_pose("right"), "right")


def test_ring_tip_must_touch_thumb_tip():
    gesture = DualRingPinchGesture()
    hand = _ring_pinch_hand()
    hand[19] = [0.16, 0.02, 0.0]
    assert not gesture.hand_matches(hand, _wrist_pose("left"), "left")


def test_index_middle_and_little_fingers_must_remain_straight():
    gesture = DualRingPinchGesture()
    hand = _ring_pinch_hand()
    hand[7] += [0.0, 0.06, 0.0]
    assert not gesture.hand_matches(hand, _wrist_pose("left"), "left")


def test_palm_must_face_the_headset():
    gesture = DualRingPinchGesture()
    wrist = _wrist_pose("left")
    wrist[:3, :3] = np.diag([1.0, -1.0, -1.0])
    assert not gesture.hand_matches(_ring_pinch_hand(), wrist, "left")


def test_dual_ring_pinch_requires_three_continuous_seconds_and_latches():
    gesture = DualRingPinchGesture()
    left_hand = _ring_pinch_hand()
    right_hand = _ring_pinch_hand()
    left_wrist = _wrist_pose("left")
    right_wrist = _wrist_pose("right")

    args = (left_hand, right_hand, left_wrist, right_wrist)
    assert not gesture.update(*args, now=0.0)
    assert not gesture.update(*args, now=2.99)
    assert gesture.update(*args, now=3.0)
    assert not gesture.update(*args, now=6.0)

    # Breaking the pose rearms it; the next valid pose needs a fresh full hold.
    released = right_hand.copy()
    released[19] = [0.16, 0.02, 0.0]
    assert not gesture.update(left_hand, released, left_wrist, right_wrist, now=6.1)
    assert not gesture.update(*args, now=7.0)
    assert gesture.update(*args, now=10.0)


def test_invalid_or_missing_hand_tracking_cannot_trigger():
    gesture = DualRingPinchGesture(hold_s=0.1)
    valid = _ring_pinch_hand()
    wrist = _wrist_pose("left")
    assert not gesture.update(np.zeros((0, 3)), valid, wrist, _wrist_pose("right"), now=0.0)
    assert not gesture.update(np.zeros((0, 3)), valid, wrist, _wrist_pose("right"), now=1.0)


def _fist_step(gesture, hand, start):
    pose = (True, False) if hand == "left" else (False, True)
    assert not gesture.update(*pose, start)
    triggered = gesture.update(*pose, start + 0.21)
    assert not gesture.update(False, False, start + 0.25)
    return triggered


def test_alternating_fists_remains_available_for_comparison():
    gesture = AlternatingFistGesture()
    assert not gesture.update(False, False, -0.1)
    assert not _fist_step(gesture, "left", 0.0)
    assert not _fist_step(gesture, "right", 0.6)
    assert not _fist_step(gesture, "left", 1.2)
    assert _fist_step(gesture, "right", 1.8)


def _double_pinch_cycle(gesture, start):
    assert not gesture.update(True, False, start)
    triggered = gesture.update(True, True, start + 0.05)
    assert not gesture.update(False, False, start + 0.15)
    return triggered


def test_triple_double_pinch_remains_available_for_comparison():
    gesture = TripleDoublePinchGesture()
    assert not gesture.update(False, False, -0.1)
    assert not _double_pinch_cycle(gesture, 0.0)
    assert not _double_pinch_cycle(gesture, 0.6)
    assert _double_pinch_cycle(gesture, 1.2)
