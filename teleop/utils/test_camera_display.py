"""Offline tests for camera selection and XR canvas composition."""

import numpy as np
import pytest

from teleop.utils.camera_display import (
    AlternatingFistGesture,
    CameraDisplay,
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


def _fist_step(gesture, hand, start):
    pose = (True, False) if hand == "left" else (False, True)
    assert not gesture.update(*pose, start)
    triggered = gesture.update(*pose, start + 0.21)
    assert not gesture.update(False, False, start + 0.25)
    return triggered


def _complete_fist_code(gesture, start=0.0):
    assert not _fist_step(gesture, "left", start)
    assert not _fist_step(gesture, "right", start + 0.6)
    assert not _fist_step(gesture, "left", start + 1.2)
    return _fist_step(gesture, "right", start + 1.8)


def test_alternating_fist_code_triggers_once_in_the_correct_order():
    gesture = AlternatingFistGesture()
    assert not gesture.update(False, False, -0.1)
    assert _complete_fist_code(gesture)


def test_held_fist_counts_only_once_and_release_is_required():
    gesture = AlternatingFistGesture()
    assert not gesture.update(False, False, -0.1)
    assert not gesture.update(True, False, 0.0)
    assert not gesture.update(True, False, 0.21)
    for now in (0.5, 1.0, 2.0):
        assert not gesture.update(True, False, now)


def test_gesture_requires_an_initial_fully_released_sample():
    gesture = AlternatingFistGesture()
    assert not gesture.update(True, False, 0.0)
    assert not gesture.update(True, False, 0.3)
    assert not gesture.update(False, False, 0.4)
    assert not _fist_step(gesture, "left", 0.5)


def test_both_fists_or_wrong_order_resets_the_sequence():
    gesture = AlternatingFistGesture()
    assert not gesture.update(False, False, -0.1)
    assert not _fist_step(gesture, "left", 0.0)
    assert not gesture.update(True, True, 0.6)
    assert not gesture.update(True, True, 0.9)
    assert not gesture.update(False, False, 1.0)
    assert _complete_fist_code(gesture, 1.1)


def test_alternating_fist_sequence_expires_and_restarts_from_left():
    gesture = AlternatingFistGesture(sequence_timeout_s=3.0)
    assert not gesture.update(False, False, -0.1)
    assert not _fist_step(gesture, "left", 0.0)
    assert not _fist_step(gesture, "right", 3.5)
    assert _complete_fist_code(gesture, 4.0)


def test_gesture_does_not_restart_while_cooling_down():
    gesture = AlternatingFistGesture(cooldown_s=1.0)
    assert not gesture.update(False, False, -0.1)
    assert _complete_fist_code(gesture)
    assert not gesture.update(True, False, 2.2)
    assert not gesture.update(True, False, 3.3)
    assert not gesture.update(False, False, 3.4)
