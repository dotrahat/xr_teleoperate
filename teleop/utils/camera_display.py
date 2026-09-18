"""Camera selection and headset-canvas composition for XR teleoperation."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping

import cv2
import numpy as np


class DualRingPinchGesture:
    """Detect a two-hand thumb-to-ring pinch pose held toward the headset.

    Both thumb tips must touch their ring fingertips, while index, middle, and
    little fingers remain straight.  Both palms must face the headset.  The full
    pose must remain valid continuously for ``hold_s`` and must be released
    after a trigger before it can trigger again.
    """

    _WRIST = 0
    _THUMB_TIP = 4
    _MIDDLE_METACARPAL = 10
    _RING_TIP = 19
    _STRAIGHT_FINGERS = (
        (5, 6, 7, 8, 9),      # index
        (10, 11, 12, 13, 14), # middle
        (20, 21, 22, 23, 24), # little
    )
    _HEAD_IN_WAIST = np.array([0.15, 0.0, 0.45], dtype=float)

    def __init__(
        self,
        hold_s: float = 3.0,
        ring_pinch_ratio: float = 0.38,
        straightness_ratio: float = 0.88,
        facing_cosine: float = 0.60,
    ):
        if hold_s <= 0.0:
            raise ValueError("hold_s must be positive")
        if not 0.0 < ring_pinch_ratio < 1.0:
            raise ValueError("ring_pinch_ratio must be between zero and one")
        if not 0.0 < straightness_ratio <= 1.0:
            raise ValueError("straightness_ratio must be between zero and one")
        if not -1.0 <= facing_cosine <= 1.0:
            raise ValueError("facing_cosine must be between minus one and one")
        self.hold_s = hold_s
        self.ring_pinch_ratio = ring_pinch_ratio
        self.straightness_ratio = straightness_ratio
        self.facing_cosine = facing_cosine
        self.reset()

    def reset(self) -> None:
        self._pose_started_at = None
        self._latched = False

    @staticmethod
    def _valid_hand_array(hand_positions) -> bool:
        return (
            isinstance(hand_positions, np.ndarray)
            and hand_positions.shape == (25, 3)
            and np.all(np.isfinite(hand_positions))
        )

    def _fingers_are_straight(self, hand_positions: np.ndarray) -> bool:
        for indices in self._STRAIGHT_FINGERS:
            joints = hand_positions[list(indices)]
            path_length = np.linalg.norm(np.diff(joints, axis=0), axis=1).sum()
            if path_length <= 1e-6:
                return False
            chord_length = np.linalg.norm(joints[-1] - joints[0])
            if chord_length / path_length < self.straightness_ratio:
                return False
        return True

    def _ring_is_pinched(self, hand_positions: np.ndarray) -> bool:
        palm_length = np.linalg.norm(
            hand_positions[self._MIDDLE_METACARPAL] - hand_positions[self._WRIST]
        )
        if palm_length <= 0.03:
            return False
        pinch_distance = np.linalg.norm(
            hand_positions[self._THUMB_TIP] - hand_positions[self._RING_TIP]
        )
        return pinch_distance <= self.ring_pinch_ratio * palm_length

    def _palm_faces_headset(self, wrist_pose, side: str) -> bool:
        wrist_pose = np.asarray(wrist_pose)
        if wrist_pose.shape != (4, 4) or not np.all(np.isfinite(wrist_pose)):
            return False
        wrist_to_head = self._HEAD_IN_WAIST - wrist_pose[:3, 3]
        distance = np.linalg.norm(wrist_to_head)
        if distance <= 1e-6:
            return False
        wrist_to_head /= distance

        # Unitree wrist convention: left +Y points palm->back, while right +Y
        # points back->palm.  Therefore the outward palm normals have opposite signs.
        palm_normal = wrist_pose[:3, 1] * (-1.0 if side == "left" else 1.0)
        normal_length = np.linalg.norm(palm_normal)
        if normal_length <= 1e-6:
            return False
        palm_normal /= normal_length
        return float(np.dot(palm_normal, wrist_to_head)) >= self.facing_cosine

    def hand_matches(self, hand_positions, wrist_pose, side: str) -> bool:
        if side not in ("left", "right"):
            raise ValueError(f"Unknown hand side: {side}")
        if not self._valid_hand_array(hand_positions):
            return False
        return (
            self._ring_is_pinched(hand_positions)
            and self._fingers_are_straight(hand_positions)
            and self._palm_faces_headset(wrist_pose, side)
        )

    def update(
        self,
        left_hand_positions,
        right_hand_positions,
        left_wrist_pose,
        right_wrist_pose,
        now: float | None = None,
    ) -> bool:
        """Consume one sample and return ``True`` after a continuous valid hold."""
        now = time.monotonic() if now is None else float(now)
        pose_matches = self.hand_matches(
            left_hand_positions, left_wrist_pose, "left"
        ) and self.hand_matches(right_hand_positions, right_wrist_pose, "right")

        if not pose_matches:
            self._pose_started_at = None
            self._latched = False
            return False
        if self._latched:
            return False
        if self._pose_started_at is None:
            self._pose_started_at = now
            return False
        if now - self._pose_started_at < self.hold_s:
            return False

        self._latched = True
        return True


def camera_configs(config: Mapping) -> dict[str, Mapping]:
    """Return camera entries from a TeleImage config, ignoring global sections."""
    return {
        name: value
        for name, value in config.items()
        if isinstance(value, Mapping)
        and ("enable_zmq" in value or "enable_webrtc" in value)
        and "image_shape" in value
    }


def resolve_camera_names(config: Mapping, selection: str) -> list[str]:
    """Resolve ``auto``, ``all``, or a comma-separated camera selection."""
    available = camera_configs(config)
    enabled = [
        name
        for name, value in available.items()
        if value.get("enable_zmq", False) or value.get("enable_webrtc", False)
    ]
    if not enabled:
        raise ValueError("The image server did not report any enabled cameras.")

    selection = selection.strip()
    if selection == "auto":
        return ["head_camera" if "head_camera" in enabled else enabled[0]]
    if selection == "all":
        return enabled

    requested = list(dict.fromkeys(name.strip() for name in selection.split(",") if name.strip()))
    if not requested:
        raise ValueError("--xr-cameras must be 'auto', 'all', or a comma-separated camera list.")
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown camera(s): {', '.join(unknown)}. Available: {', '.join(available)}"
        )
    disabled = [
        name
        for name in requested
        if not available[name].get("enable_zmq", False)
        and not available[name].get("enable_webrtc", False)
    ]
    if disabled:
        raise ValueError(f"Selected camera(s) are disabled: {', '.join(disabled)}")
    return requested


class CameraDisplay:
    """Compose selected camera frames into the fixed canvas expected by Vuer."""

    def __init__(self, config: Mapping, camera_names: list[str], layout: str):
        if layout not in ("single", "side-by-side"):
            raise ValueError(f"Unsupported camera layout: {layout}")
        if not camera_names:
            raise ValueError("At least one camera must be selected.")

        self.config = camera_configs(config)
        self.camera_names = tuple(camera_names)
        self.layout = layout
        self._active_index = 0
        self._lock = threading.Lock()

        reference = self.config[self.camera_names[0]]
        shape = reference.get("image_shape")
        if not isinstance(shape, (list, tuple)) or len(shape) < 2:
            raise ValueError(f"Invalid image_shape for {self.camera_names[0]}: {shape!r}")
        self.height, self.width = int(shape[0]), int(shape[1])
        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"Invalid image_shape for {self.camera_names[0]}: {shape!r}")
        self.binocular = bool(reference.get("binocular", False))

    @property
    def active_camera(self) -> str:
        with self._lock:
            return self.camera_names[self._active_index]

    @property
    def displayed_camera_names(self) -> tuple[str, ...]:
        if self.layout == "side-by-side":
            return self.camera_names
        return (self.active_camera,)

    def cycle(self) -> str:
        """Select and return the next camera in single-camera layout."""
        with self._lock:
            if self.layout == "single":
                self._active_index = (self._active_index + 1) % len(self.camera_names)
            return self.camera_names[self._active_index]

    def requires_local_zmq(self) -> bool:
        """Whether the selected view cannot use a single direct WebRTC stream."""
        return self.layout == "side-by-side" or len(self.camera_names) > 1

    def validate_local_zmq(self) -> None:
        unavailable = [
            name for name in self.camera_names if not self.config[name].get("enable_zmq", False)
        ]
        if unavailable:
            raise ValueError(
                "Multi-camera composition and camera cycling require ZMQ for every selected "
                f"camera; ZMQ is disabled for: {', '.join(unavailable)}"
            )

    @staticmethod
    def _array(frame):
        if frame is None:
            return None
        image = getattr(frame, "bgr", frame)
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            return None
        return image

    @staticmethod
    def _letterbox(image: np.ndarray | None, height: int, width: int) -> np.ndarray:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        if image is None or image.size == 0:
            return canvas
        scale = min(width / image.shape[1], height / image.shape[0])
        resized_width = max(1, int(round(image.shape[1] * scale)))
        resized_height = max(1, int(round(image.shape[0] * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        x = (width - resized_width) // 2
        y = (height - resized_height) // 2
        canvas[y:y + resized_height, x:x + resized_width] = resized
        return canvas

    def _mono_view(self, name: str, frame) -> np.ndarray | None:
        image = self._array(frame)
        if image is not None and self.config[name].get("binocular", False):
            image = image[:, : image.shape[1] // 2]
        return image

    @staticmethod
    def _label(image: np.ndarray, name: str) -> None:
        label = name.removesuffix("_camera").replace("_", " ")
        cv2.putText(
            image,
            label,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def compose(self, frames: Mapping[str, object]) -> np.ndarray:
        """Return a BGR canvas with the exact shape configured for Vuer."""
        if self.layout == "single":
            name = self.active_camera
            image = self._array(frames.get(name))
            source_is_binocular = bool(self.config[name].get("binocular", False))
            if self.binocular and source_is_binocular:
                return self._letterbox(image, self.height, self.width)
            if self.binocular:
                eye = self._letterbox(image, self.height, self.width // 2)
                return np.concatenate((eye, eye), axis=1)
            return self._letterbox(self._mono_view(name, image), self.height, self.width)

        dashboard_width = self.width // 2 if self.binocular else self.width
        dashboard = np.zeros((self.height, dashboard_width, 3), dtype=np.uint8)
        count = len(self.camera_names)
        for index, name in enumerate(self.camera_names):
            x0 = index * dashboard_width // count
            x1 = (index + 1) * dashboard_width // count
            tile = self._letterbox(self._mono_view(name, frames.get(name)), self.height, x1 - x0)
            self._label(tile, name)
            dashboard[:, x0:x1] = tile
        if self.binocular:
            return np.concatenate((dashboard, dashboard), axis=1)
        return dashboard
