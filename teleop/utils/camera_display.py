"""Camera selection and headset-canvas composition for XR teleoperation."""

from __future__ import annotations

import threading
from collections.abc import Mapping

import cv2
import numpy as np


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
