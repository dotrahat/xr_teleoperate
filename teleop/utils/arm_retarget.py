"""Shoulder-relative wrist-position retargeting for the G1-23 + BrainCo setup."""

from dataclasses import dataclass

import numpy as np


# Geometry measured from g1_23dof_mode_10_with_brainco.urdf.
G1_23_BRAINCO_WRIST_OFFSET_M = 0.121
G1_23_FORWARD_SHOULDER_TO_WRIST_M = 0.41832220958492194

# The head-to-waist translation remains owned by TeleVuerWrapper. It is repeated here only
# to express experimental human shoulder anchors in the more intuitive head-relative frame.
HEAD_TO_WAIST_BODY_OFFSET_M = np.array([0.15, 0.0, 0.45], dtype=float)

# Shoulder-pitch joint origins in the fixed-pelvis frame used by G1_23_ArmIK.
G1_23_LEFT_SHOULDER_M = np.array([-0.0000072, 0.10022, 0.29178], dtype=float)
G1_23_RIGHT_SHOULDER_M = np.array([-0.0000072, -0.10021, 0.29178], dtype=float)

# Initial calibration: applying the unchanged body offset maps these human anchors exactly
# onto the corresponding robot shoulder. These values are intentionally configurable.
DEFAULT_LEFT_HUMAN_SHOULDER_HEAD_M = G1_23_LEFT_SHOULDER_M - HEAD_TO_WAIST_BODY_OFFSET_M
DEFAULT_RIGHT_HUMAN_SHOULDER_HEAD_M = G1_23_RIGHT_SHOULDER_M - HEAD_TO_WAIST_BODY_OFFSET_M


def parse_xyz(value):
    """Parse a CLI xyz value without coupling this module to argparse."""
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = value
    if len(parts) != 3:
        raise ValueError(f"expected x,y,z, got {value!r}")
    try:
        xyz = np.asarray([float(part) for part in parts], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected numeric x,y,z, got {value!r}") from exc
    if not np.all(np.isfinite(xyz)):
        raise ValueError(f"shoulder anchor must be finite, got {value!r}")
    return xyz


@dataclass(frozen=True)
class G1_23_BraincoArmRetargeter:
    """Map tracked wrist translations between calibrated human and robot shoulders."""

    human_arm_length_m: float = 0.48
    left_human_shoulder_head_m: tuple = tuple(DEFAULT_LEFT_HUMAN_SHOULDER_HEAD_M)
    right_human_shoulder_head_m: tuple = tuple(DEFAULT_RIGHT_HUMAN_SHOULDER_HEAD_M)

    def __post_init__(self):
        if not np.isfinite(self.human_arm_length_m) or self.human_arm_length_m <= 0.0:
            raise ValueError("human_arm_length_m must be a positive finite value")
        object.__setattr__(
            self,
            "left_human_shoulder_head_m",
            tuple(parse_xyz(self.left_human_shoulder_head_m)),
        )
        object.__setattr__(
            self,
            "right_human_shoulder_head_m",
            tuple(parse_xyz(self.right_human_shoulder_head_m)),
        )

    @property
    def scale(self):
        return G1_23_FORWARD_SHOULDER_TO_WRIST_M / self.human_arm_length_m

    def _retarget_pose(self, wrist_pose, human_shoulder_head_m, robot_shoulder_m):
        wrist_pose = np.asarray(wrist_pose, dtype=float)
        if wrist_pose.shape != (4, 4):
            raise ValueError(f"wrist pose must have shape (4, 4), got {wrist_pose.shape}")

        retargeted = wrist_pose.copy()
        human_shoulder_waist_m = (
            np.asarray(human_shoulder_head_m, dtype=float) + HEAD_TO_WAIST_BODY_OFFSET_M
        )
        retargeted[:3, 3] = robot_shoulder_m + self.scale * (
            wrist_pose[:3, 3] - human_shoulder_waist_m
        )
        return retargeted

    def retarget(self, left_wrist_pose, right_wrist_pose):
        return (
            self._retarget_pose(
                left_wrist_pose,
                self.left_human_shoulder_head_m,
                G1_23_LEFT_SHOULDER_M,
            ),
            self._retarget_pose(
                right_wrist_pose,
                self.right_human_shoulder_head_m,
                G1_23_RIGHT_SHOULDER_M,
            ),
        )
