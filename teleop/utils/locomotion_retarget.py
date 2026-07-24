# Head/body pose -> locomotion twist.
#
# Pure math: no threads, no IO, no vuer, no DDS, so it can be unit-tested offline against
# hand-built SE(3) matrices. Transport and timing live in locomotion_publisher.py.
#
# The operator's *displacement from a calibrated neutral* sets velocity. That covers both
# leaning and physically stepping -- Quest 3's inside-out SLAM makes head translation a true
# room-scale position, so stepping forward and STAYING there is a sustained walk command.
# Releasing the deadman re-calibrates, which gives a ratchet: step -> release -> walk back ->
# re-engage. A small room therefore yields unlimited robot travel.
#
# Frame: tv_wrapper hands us head_pose already in ROBOT convention (x front, y left, z up),
# via Brobot_world_head = T_ROBOT_OPENXR @ Bxr_world_head @ T_OPENXR_ROBOT (tv_wrapper.py:282).
# Because T_OPENXR_ROBOT's first column is (0,0,-1) -- the OpenXR camera's forward axis --
# the columns of the resulting rotation are, in robot axes:
#     column 0 = gaze forward     column 1 = head left     column 2 = head up
import math
from dataclasses import dataclass, field

import numpy as np

from teleop.utils.weighted_moving_filter import WeightedMovingFilter


@dataclass(frozen=True)
class Twist:
    vx: float
    vy: float
    wz: float
    height: float = 0.8

    def as_list(self):
        return [self.vx, self.vy, self.wz, self.height]

    def is_zero(self, eps=1e-6):
        return abs(self.vx) < eps and abs(self.vy) < eps and abs(self.wz) < eps


ZERO_TWIST = Twist(0.0, 0.0, 0.0, 0.8)


@dataclass
class LocoTuning:
    """Defaults are STEP-scaled, not lean-scaled: the operator physically walks."""
    max_vx: float = 0.40
    max_vy: float = 0.25
    max_wz: float = 0.60

    step_deadzone_m: float = 0.10   # absorbs the ~+-5 cm lateral sway of natural walking
    step_full_m: float = 0.45       # displacement giving full commanded speed

    roll_deadzone_rad: float = 0.17  # 10 deg
    roll_full_rad: float = 0.52      # 30 deg

    # The XR camera sits ~10 cm ahead of the neck pivot, so a pure head *rotation* produces
    # ~10 cm of camera *translation*. Without this, glancing sideways fakes a lateral step.
    neck_offset_m: float = 0.10

    # Safety thresholds -- see the sanitisation ladder in update().
    stale_timeout_s: float = 0.50
    jump_pos_m: float = 0.15
    jump_yaw_rad: float = 0.79       # 45 deg

    filter_weights: tuple = (0.5, 0.3, 0.2)
    height: float = 0.8


# --- free functions (individually unit-testable) ---------------------------------------
def head_axes(head_pose):
    """(forward, left, up) unit axes of the head, expressed in robot world axes."""
    r = np.asarray(head_pose)[:3, :3]
    return r[:, 0], r[:, 1], r[:, 2]


def head_ypr(head_pose):
    """(yaw, pitch, roll) in radians.

    yaw   +ve = looking left        (rotation about robot +z, CCW from above)
    pitch +ve = looking up
    roll  +ve = tilting head LEFT   (left ear toward left shoulder)

    Roll is defined left-positive so it pairs intuitively with yaw rate: tilt left -> turn
    left (+wz is CCW). Tilting left tips the head's 'left' axis downward, so its z-component
    goes negative -- hence the negation.
    """
    m = np.asarray(head_pose)
    forward = m[:3, 0]
    yaw = math.atan2(forward[1], forward[0])
    pitch = math.asin(float(np.clip(forward[2], -1.0, 1.0)))
    roll = math.asin(float(np.clip(-m[2, 1], -1.0, 1.0)))
    return yaw, pitch, roll


def neck_pivot(head_pose, offset_m):
    """Head position walked back along the horizontal gaze direction to the neck pivot."""
    m = np.asarray(head_pose)
    position = m[:3, 3].astype(float)
    forward = m[:3, 0].astype(float)
    horizontal = np.array([forward[0], forward[1], 0.0])
    norm = np.linalg.norm(horizontal)
    if norm < 1e-6:
        # Looking straight up/down: gaze has no horizontal component to walk back along.
        return position
    return position - offset_m * (horizontal / norm)


def wrap_to_pi(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def shaped_axis(value, deadzone, full, limit):
    """Zero inside the deadzone, linear from deadzone->full, clamped to +-limit."""
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    span = full - deadzone
    if span <= 1e-9:
        scaled = 1.0
    else:
        scaled = min((magnitude - deadzone) / span, 1.0)
    return math.copysign(scaled * limit, value)


def eval_deadman(tele_data, kind, input_mode):
    """Hold-to-walk gate.

    Only reads fields that TeleData actually populates for the active input mode: the
    hand-tracking and controller branches of get_tele_data() are mutually exclusive
    (tv_wrapper.py:366-382 vs :412-432), so controller fields stay at defaults in hand mode.
    """
    if kind == "none":
        return True
    if input_mode == "hand":
        if kind == "left_fist":
            return bool(tele_data.left_hand_squeeze)
        if kind == "right_fist":
            return bool(tele_data.right_hand_squeeze)
        if kind == "both_fist":
            return bool(tele_data.left_hand_squeeze and tele_data.right_hand_squeeze)
        if kind == "left_pinch":
            return bool(tele_data.left_hand_pinch)
        if kind == "right_pinch":
            return bool(tele_data.right_hand_pinch)
    else:
        # triggerValue is remapped 10.0 = released -> 0.0 = fully pressed (tv_wrapper.py:417).
        if kind == "left_trigger":
            return float(tele_data.left_ctrl_triggerValue) < 3.0
        if kind == "right_trigger":
            return float(tele_data.right_ctrl_triggerValue) < 3.0
        if kind == "left_fist":
            return bool(tele_data.left_ctrl_squeeze)
        if kind == "right_fist":
            return bool(tele_data.right_ctrl_squeeze)
        if kind == "both_fist":
            return bool(tele_data.left_ctrl_squeeze and tele_data.right_ctrl_squeeze)
    raise ValueError(f"deadman {kind!r} is not available in --input-mode {input_mode}")


class HeadLocomotionRetargeter:
    """Turns head pose + a hold-to-walk gate into a locomotion twist."""

    def __init__(self, tuning=None, yaw_mode="roll"):
        self.tuning = tuning or LocoTuning()
        if yaw_mode not in ("roll", "off"):
            raise ValueError(f"yaw_mode must be 'roll' or 'off', got {yaw_mode!r}")
        self.yaw_mode = yaw_mode
        self._filter = None
        self.reset()

    def reset(self):
        self._calibrated = False
        self._p0 = None
        self._f0 = None
        self._l0 = None
        self._prev_pose = None
        self._prev_position = None
        self._prev_yaw = None
        self._last_change_time = None
        self._prev_walk_enable = False
        self._stale = False
        self._filter = WeightedMovingFilter(np.array(self.tuning.filter_weights), 3)
        self._status = {"calibrated": False, "stale": False, "reason": "reset",
                        "fwd_m": 0.0, "lat_m": 0.0, "roll_rad": 0.0}

    @property
    def status(self):
        return dict(self._status)

    def _zero(self, reason):
        self._status.update(calibrated=self._calibrated, stale=self._stale, reason=reason,
                            fwd_m=0.0, lat_m=0.0, roll_rad=0.0)
        # Never let a stale filter history bleed into the next engagement.
        self._filter = WeightedMovingFilter(np.array(self.tuning.filter_weights), 3)
        return Twist(0.0, 0.0, 0.0, self.tuning.height)

    def _calibrate(self, position, yaw):
        self._p0 = position.copy()
        self._f0 = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        self._l0 = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        self._calibrated = True

    def update(self, head_pose, walk_enable, now):
        """Return the twist to command. Every abnormal path returns a HARD zero."""
        tuning = self.tuning
        pose = np.asarray(head_pose, dtype=float)

        # 1. Finiteness.
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            self._calibrated = False
            self._prev_pose = None
            return self._zero("non-finite pose")

        # 2. Staleness. safe_mat_update (tv_wrapper.py:70) only rejects singular matrices, so
        #    a removed headset or a stalled websocket keeps returning the last good pose
        #    forever. Real tracking always jitters, so bit-identical frames mean a dead feed --
        #    and a stale lean would otherwise walk the robot away indefinitely.
        if self._prev_pose is None or not np.array_equal(pose, self._prev_pose):
            self._last_change_time = now
            self._stale = False
        elif self._last_change_time is not None and (now - self._last_change_time) > tuning.stale_timeout_s:
            self._stale = True
            self._prev_pose = pose
            self._calibrated = False
            return self._zero("stale head pose")
        self._prev_pose = pose

        position = neck_pivot(pose, tuning.neck_offset_m)
        yaw, _pitch, roll = head_ypr(pose)

        # 3. Jump rejection. Catches the CONST_HEAD_POSE fallback (tv_wrapper.py:123), which
        #    sits metres from any calibration and would otherwise command a full-scale twist.
        if self._prev_position is not None:
            moved = float(np.linalg.norm(position - self._prev_position))
            turned = abs(wrap_to_pi(yaw - self._prev_yaw))
            if moved > tuning.jump_pos_m or turned > tuning.jump_yaw_rad:
                self._prev_position = position
                self._prev_yaw = yaw
                self._calibrated = False
                return self._zero(f"pose jump ({moved:.2f} m, {math.degrees(turned):.0f} deg)")
        self._prev_position = position
        self._prev_yaw = yaw

        # 4. Deadman edges. Rising edge calibrates here and now -- this IS the ratchet.
        rising = walk_enable and not self._prev_walk_enable
        self._prev_walk_enable = walk_enable
        if rising:
            self._calibrate(position, yaw)
        if not walk_enable:
            self._calibrated = False
            return self._zero("deadman released")
        if not self._calibrated:
            return self._zero("not calibrated")

        # 5. Displacement -> vx, vy, projected onto the neutral heading.
        delta = position - self._p0
        forward_m = float(delta @ self._f0)
        lateral_m = float(delta @ self._l0)
        vx = shaped_axis(forward_m, tuning.step_deadzone_m, tuning.step_full_m, tuning.max_vx)
        vy = shaped_axis(lateral_m, tuning.step_deadzone_m, tuning.step_full_m, tuning.max_vy)

        # Roll -> wz. Deliberately NOT head yaw: the headset shows the robot's fixed camera,
        # so turning your head does not change the view and there is no natural cue to stop
        # turning. Roll is never used for looking, so gaze and steering stay fully decoupled.
        if self.yaw_mode == "roll":
            wz = shaped_axis(roll, tuning.roll_deadzone_rad, tuning.roll_full_rad, tuning.max_wz)
        else:
            wz = 0.0

        # 6. Smooth. Bypassed entirely by every zero path above.
        self._filter.add_data(np.array([vx, vy, wz]))
        smoothed = self._filter.filtered_data

        self._status.update(calibrated=True, stale=False, reason="ok",
                            fwd_m=forward_m, lat_m=lateral_m, roll_rad=roll)
        return Twist(float(smoothed[0]), float(smoothed[1]), float(smoothed[2]), tuning.height)
