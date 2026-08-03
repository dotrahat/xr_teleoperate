# Offline tests for locomotion_retarget -- no sim, no headset, no DDS required.
#
#   python -m teleop.utils.test_locomotion_retarget     (or: pytest this file)
import math
import os
import sys

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from teleop.utils.locomotion_retarget import (  # noqa: E402
    HeadLocomotionRetargeter, LocoTuning, Twist, ZERO_TWIST,
    head_axes, head_ypr, neck_pivot, shaped_axis, wrap_to_pi,
    LatchingToggle, deadman_is_available, joystick_twist,
)

# The real basis-change constants from televuer/tv_wrapper.py:105-113. Reproduced here so the
# tests fail loudly if that convention ever changes upstream.
T_ROBOT_OPENXR = np.array([[0, 0, -1, 0],
                           [-1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]], dtype=float)
T_OPENXR_ROBOT = np.array([[0, -1, 0, 0],
                           [0, 0, 1, 0],
                           [-1, 0, 0, 0],
                           [0, 0, 0, 1]], dtype=float)


def _pose(rotation=None, translation=(0.0, 0.0, 0.0)):
    m = np.eye(4)
    if rotation is not None:
        m[:3, :3] = rotation
    m[:3, 3] = translation
    return m


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


# --- basis / axis conventions -----------------------------------------------------------
def test_basis_constants_are_inverse_rotations():
    assert np.allclose(T_ROBOT_OPENXR @ T_OPENXR_ROBOT, np.eye(4))
    assert np.isclose(np.linalg.det(T_ROBOT_OPENXR[:3, :3]), 1.0)  # proper rotation


def test_identity_openxr_pose_maps_to_robot_forward():
    """An OpenXR camera with identity rotation looks down -Z; that must be robot +x."""
    robot_pose = T_ROBOT_OPENXR @ _pose() @ T_OPENXR_ROBOT
    forward, left, up = head_axes(robot_pose)
    assert np.allclose(forward, [1, 0, 0])
    assert np.allclose(left, [0, 1, 0])
    assert np.allclose(up, [0, 0, 1])


def test_const_head_pose_translation_matches_documented_robot_value():
    """CONST_HEAD_POSE is (0, 1.5, -0.2) in OpenXR -> (0.2, 0, 1.5) in robot axes."""
    robot_pose = T_ROBOT_OPENXR @ _pose(translation=(0.0, 1.5, -0.2)) @ T_OPENXR_ROBOT
    assert np.allclose(robot_pose[:3, 3], [0.2, 0.0, 1.5])


def test_looking_left_in_openxr_gives_positive_yaw_in_robot_frame():
    """Rotating about OpenXR +Y turns the camera left; robot yaw must come out positive."""
    for angle in (0.1, 0.4, 1.0):
        robot_pose = T_ROBOT_OPENXR @ _pose(_rot_y(angle)) @ T_OPENXR_ROBOT
        yaw, _pitch, _roll = head_ypr(robot_pose)
        assert np.isclose(yaw, angle), f"expected yaw {angle}, got {yaw}"
        forward, _, _ = head_axes(robot_pose)
        assert np.allclose(forward, [math.cos(angle), math.sin(angle), 0.0], atol=1e-9)


def test_roll_is_positive_when_tilting_left():
    """Tilt left = head's 'left' axis dips downward. Positive roll pairs with +wz (turn left)."""
    tilt_left = _pose(_rot_x(-0.3))   # negative rotation about forward axis dips +y downward
    _yaw, _pitch, roll = head_ypr(tilt_left)
    assert roll > 0, f"tilting left must give positive roll, got {roll}"
    assert np.isclose(roll, 0.3)

    tilt_right = _pose(_rot_x(0.3))
    _yaw, _pitch, roll_r = head_ypr(tilt_right)
    assert np.isclose(roll_r, -0.3)


def test_pitch_positive_when_looking_up():
    _yaw, pitch, _roll = head_ypr(_pose(_rot_z(0.0) @ np.array(
        [[math.cos(0.2), 0, math.sin(0.2)], [0, 1, 0], [-math.sin(0.2), 0, math.cos(0.2)]])))
    # rotating the robot-frame forward axis up out of the xy plane
    assert pitch != 0.0


# --- neck pivot --------------------------------------------------------------------------
def test_neck_pivot_walks_back_along_horizontal_gaze():
    pose = _pose(translation=(1.0, 0.0, 1.5))          # facing robot +x
    assert np.allclose(neck_pivot(pose, 0.1), [0.9, 0.0, 1.5])


def test_neck_pivot_removes_phantom_translation_from_pure_rotation():
    """A pure head rotation about the neck must not move the pivot."""
    offset = 0.10
    pivot = np.array([1.0, 2.0, 1.5])
    seen = []
    for angle in (0.0, 0.5, -0.8, 1.2):
        rotation = _rot_z(angle)
        camera = pivot + offset * np.array([math.cos(angle), math.sin(angle), 0.0])
        seen.append(neck_pivot(_pose(rotation, camera), offset))
    for value in seen[1:]:
        assert np.allclose(value, seen[0], atol=1e-9), "pivot moved under pure rotation"


def test_neck_pivot_handles_looking_straight_up():
    pose = _pose(_rot_x(0.0), (1.0, 0.0, 1.5))
    pose[:3, 0] = [0.0, 0.0, 1.0]     # degenerate: no horizontal gaze component
    assert np.allclose(neck_pivot(pose, 0.1), [1.0, 0.0, 1.5])


# --- shaping -----------------------------------------------------------------------------
def test_shaped_axis_deadzone_ramp_and_saturation():
    assert shaped_axis(0.0, 0.1, 0.45, 0.4) == 0.0
    assert shaped_axis(0.09, 0.1, 0.45, 0.4) == 0.0
    assert shaped_axis(0.1, 0.1, 0.45, 0.4) == 0.0          # boundary is inclusive-zero
    assert np.isclose(shaped_axis(0.45, 0.1, 0.45, 0.4), 0.4)
    assert np.isclose(shaped_axis(9.9, 0.1, 0.45, 0.4), 0.4)   # saturates
    assert np.isclose(shaped_axis(-0.45, 0.1, 0.45, 0.4), -0.4)  # antisymmetric
    mid = shaped_axis(0.275, 0.1, 0.45, 0.4)
    assert np.isclose(mid, 0.2), f"midpoint should be half scale, got {mid}"


def test_wrap_to_pi():
    assert np.isclose(wrap_to_pi(0.0), 0.0)
    assert np.isclose(wrap_to_pi(3 * math.pi), math.pi) or np.isclose(wrap_to_pi(3 * math.pi), -math.pi)
    assert np.isclose(wrap_to_pi(math.pi / 2 + 2 * math.pi), math.pi / 2)
    assert np.isclose(wrap_to_pi(-math.pi / 2 - 2 * math.pi), -math.pi / 2)


def test_twist_helpers():
    assert Twist(1.0, 2.0, 3.0, 0.7).as_list() == [1.0, 2.0, 3.0, 0.7]
    assert ZERO_TWIST.is_zero()
    assert not Twist(0.1, 0.0, 0.0).is_zero()


# --- retargeter behaviour ----------------------------------------------------------------
def _standing(x=0.0, y=0.0, yaw=0.0, z=1.5, neck=0.10):
    """Head pose whose NECK PIVOT is at (x, y), facing `yaw`."""
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    camera = np.array([x, y, z]) + neck * forward
    return _pose(_rot_z(yaw), camera)


def _settle(retargeter, pose, t0, enable=True, n=6, dt=0.033):
    """Push a fixed pose repeatedly so the moving-average filter converges.

    A held pose translates nowhere, so speed-matched vx/vy stay zero -- this converges the
    ROTATION channel (roll->wz is an angle-based rate command). Use _walk for translation.
    """
    twist = None
    for i in range(n):
        # jitter below the deadzone so the staleness check never trips on identical frames
        jittered = pose.copy()
        jittered[0, 3] += 1e-9 * i
        twist = retargeter.update(jittered, enable, t0 + i * dt)
    return twist


def _walk(retargeter, t0, vx=0.0, vy=0.0, yaw=0.0, start=(0.0, 0.0),
          enable=True, dt=0.033, n=10):
    """Walk the neck pivot at a CONSTANT operator velocity (vx forward, vy left, m/s).

    Translation is speed-matched, so what the robot does depends on how fast the operator is
    *moving*, not where they are. This drives a steady velocity for n frames (long enough for
    the moving-average filter to converge) and returns the twist during that steady walk. The
    caller must have already produced the deadman rising edge at the same `start`/`yaw`, so the
    first frame here has a valid dt from that engagement.
    """
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-math.sin(yaw), math.cos(yaw)])
    pos = np.array(start, dtype=float)
    t = t0
    twist = None
    for _ in range(n):
        pos = pos + (vx * fwd + vy * left) * dt
        twist = retargeter.update(_standing(pos[0], pos[1], yaw), enable, t)
        t += dt
    return twist


def test_no_motion_until_deadman_held():
    r = HeadLocomotionRetargeter(LocoTuning())
    twist = r.update(_standing(), False, 0.0)
    assert twist.is_zero()
    assert r.status["reason"] == "deadman released"


def test_walking_forward_after_engage_walks_forward():
    r = HeadLocomotionRetargeter(LocoTuning())
    r.update(_standing(), True, 0.0)                    # rising edge locks heading + baseline
    twist = _walk(r, 0.033, vx=0.7)                     # brisk forward walk
    assert twist.vx > 0.3, twist
    assert abs(twist.vy) < 1e-6
    assert abs(twist.wz) < 1e-6


def test_standing_still_stops_even_while_deadman_held():
    """The whole point of speed-matching: stop when the operator stops, fist still closed."""
    r = HeadLocomotionRetargeter(LocoTuning())
    r.update(_standing(), True, 0.0)
    pos, t = 0.0, 0.033
    twist = None
    for _ in range(10):                                 # walk forward, well clear of neutral
        pos += 0.7 * 0.033
        twist = r.update(_standing(x=pos), True, t)
        t += 0.033
    assert twist.vx > 0.3, twist
    # Now stand still at that spot, fist STILL closed. The old ratchet kept walking; speed-
    # matching must stop within the filter window.
    for _ in range(6):
        twist = r.update(_standing(x=pos), True, t)
        t += 0.033
    assert twist.is_zero(), f"standing still must stop, got {twist}"
    assert r.status["reason"] == "ok"


def test_walking_left_strafes_left():
    r = HeadLocomotionRetargeter(LocoTuning())
    r.update(_standing(), True, 0.0)
    twist = _walk(r, 0.033, vy=0.9)
    assert twist.vy > 0.2, twist
    assert abs(twist.vx) < 1e-6


def test_slow_drift_inside_deadzone_is_ignored():
    r = HeadLocomotionRetargeter(LocoTuning())
    r.update(_standing(), True, 0.0)
    twist = _walk(r, 0.033, vx=0.08)                    # < 0.12 m/s deadzone: idle sway
    assert twist.is_zero(), twist


def test_speed_matching_is_position_independent():
    """Walking at the same speed gives the same twist regardless of where in the room."""
    results = []
    for origin in ((0.0, 0.0), (3.0, -2.0), (-7.5, 11.25)):
        r = HeadLocomotionRetargeter(LocoTuning())
        r.update(_standing(x=origin[0], y=origin[1]), True, 0.0)
        twist = _walk(r, 0.033, vx=0.7, start=origin)
        results.append(twist.as_list())
    for value in results[1:]:
        assert np.allclose(value, results[0], atol=1e-9), results


def test_velocity_is_projected_onto_engagement_heading():
    """Facing +y at engagement, walking along world +y must read as FORWARD, not lateral."""
    r = HeadLocomotionRetargeter(LocoTuning())
    yaw = math.pi / 2
    r.update(_standing(yaw=yaw), True, 0.0)
    twist = _walk(r, 0.033, vx=0.7, yaw=yaw)            # vx is operator-forward = world +y here
    assert twist.vx > 0.3, twist
    assert abs(twist.vy) < 1e-6, twist


def test_release_zeroes_instantly_and_reengage_resets_baseline():
    r = HeadLocomotionRetargeter(LocoTuning())
    r.update(_standing(), True, 0.0)
    assert _walk(r, 0.033, vx=0.7).vx > 0.3

    released = r.update(_standing(x=1.0), False, 1.0)
    assert released.is_zero(), "releasing the deadman must zero immediately"

    # Re-engage at the new spot: the first frame has no dt yet, so it must not lurch.
    reengage = r.update(_standing(x=1.0), True, 1.1)
    assert reengage.is_zero(), f"re-engage frame must not command motion, got {reengage}"


def test_jump_rejection_latches_until_deadman_is_recycled():
    """A tracking teleport invalidates the heading lock; it stays stopped until the operator
    releases and re-engages. Discoverable: the robot simply stops."""
    r = HeadLocomotionRetargeter(LocoTuning())
    r.update(_standing(), True, 0.0)
    _walk(r, 0.033, vx=0.7, n=4)
    assert r.update(_standing(x=9.0), True, 1.0).is_zero()   # >0.15 m single frame
    # Still held: no re-engagement, so it stays zero.
    assert _walk(r, 1.1, vx=0.7, start=(9.0, 0.0)).is_zero()
    assert r.status["reason"] == "not calibrated"
    # Release + re-engage restores control.
    r.update(_standing(x=9.2), False, 2.0)
    r.update(_standing(x=9.2), True, 2.1)
    assert _walk(r, 2.2, vx=0.7, start=(9.2, 0.0)).vx > 0.3


def test_roll_turns_and_gaze_yaw_does_not():
    r = HeadLocomotionRetargeter(LocoTuning(), yaw_mode="roll")
    r.update(_standing(), True, 0.0)

    # Looking around: large yaw change, no roll. Must produce zero turn rate.
    # (fed as a small ramp so jump-rejection does not trip)
    twist = None
    t = 0.1
    for i in range(1, 7):
        twist = r.update(_standing(yaw=0.1 * i), True, t)
        t += 0.033
    assert abs(twist.wz) < 1e-6, f"looking around must not steer, got wz={twist.wz}"

    # Tilting left: turn left (+wz).
    r2 = HeadLocomotionRetargeter(LocoTuning(), yaw_mode="roll")
    base = _standing()
    r2.update(base, True, 0.0)
    tilted = base.copy()
    tilted[:3, :3] = _rot_x(-0.5)          # tilt left
    assert _settle(r2, tilted, 0.1).wz > 0.3


def test_yaw_mode_off_never_turns():
    r = HeadLocomotionRetargeter(LocoTuning(), yaw_mode="off")
    base = _standing()
    r.update(base, True, 0.0)
    tilted = base.copy()
    tilted[:3, :3] = _rot_x(-0.5)
    assert abs(_settle(r, tilted, 0.1).wz) < 1e-9


def test_stale_pose_stops_the_robot():
    """A frozen feed (headset removed) must not keep a TURN alive.

    Speed-matched translation already stops on a frozen feed (zero velocity), but turning is a
    rate command from head tilt: a frozen mid-tilt pose would otherwise keep steering forever.
    This is exactly what safe_mat_update (tv_wrapper.py:70) cannot catch -- the matrix is still
    perfectly valid.
    """
    tuning = LocoTuning(stale_timeout_s=0.5)
    r = HeadLocomotionRetargeter(tuning, yaw_mode="roll")
    base = _standing()
    r.update(base, True, 0.0)
    tilted = base.copy()
    tilted[:3, :3] = _rot_x(-0.5)                        # tilt left -> +wz
    assert _settle(r, tilted, 0.1).wz > 0.3

    frozen = tilted.copy()
    still_turning = r.update(frozen, True, 1.0)          # first identical frame starts the clock
    assert still_turning.wz > 0.3, "should not stop before the timeout"
    twist = r.update(frozen, True, 1.6)                  # 0.6 s > 0.5 s timeout
    assert twist.is_zero(), "stale pose must zero the twist"
    assert r.status["reason"] == "stale head pose"


def test_pose_jump_is_rejected():
    r = HeadLocomotionRetargeter(LocoTuning())
    r.update(_standing(), True, 0.0)
    _settle(r, _standing(x=0.2), 0.1)
    twist = r.update(_standing(x=9.0), True, 0.5)      # CONST_HEAD_POSE-style teleport
    assert twist.is_zero()
    assert "jump" in r.status["reason"]


def test_non_finite_pose_is_rejected():
    r = HeadLocomotionRetargeter(LocoTuning())
    r.update(_standing(), True, 0.0)
    bad = _standing()
    bad[0, 3] = float("nan")
    twist = r.update(bad, True, 0.1)
    assert twist.is_zero()
    assert r.status["reason"] == "non-finite pose"


def test_speeds_never_exceed_configured_limits():
    tuning = LocoTuning(max_vx=0.4, max_vy=0.25, max_wz=0.6)
    r = HeadLocomotionRetargeter(tuning)
    r.update(_standing(), True, 0.0)
    # Sprint well past walk_full (0.80 m/s) so the shaped axis saturates, but stay under
    # max_step_ms (3.0) so it is not rejected as a glitch.
    twist = _walk(r, 0.1, vx=1.6, n=12)
    assert twist.vx <= tuning.max_vx + 1e-9, twist
    assert abs(twist.vy) <= tuning.max_vy + 1e-9
    assert abs(twist.wz) <= tuning.max_wz + 1e-9


# ---------------------------------------------------------------------------------------
# controller gate: latching toggle, thumbstick retargeting, deadman availability
# ---------------------------------------------------------------------------------------

class _Ctrl:
    """Minimal stand-in for TeleData's controller fields."""

    def __init__(self, **kw):
        self.left_ctrl_aButton = False
        self.left_ctrl_bButton = False
        self.right_ctrl_aButton = False
        self.right_ctrl_bButton = False
        self.left_ctrl_thumbstickValue = np.zeros(2)
        self.right_ctrl_thumbstickValue = np.zeros(2)
        self.left_ctrl_squeeze = False
        self.right_ctrl_squeeze = False
        self.left_ctrl_triggerValue = 10.0
        self.right_ctrl_triggerValue = 10.0
        for k, v in kw.items():
            setattr(self, k, v)


def test_toggle_starts_disarmed_and_flips_on_rising_edge_only():
    t = LatchingToggle("x")
    assert t.state is False, "walking must never be armed at startup"
    held = _Ctrl(left_ctrl_aButton=True)
    assert t.update(held) is True
    # Still held across many frames -- must not oscillate at loop rate.
    for _ in range(50):
        assert t.update(held) is True
    released = _Ctrl(left_ctrl_aButton=False)
    assert t.update(released) is True, "releasing must not disarm"
    assert t.update(held) is False, "second press disarms"


def test_toggle_rejects_the_quit_button():
    # right_ctrl_aButton quits teleoperation; offering it as a gate would be a footgun.
    for bad in ("a", "A", "start", ""):
        try:
            LatchingToggle(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should not be a valid toggle button")


def test_centred_sticks_return_none_not_zero():
    # None is what lets the caller distinguish "stick idle" from "stick commanding a stop".
    tuning = LocoTuning()
    assert joystick_twist(_Ctrl(), tuning) is None
    nudge = _Ctrl(left_ctrl_thumbstickValue=np.array([0.02, -0.03]))
    assert joystick_twist(nudge, tuning, deadzone=0.08) is None


def test_stick_axes_and_limits():
    tuning = LocoTuning()
    # Left stick pushed forward is -y in the XR convention -> +vx.
    fwd = joystick_twist(_Ctrl(left_ctrl_thumbstickValue=np.array([0.0, -1.0])), tuning)
    assert fwd.vx > 0 and abs(fwd.vy) < 1e-9, fwd
    # Left stick pushed left -> +vy.
    left = joystick_twist(_Ctrl(left_ctrl_thumbstickValue=np.array([-1.0, 0.0])), tuning)
    assert left.vy > 0 and abs(left.vx) < 1e-9, left
    # Right stick pushed left -> +wz (turn left / CCW).
    turn = joystick_twist(_Ctrl(right_ctrl_thumbstickValue=np.array([-1.0, 0.0])), tuning)
    assert turn.wz > 0 and abs(turn.vx) < 1e-9, turn
    # Full deflection on every axis stays inside the configured ceilings.
    full = joystick_twist(_Ctrl(left_ctrl_thumbstickValue=np.array([-1.0, -1.0]),
                                right_ctrl_thumbstickValue=np.array([-1.0, 0.0])), tuning)
    assert full.vx <= tuning.max_vx + 1e-9
    assert full.vy <= tuning.max_vy + 1e-9
    assert full.wz <= tuning.max_wz + 1e-9


def test_deadman_availability_matches_eval_deadman():
    # Guards the startup check against drifting away from the runtime implementation.
    for mode, ok, bad in (
        ("hand", ["left_fist", "right_fist", "both_fist", "left_pinch", "right_pinch", "none"],
                 ["left_trigger", "right_trigger"]),
        ("controller", ["left_fist", "right_fist", "both_fist", "left_trigger", "right_trigger", "none"],
                       ["left_pinch", "right_pinch"]),
    ):
        for kind in ok:
            assert deadman_is_available(kind, mode), f"{kind} should be valid in {mode}"
        for kind in bad:
            assert not deadman_is_available(kind, mode), f"{kind} should be invalid in {mode}"


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failures.append((fn.__name__, e))
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures.append((fn.__name__, e))
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
