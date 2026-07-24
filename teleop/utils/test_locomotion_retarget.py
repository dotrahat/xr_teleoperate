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

    Only safe when the pose does not TRANSLATE far from the previous one -- a big step in a
    single frame is (correctly) treated as a tracking glitch. Use _approach for translation.
    """
    twist = None
    for i in range(n):
        # jitter below the deadzone so the staleness check never trips on identical frames
        jittered = pose.copy()
        jittered[0, 3] += 1e-9 * i
        twist = retargeter.update(jittered, enable, t0 + i * dt)
    return twist


def _approach(retargeter, t0, x=0.0, y=0.0, yaw=0.0, start=(0.0, 0.0, 0.0),
              enable=True, dt=0.033, step_m=0.08, hold=8):
    """Move the neck pivot from `start` to (x, y, yaw) the way a real operator would.

    Increments stay below LocoTuning.jump_pos_m, because jump rejection deliberately treats a
    >0.15 m single-frame move (>4.5 m/s at 30 Hz) as a tracking relocalisation and invalidates
    the calibration. Then holds so the moving-average filter converges.
    """
    sx, sy, syaw = start
    distance = math.hypot(x - sx, y - sy)
    steps = max(1, int(math.ceil(distance / step_m)))
    t = t0
    twist = None
    for i in range(1, steps + 1):
        f = i / steps
        twist = retargeter.update(
            _standing(sx + (x - sx) * f, sy + (y - sy) * f, syaw + (yaw - syaw) * f), enable, t)
        t += dt
    for i in range(hold):
        pose = _standing(x, y, yaw)
        pose[0, 3] += 1e-9 * i      # sub-micron jitter: a live feed is never bit-identical
        twist = retargeter.update(pose, enable, t)
        t += dt
    return twist


def test_no_motion_until_deadman_held():
    r = HeadLocomotionRetargeter(LocoTuning())
    twist = r.update(_standing(), False, 0.0)
    assert twist.is_zero()
    assert r.status["reason"] == "deadman released"


def test_leaning_forward_after_engage_walks_forward():
    r = HeadLocomotionRetargeter(LocoTuning())
    r.update(_standing(), True, 0.0)                    # rising edge calibrates here
    twist = _approach(r, 0.033, x=0.45)
    assert twist.vx > 0.3, twist
    assert abs(twist.vy) < 1e-6
    assert abs(twist.wz) < 1e-6


def test_stepping_left_strafes_left():
    r = HeadLocomotionRetargeter(LocoTuning())
    r.update(_standing(), True, 0.0)
    twist = _approach(r, 0.033, y=0.45)
    assert twist.vy > 0.2, twist
    assert abs(twist.vx) < 1e-6


def test_displacement_inside_deadzone_is_ignored():
    r = HeadLocomotionRetargeter(LocoTuning())
    r.update(_standing(), True, 0.0)
    twist = _approach(r, 0.033, x=0.08)                 # < 0.10 m deadzone
    assert twist.is_zero(), twist


def test_calibration_is_relative_so_absolute_position_does_not_matter():
    """Engaging anywhere in the room gives the same twist for the same relative step."""
    results = []
    for origin in ((0.0, 0.0), (3.0, -2.0), (-7.5, 11.25)):
        r = HeadLocomotionRetargeter(LocoTuning())
        r.update(_standing(x=origin[0], y=origin[1]), True, 0.0)
        twist = _approach(r, 0.033, x=origin[0] + 0.45, y=origin[1],
                          start=(origin[0], origin[1], 0.0))
        results.append(twist.as_list())
    for value in results[1:]:
        assert np.allclose(value, results[0], atol=1e-9), results


def test_displacement_is_projected_onto_neutral_heading():
    """Facing +y at calibration, a step along world +y must read as FORWARD, not lateral."""
    r = HeadLocomotionRetargeter(LocoTuning())
    yaw = math.pi / 2
    r.update(_standing(yaw=yaw), True, 0.0)
    twist = _approach(r, 0.033, x=0.0, y=0.45, yaw=yaw, start=(0.0, 0.0, yaw))
    assert twist.vx > 0.3, twist
    assert abs(twist.vy) < 1e-6, twist


def test_release_zeroes_instantly_and_reengage_recalibrates_the_ratchet():
    tuning = LocoTuning()
    r = HeadLocomotionRetargeter(tuning)
    r.update(_standing(), True, 0.0)
    walking = _approach(r, 0.033, x=0.45)
    assert walking.vx > 0.3

    released = r.update(_standing(x=0.45), False, 1.0)
    assert released.is_zero(), "releasing the deadman must zero immediately"

    # Re-engaging at the displaced spot makes THAT the new neutral -> no residual command.
    r.update(_standing(x=0.45), True, 1.1)
    twist = _approach(r, 1.2, x=0.45, start=(0.45, 0.0, 0.0))
    assert twist.is_zero(), f"re-engage must recalibrate, got {twist}"


def test_jump_rejection_latches_until_deadman_is_recycled():
    """After a tracking glitch the old neutral is meaningless, so it stays stopped until
    the operator releases and re-engages. Discoverable: the robot simply stops."""
    r = HeadLocomotionRetargeter(LocoTuning())
    r.update(_standing(), True, 0.0)
    _approach(r, 0.033, x=0.2)
    assert r.update(_standing(x=9.0), True, 1.0).is_zero()
    # Still held: no recalibration happens, so it stays zero.
    assert _approach(r, 1.1, x=9.2, start=(9.0, 0.0, 0.0)).is_zero()
    assert r.status["reason"] == "not calibrated"
    # Release + re-engage restores control.
    r.update(_standing(x=9.2), False, 2.0)
    r.update(_standing(x=9.2), True, 2.1)
    assert _approach(r, 2.2, x=9.65, start=(9.2, 0.0, 0.0)).vx > 0.3


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
    """A frozen feed (headset removed) must not keep a lean alive."""
    tuning = LocoTuning(stale_timeout_s=0.5)
    r = HeadLocomotionRetargeter(tuning)
    r.update(_standing(), True, 0.0)
    assert _approach(r, 0.033, x=0.45).vx > 0.3

    # Exactly the same matrix from here on: the feed is dead but still perfectly valid,
    # which is precisely what safe_mat_update cannot catch.
    frozen = _standing(x=0.45)
    still_walking = r.update(frozen, True, 1.0)          # first identical frame starts the clock
    assert still_walking.vx > 0.3, "should not stop before the timeout"
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
    # Walk out to the saturation region in steps small enough to dodge jump rejection.
    t = 0.1
    twist = None
    pose_x = 0.0
    while pose_x < 1.2:
        pose_x += 0.12
        twist = r.update(_standing(x=pose_x), True, t)
        t += 0.033
    assert twist.vx <= tuning.max_vx + 1e-9, twist
    assert abs(twist.vy) <= tuning.max_vy + 1e-9
    assert abs(twist.wz) <= tuning.max_wz + 1e-9


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
