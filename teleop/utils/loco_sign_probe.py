# Phase 0: empirical characterisation of the sim locomotion wire convention.
#
# Nothing about head tracking is involved. The only goal is to find out which wire sign
# means "forward", "left" and "counter-clockwise" for a given wholebody policy, so that
# --loco-sign can be set correctly before any head pose is wired up.
#
# It runs with sign=(1,1,1) deliberately: you observe the RAW convention, not the reference
# senders' negation. send_commands_keyboard.py:286 negates y and yaw relative to its own
# internal variables, but those variables' convention is undocumented -- hence this probe.
#
# The two policies are separate ONNX files from separate training runs. Do NOT assume they
# share a convention: run this against G129 and G123 independently.
#
# Usage (sim must already be running):
#   python -m teleop.utils.loco_sign_probe --axis vx --value 0.3 --duration 3
#   python -m teleop.utils.loco_sign_probe --axis vy --value 0.3 --duration 3
#   python -m teleop.utils.loco_sign_probe --axis wz --value 0.5 --duration 3
#   python -m teleop.utils.loco_sign_probe --axis vx --value 0.3 --duration 3 --hold
import argparse
import json
import math
import os
import sys
import threading
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

import logging_mp
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from teleop.utils.locomotion_publisher import LocomotionCommandPublisher

AXES = {"vx": 0, "vy": 1, "wz": 2}

EXPECTATION = {
    "vx": "POSITIVE value should walk FORWARD  (robot +x). If it walks backward -> sx = -1",
    "vy": "POSITIVE value should strafe LEFT   (robot +y). If it strafes right  -> sy = -1",
    "wz": "POSITIVE value should turn CCW seen from ABOVE (turning left). If CW  -> sw = -1",
}

# Displacement below this is treated as "did not move" rather than a direction.
MIN_MOVE_M = 0.05
MIN_TURN_RAD = 0.10


def _find_key(node, target):
    """Depth-first search for `target` anywhere in a nested dict/list."""
    if isinstance(node, dict):
        if target in node:
            return node[target]
        for value in node.values():
            found = _find_key(value, target)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_key(item, target)
            if found is not None:
                return found
    return None


def _decode_nested(node, depth=0):
    """Decode JSON strings nested inside an already-decoded payload.

    rt/sim_state is DOUBLE-encoded: sim_main.py builds
    {"init_state": sim_state_to_json(env_state), ...} -- so init_state is itself a JSON
    *string* -- and SimStateDDS.dds_publisher then json.dumps() the whole dict again.
    A single json.loads() therefore leaves init_state as a str, and the articulation data
    is invisible to a plain dict walk.
    """
    if depth > 3:
        return node
    if isinstance(node, str):
        stripped = node.lstrip()
        if stripped[:1] in ("{", "["):
            try:
                return _decode_nested(json.loads(node), depth + 1)
            except Exception:
                return node
        return node
    if isinstance(node, dict):
        return {k: _decode_nested(v, depth + 1) for k, v in node.items()}
    if isinstance(node, list):
        return [_decode_nested(v, depth + 1) for v in node]
    return node


def extract_base_pose(payload):
    """(x, y, yaw) of the robot base from an rt/sim_state message, or None.

    The sim publishes env.scene.get_state() every loop iteration (sim_main.py:485-493).
    Verified wire shape:
        {"init_state": "{\\"articulation\\": {\\"robot\\": {\\"root_pose\\": [[x, y, z, qw, qx, qy, qz]], ...
    i.e. root_pose is batched (env dim first) and the quaternion is w-first.
    """
    root_pose = _find_key(_decode_nested(payload), "root_pose")
    if root_pose is None:
        return None
    while isinstance(root_pose, list) and root_pose and isinstance(root_pose[0], list):
        root_pose = root_pose[0]
    if not isinstance(root_pose, list) or len(root_pose) < 7:
        return None
    x, y = float(root_pose[0]), float(root_pose[1])
    qw, qx, qy, qz = (float(v) for v in root_pose[3:7])
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return x, y, yaw


class SimStateWatcher:
    """Tracks the robot base pose from rt/sim_state so the probe can MEASURE the result."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pose = None
        self._messages = 0
        self._sub = ChannelSubscriber("rt/sim_state", String_)
        self._sub.Init(self._on_msg, 1)

    def _on_msg(self, msg):
        try:
            pose = extract_base_pose(json.loads(msg.data))
        except Exception:
            return
        if pose is not None:
            with self._lock:
                self._pose = pose
                self._messages += 1

    @property
    def pose(self):
        with self._lock:
            return self._pose

    @property
    def messages(self):
        with self._lock:
            return self._messages


def main():
    parser = argparse.ArgumentParser(description="Probe the sim locomotion wire sign convention.")
    parser.add_argument("--axis", choices=list(AXES), required=True, help="which axis to drive")
    parser.add_argument("--value", type=float, default=0.3, help="value to hold on that axis")
    parser.add_argument("--duration", type=float, default=3.0, help="seconds to hold it")
    parser.add_argument("--rate", type=float, default=100.0, help="publish rate (Hz)")
    parser.add_argument("--height", type=float, default=0.8, help="4th element (G123 ignores it)")
    parser.add_argument("--hold", action="store_true",
                        help="stop publishing WITHOUT zeroing, to verify the sim-side "
                             "consume-once reset and the publisher watchdog")
    parser.add_argument("--domain", type=int, default=1, help="DDS domain (1 = sim)")
    parser.add_argument("--no-measure", action="store_true",
                        help="skip the rt/sim_state measurement and rely on visual observation")
    args = parser.parse_args()

    ChannelFactoryInitialize(args.domain)

    watcher = None
    if not args.no_measure:
        watcher = SimStateWatcher()
        time.sleep(0.5)
        if watcher.messages == 0:
            logger_mp.warning("no rt/sim_state messages yet -- is the sim running? "
                              "Falling back to visual observation only.")
            watcher = None

    # sign=(1,1,1): observe the raw wire convention, not an adapted one.
    pub = LocomotionCommandPublisher(sim=True, rate_hz=args.rate, sign=(1.0, 1.0, 1.0),
                                     height=args.height)
    pub.start()

    twist = [0.0, 0.0, 0.0]
    twist[AXES[args.axis]] = args.value

    logger_mp.info("=" * 74)
    logger_mp.info(f"PROBE  axis={args.axis}  value={args.value:+.3f}  duration={args.duration}s")
    logger_mp.info(f"WATCH: {EXPECTATION[args.axis]}")
    logger_mp.info("=" * 74)

    start_pose = watcher.pose if watcher else None

    try:
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            pub.set_twist(twist)
            time.sleep(0.02)

        if args.hold:
            # Deliberately stop feeding setpoints without zeroing. Two things should happen:
            # the publisher watchdog trips after ~0.25 s and starts sending zeros, and the
            # sim's consume-once reset means it would stand even if we sent nothing at all.
            logger_mp.info("--hold: setpoints stopped WITHOUT zeroing. "
                           "Robot must come to a stand within ~0.3 s (watchdog).")
            time.sleep(3.0)
        else:
            logger_mp.info("holding zero for 1 s ...")
            pub.zero()
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                pub.set_twist([0.0, 0.0, 0.0])
                time.sleep(0.02)
    except KeyboardInterrupt:
        logger_mp.info("interrupted")
    finally:
        pub.stop()

    end_pose = watcher.pose if watcher else None
    if start_pose and end_pose:
        _report_measurement(args, start_pose, end_pose)
    else:
        logger_mp.info(f"RESULT for {args.axis}: record what you observed, then set "
                       f"--loco-sign accordingly (order: vx,vy,wz).")


def _report_measurement(args, start_pose, end_pose):
    """Turn the measured base displacement into a sign, so nobody has to eyeball it."""
    x0, y0, yaw0 = start_pose
    x1, y1, yaw1 = end_pose
    dx_world, dy_world = x1 - x0, y1 - y0
    # Express the displacement in the body frame the robot STARTED in.
    forward = dx_world * math.cos(yaw0) + dy_world * math.sin(yaw0)
    left = -dx_world * math.sin(yaw0) + dy_world * math.cos(yaw0)
    dyaw = (yaw1 - yaw0 + math.pi) % (2.0 * math.pi) - math.pi

    logger_mp.info("=" * 74)
    logger_mp.info("MEASURED (from rt/sim_state base pose):")
    logger_mp.info(f"    forward : {forward:+.3f} m   (+ = forward)")
    logger_mp.info(f"    left    : {left:+.3f} m   (+ = left)")
    logger_mp.info(f"    yaw     : {math.degrees(dyaw):+.1f} deg  (+ = CCW / turning left)")

    if args.axis == "vx":
        observed, threshold, unit = forward, MIN_MOVE_M, "m forward"
    elif args.axis == "vy":
        observed, threshold, unit = left, MIN_MOVE_M, "m left"
    else:
        observed, threshold, unit = dyaw, MIN_TURN_RAD, "rad CCW"

    if abs(observed) < threshold:
        logger_mp.warning(f"    -> INCONCLUSIVE: only {observed:+.3f} {unit}. The robot barely "
                          f"moved; try a larger --value or --duration. (G123 ignores commands "
                          f"below its 0.1 stand threshold.)")
        return

    # The wire carried +args.value on this axis (sign was (1,1,1)). If the robot moved in the
    # POSITIVE robot-frame direction, the wire already matches robot convention -> sign +1.
    sign = 1 if (observed > 0) == (args.value > 0) else -1
    index = {"vx": 0, "vy": 1, "wz": 2}[args.axis]
    template = ["?", "?", "?"]
    template[index] = f"{sign:+d}".replace("+1", "1")
    logger_mp.info(f"    -> {args.axis} sign = {sign:+d}   (use --loco-sign "
                   f"\"{','.join(template)}\", filling in the other axes from their probes)")
    logger_mp.info("=" * 74)


if __name__ == "__main__":
    main()
