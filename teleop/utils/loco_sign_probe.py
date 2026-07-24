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
import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

import logging_mp
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from teleop.utils.locomotion_publisher import LocomotionCommandPublisher

AXES = {"vx": 0, "vy": 1, "wz": 2}

EXPECTATION = {
    "vx": "POSITIVE value should walk FORWARD  (robot +x). If it walks backward -> sx = -1",
    "vy": "POSITIVE value should strafe LEFT   (robot +y). If it strafes right  -> sy = -1",
    "wz": "POSITIVE value should turn CCW seen from ABOVE (turning left). If CW  -> sw = -1",
}


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
    args = parser.parse_args()

    ChannelFactoryInitialize(args.domain)

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

    logger_mp.info(f"RESULT for {args.axis}: record what you observed, then set --loco-sign "
                   f"accordingly (order: vx,vy,wz).")


if __name__ == "__main__":
    main()
