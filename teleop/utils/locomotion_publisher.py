# Locomotion command publisher.
#
# Owns the wire and the clock for locomotion twists. The teleop main loop is only a
# *setpoint producer* (~30 Hz); this publisher re-sends the held setpoint at its own,
# higher rate. That is required, not cosmetic: both Isaac Lab wholebody action providers
# consume-once -- they reset the shared command to [0,0,0,0.8] after every read
# (action_provider_wh_dds.py:338, action_provider_wh_g123_dds.py:346) -- while sim control
# runs at 50 Hz. Publishing inline at 30 Hz would leave the robot standing on the gaps.
# The reference senders (send_commands_keyboard.py / send_commands_8bit.py) use 100 Hz.
#
# Two transports, selected by target:
#   sim  -> DDS String_ on "rt/run_command/cmd", payload str([vx, vy, wz, height])
#   real -> LocoClientWrapper.Move(vx, vy, wz)
import ast
import threading
import time

import logging_mp

from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

logger_mp = logging_mp.getLogger(__name__)

RUN_COMMAND_TOPIC = "rt/run_command/cmd"

# Envelope the wholebody policies were trained against, taken from the ranges the reference
# senders expose (send_commands_keyboard.py:45-50). Applied unconditionally as the last step
# before the wire so no CLI value can push the policy outside what it has seen.
VX_LIMITS = (-0.6, 1.0)
VY_LIMITS = (-0.5, 0.5)
WZ_LIMITS = (-1.57, 1.57)
HEIGHT_LIMITS = (0.3, 0.8)

DEFAULT_HEIGHT = 0.8


def _clamp(value, lo, hi):
    return lo if value < lo else (hi if value > hi else value)


def _slew(current, target, max_delta):
    """Asymmetric rate limit: ramp when speeding up, jump instantly when slowing down.

    Deceleration must never be delayed -- deadman release, watchdog trips and stop() all
    reach the wire within one publish period. A reversal (|target| == |current|, opposite
    sign) is ramped, so it passes through zero rather than snapping to full opposite speed.
    """
    if abs(target) < abs(current):
        return target
    delta = _clamp(target - current, -max_delta, max_delta)
    return current + delta


def _as_twist_tuple(twist, default_height):
    """Accept a Twist (anything with .as_list()), or a plain 3/4 sequence."""
    if hasattr(twist, "as_list"):
        values = list(twist.as_list())
    else:
        values = list(twist)
    if len(values) == 3:
        values.append(default_height)
    if len(values) != 4:
        raise ValueError(f"twist must have 3 or 4 elements, got {len(values)}: {values!r}")
    return tuple(float(v) for v in values)


class LocomotionCommandPublisher:
    """Publishes locomotion twists on a dedicated thread with a staleness watchdog.

    `sign` adapts the retargeter's robot-frame twist (x front, y left, z CCW) to the *sim
    wire* convention. MEASURED on the G129 Inspire wholebody policy (policy.onnx) with
    loco_sign_probe.py: the wire is plain robot-frame, so (1, 1, 1).

      +vx 0.5 -> +0.80 m / +1.21 m forward      +vy 0.4 -> +1.52 m left
      -vy 0.4 -> -1.92 m left (i.e. right)      +wz -> +yaw CCW, -wz -> -yaw CW

    Note this does NOT match the reference senders' str([x, -y, -yaw, h]): they negate
    their own internal variables, whose convention is undocumented. Re-verify per policy --
    G123 uses a different ONNX from a different training run.

    Deliberately NOT applied to the LocoClient path: that API already takes robot frame.
    """

    def __init__(self, sim, loco_wrapper=None, rate_hz=None, watchdog_timeout_s=0.25,
                 sign=(1.0, 1.0, 1.0), accel_xy=1.0, accel_yaw=2.0, height=DEFAULT_HEIGHT):
        self.sim = sim
        self.loco_wrapper = loco_wrapper
        # LocoClient.Move is an RPC per call and would saturate at 100 Hz; the existing
        # thumbstick path already drives it at the 30 Hz loop rate.
        self.rate_hz = float(rate_hz) if rate_hz else (100.0 if sim else 30.0)
        self.watchdog_timeout_s = watchdog_timeout_s
        self.sign = tuple(float(s) for s in sign)
        self.accel_xy = accel_xy
        self.accel_yaw = accel_yaw
        self.height = _clamp(height, *HEIGHT_LIMITS)

        if not sim and loco_wrapper is None:
            raise ValueError("hardware target requires a loco_wrapper (LocoClientWrapper)")

        self._lock = threading.Lock()
        self._target = (0.0, 0.0, 0.0, self.height)
        self._stamp = time.monotonic()
        self._current = [0.0, 0.0, 0.0]
        self._running = False
        self._thread = None
        self._publisher = None
        self._last_watchdog_warn = 0.0
        self._watchdog_active = False

        if self.sim:
            self._publisher = ChannelPublisher(RUN_COMMAND_TOPIC, String_)
            self._publisher.Init()

    # -- description used in the startup banner so a mis-targeted run is visible ----------
    @property
    def transport_description(self):
        if self.sim:
            return f"DDS String_ -> {RUN_COMMAND_TOPIC} @ {self.rate_hz:.0f} Hz"
        return f"LocoClient.Move() @ {self.rate_hz:.0f} Hz"

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="locomotion-publisher")
        self._thread.start()
        logger_mp.info(f"[locomotion] publisher started: {self.transport_description}, "
                       f"sign={self.sign}")

    def set_twist(self, twist):
        """Update the held setpoint. Called from the main loop."""
        values = _as_twist_tuple(twist, self.height)
        with self._lock:
            self._target = values
            self._stamp = time.monotonic()

    def zero(self):
        """Stop immediately, bypassing the ramp."""
        with self._lock:
            self._target = (0.0, 0.0, 0.0, self.height)
            self._stamp = time.monotonic()
            self._current = [0.0, 0.0, 0.0]

    def stop(self):
        """Stop the robot, then the thread. Safe to call more than once."""
        self.zero()
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        # Explicit zero frames after the thread is down, so the last thing on the wire is
        # a stand command even if the join timed out.
        try:
            for _ in range(20):
                self._emit(0.0, 0.0, 0.0, self.height)
                time.sleep(1.0 / self.rate_hz)
        except Exception as e:
            logger_mp.error(f"[locomotion] failed to publish final zero frames: {e}")
        logger_mp.info("[locomotion] publisher stopped")

    # -- internals -----------------------------------------------------------------------
    def _loop(self):
        period = 1.0 / self.rate_hz
        next_deadline = time.monotonic()
        while self._running:
            now = time.monotonic()
            with self._lock:
                target = self._target
                stamp = self._stamp

            stale = (now - stamp) > self.watchdog_timeout_s
            if stale:
                target = (0.0, 0.0, 0.0, self.height)
                if not self._watchdog_active or (now - self._last_watchdog_warn) > 1.0:
                    logger_mp.warning(
                        f"[locomotion] watchdog: no setpoint for {now - stamp:.2f}s "
                        f"(> {self.watchdog_timeout_s}s) -- commanding zero")
                    self._last_watchdog_warn = now
                self._watchdog_active = True
            elif self._watchdog_active:
                logger_mp.info("[locomotion] watchdog cleared, setpoints flowing again")
                self._watchdog_active = False

            self._current[0] = _slew(self._current[0], target[0], self.accel_xy * period)
            self._current[1] = _slew(self._current[1], target[1], self.accel_xy * period)
            self._current[2] = _slew(self._current[2], target[2], self.accel_yaw * period)

            try:
                self._emit(self._current[0], self._current[1], self._current[2], target[3])
            except Exception as e:
                logger_mp.error(f"[locomotion] publish failed: {e}")

            next_deadline += period
            sleep_for = next_deadline - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Fell behind (scheduling hiccup); resync rather than spin to catch up.
                next_deadline = time.monotonic()

    def _emit(self, vx, vy, wz, height):
        if self.sim:
            # Sign adapts robot frame -> sim wire convention (see class docstring).
            wire_vx = _clamp(self.sign[0] * vx, *VX_LIMITS)
            wire_vy = _clamp(self.sign[1] * vy, *VY_LIMITS)
            wire_wz = _clamp(self.sign[2] * wz, *WZ_LIMITS)
            wire_h = _clamp(height, *HEIGHT_LIMITS)
            # Always 4 elements: the G129 provider requires len >= 4 and silently drops
            # anything shorter; the G123 provider reads 3 and ignores height.
            payload = [round(wire_vx, 4), round(wire_vy, 4), round(wire_wz, 4), round(wire_h, 4)]
            self._publisher.Write(String_(data=str(payload)))
        else:
            # LocoClient already takes a robot-frame twist; no sign adaptation.
            self.loco_wrapper.Move(_clamp(vx, *VX_LIMITS),
                                   _clamp(vy, *VY_LIMITS),
                                   _clamp(wz, *WZ_LIMITS))


def parse_sign(text):
    """Parse a --loco-sign value like "1,-1,-1" into a 3-tuple of floats."""
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError(f"--loco-sign must be three comma-separated values, got {text!r}")
    values = tuple(float(p) for p in parts)
    for v in values:
        if v not in (-1.0, 1.0):
            raise ValueError(f"--loco-sign entries must be 1 or -1, got {text!r}")
    return values


def decode_run_command(data):
    """Decode a run_command payload the same way the sim does. Used by tests/probe."""
    return ast.literal_eval(data) if isinstance(data, str) else data
