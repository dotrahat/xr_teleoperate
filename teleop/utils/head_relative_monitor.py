"""Standalone head-relative wrist CSV logger and live Matplotlib monitor.

This module deliberately has no dependency on EpisodeWriter or the episode
recording pipeline.  Matplotlib and Pinocchio work happen only in a spawned
child process; the control loop performs a non-blocking queue write.
"""

import csv
import json
import multiprocessing as mp
import os
import time
from collections import deque
from dataclasses import dataclass
from queue import Empty, Full

import numpy as np

import logging_mp

logger_mp = logging_mp.getLogger(__name__)


# Legacy TeleVuer head-to-waist correction.  Keep it as the default so the
# calibration can be trialled explicitly before changing established control
# behaviour.
ROBOT_HEAD_IN_WAIST_M = np.array([0.15, 0.0, 0.45], dtype=float)
# G1 camera origin expressed in the fixed-pelvis IK frame at neutral waist.
# The physical D435 is 17.53 mm left of centre; use the robot sagittal plane as
# the head origin so left/right arm targets remain symmetric.
G1_CALIBRATED_HEAD_IN_WAIST_M = np.array([0.05366, 0.0, 0.47387], dtype=float)
SUPPORTED_G1_ARMS = frozenset(("G1_23", "G1_29"))

_ARMS = ("left", "right")
_AXES = ("x", "y", "z")
_CSV_HEADER = ["seq", "t_wall", "t_rel", "ik_ok"]
for _arm in _ARMS:
    _CSV_HEADER += [f"{_arm}_human_d{axis}" for axis in _AXES]
    _CSV_HEADER += [f"{_arm}_human_distance"]
    _CSV_HEADER += [f"{_arm}_robot_d{axis}" for axis in _AXES]
    _CSV_HEADER += [f"{_arm}_robot_distance"]
    _CSV_HEADER += [f"{_arm}_error_d{axis}" for axis in _AXES]
    _CSV_HEADER += [f"{_arm}_error_distance"]


def validate_head_relative_monitor_config(arm, arm_reference_mode,
                                          window_seconds, plot_rate_hz):
    """Validate the CLI-facing configuration for the G1-only monitor."""
    if arm not in SUPPORTED_G1_ARMS:
        raise ValueError(
            "--head-relative-monitor supports G1 teleoperation only "
            f"(--arm=G1_23 or --arm=G1_29; got {arm})."
        )
    if arm_reference_mode not in ("head_yaw", "head_position"):
        raise ValueError(
            "--head-relative-monitor requires a supported arm reference mode "
            f"(got {arm_reference_mode})."
        )
    if window_seconds <= 0.0:
        raise ValueError("--head-relative-monitor-window must be positive.")
    if plot_rate_hz <= 0.0:
        raise ValueError("--head-relative-monitor-rate must be positive.")


def robot_head_in_waist(arm, use_g1_calibration=False):
    """Select the control/monitor head origin without changing legacy defaults."""
    if use_g1_calibration and arm not in SUPPORTED_G1_ARMS:
        raise ValueError(
            "--g1-head-origin-calibration supports G1 teleoperation only "
            f"(--arm=G1_23 or --arm=G1_29; got {arm})."
        )
    selected = (
        G1_CALIBRATED_HEAD_IN_WAIST_M
        if use_g1_calibration
        else ROBOT_HEAD_IN_WAIST_M
    )
    return selected.copy()


def retarget_wrist_pose_head_origin(
        wrist_pose, target_head_in_waist,
        source_head_in_waist=ROBOT_HEAD_IN_WAIST_M):
    """Move a TeleVuer wrist target from one virtual head origin to another."""
    pose = np.asarray(wrist_pose, dtype=float)
    target = np.asarray(target_head_in_waist, dtype=float)
    source = np.asarray(source_head_in_waist, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("wrist_pose must be a finite 4x4 pose")
    if target.shape != (3,) or source.shape != (3,):
        raise ValueError("head origins must be xyz vectors")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(source)):
        raise ValueError("head origins must be finite")
    remapped = pose.copy()
    remapped[:3, 3] += target - source
    return remapped


def head_relative_monitor_run_name(sim, arm, ee, input_mode,
                                   arm_reference_mode, timestamp):
    """Return a filesystem-safe run name from argparse-constrained values."""
    target = "sim" if sim else "real"
    end_effector = ee or "no_ee"
    return (
        f"{target}_{arm}_{end_effector}_{input_mode}_"
        f"{arm_reference_mode}_{timestamp}"
    )


@dataclass(frozen=True)
class RelativeMeasurement:
    human: np.ndarray
    robot: np.ndarray
    error: np.ndarray

    @property
    def human_distance(self):
        return float(np.linalg.norm(self.human))

    @property
    def robot_distance(self):
        return float(np.linalg.norm(self.robot))

    @property
    def error_distance(self):
        return float(np.linalg.norm(self.error))


def relative_measurement(human_wrist_in_waist, robot_wrist_in_waist,
                         robot_head_in_waist=ROBOT_HEAD_IN_WAIST_M):
    """Compare wrist vectors relative to their respective head references.

    TeleVuer has already expressed the human wrist in the selected head frame
    and added the fixed head-to-waist correction.  Subtracting that correction
    recovers the unscaled human wrist-minus-head vector.  The same subtraction
    puts measured robot FK in the corresponding virtual-head frame.
    """
    head = np.asarray(robot_head_in_waist, dtype=float)
    human = np.asarray(human_wrist_in_waist, dtype=float) - head
    robot = np.asarray(robot_wrist_in_waist, dtype=float) - head
    if human.shape != (3,) or robot.shape != (3,) or head.shape != (3,):
        raise ValueError("head and wrist translations must be xyz vectors")
    return RelativeMeasurement(human=human, robot=robot, error=robot - human)


def csv_row(seq, t_wall, t_rel, ik_ok, left, right):
    """Build one stable, flat CSV row from left/right relative measurements."""
    row = [int(seq), float(t_wall), float(t_rel), int(bool(ik_ok))]
    for measurement in (left, right):
        row += measurement.human.tolist() + [measurement.human_distance]
        row += measurement.robot.tolist() + [measurement.robot_distance]
        row += measurement.error.tolist() + [measurement.error_distance]
    return row


class _PlotWindow:
    """Rolling Matplotlib view. Imported and constructed only in the worker."""

    def __init__(self, plt, window_seconds):
        self._plt = plt
        self.window_seconds = window_seconds
        self.history = {arm: deque() for arm in _ARMS}
        self.figure, axes = plt.subplots(5, 2, figsize=(14, 11), sharex="col")
        if hasattr(self.figure.canvas.manager, "set_window_title"):
            self.figure.canvas.manager.set_window_title("Head-relative wrist comparison")
        self.lines = {}

        row_specs = (
            ("Distance magnitude", "distance", "m"),
            ("Relative X", "x", "m"),
            ("Relative Y", "y", "m"),
            ("Relative Z", "z", "m"),
            ("XYZ error magnitude", "error", "m"),
        )
        for col, arm in enumerate(_ARMS):
            for row, (title, metric, unit) in enumerate(row_specs):
                axis = axes[row, col]
                axis.set_title(f"{arm.title()} — {title}")
                axis.set_ylabel(unit)
                axis.grid(True, alpha=0.3)
                if metric == "error":
                    (error_line,) = axis.plot([], [], color="tab:red", label="|robot − human|")
                    self.lines[(arm, metric, "error")] = error_line
                else:
                    (human_line,) = axis.plot([], [], color="tab:blue", label="human")
                    (robot_line,) = axis.plot([], [], color="tab:orange", label="robot actual")
                    self.lines[(arm, metric, "human")] = human_line
                    self.lines[(arm, metric, "robot")] = robot_line
                axis.legend(loc="upper right")
            axes[-1, col].set_xlabel("time (s)")

        self.figure.tight_layout()
        plt.ion()
        plt.show(block=False)

    @property
    def is_open(self):
        return self._plt.fignum_exists(self.figure.number)

    def append(self, t_rel, left, right):
        for arm, measurement in zip(_ARMS, (left, right)):
            history = self.history[arm]
            history.append((float(t_rel), measurement))
            cutoff = float(t_rel) - self.window_seconds
            while history and history[0][0] < cutoff:
                history.popleft()

    def redraw(self):
        if not self.is_open:
            return False

        for arm in _ARMS:
            history = self.history[arm]
            if not history:
                continue
            times = np.asarray([item[0] for item in history])
            humans = np.asarray([item[1].human for item in history])
            robots = np.asarray([item[1].robot for item in history])
            errors = np.asarray([item[1].error_distance for item in history])
            human_distances = np.linalg.norm(humans, axis=1)
            robot_distances = np.linalg.norm(robots, axis=1)

            series = {
                "distance": (human_distances, robot_distances),
                "x": (humans[:, 0], robots[:, 0]),
                "y": (humans[:, 1], robots[:, 1]),
                "z": (humans[:, 2], robots[:, 2]),
            }
            for metric, (human_values, robot_values) in series.items():
                self.lines[(arm, metric, "human")].set_data(times, human_values)
                self.lines[(arm, metric, "robot")].set_data(times, robot_values)
            self.lines[(arm, "error", "error")].set_data(times, errors)

        for axis in self.figure.axes:
            axis.relim()
            axis.autoscale_view(scalex=False, scaley=True)
        newest = max(
            (history[-1][0] for history in self.history.values() if history),
            default=0.0,
        )
        left = max(0.0, newest - self.window_seconds)
        right = max(self.window_seconds, newest)
        for axis in self.figure.axes:
            axis.set_xlim(left, right)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        self._plt.pause(0.001)
        return self.is_open

    def close(self):
        if self.is_open:
            self._plt.close(self.figure)


def _open_plot(window_seconds, out_dir):
    # A display-less run must still produce CSV. Do not force a GUI backend.
    if os.name != "nt" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        logger_mp.warning("Head-relative monitor: no graphical display; continuing with CSV only.")
        return None
    try:
        # Keep Matplotlib's cache inside this standalone run. This avoids writes to
        # ~/.config and keeps first-import work out of the teleoperation process.
        mpl_config_dir = os.path.join(out_dir, ".matplotlib")
        os.makedirs(mpl_config_dir, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)
        import matplotlib
        # OpenCV in the teleoperation process ships its own Qt plugins, which can
        # make Matplotlib's automatic Qt backend abort while loading xcb.  Tk is
        # part of the tv-2 environment and keeps this plotting process independent
        # of OpenCV's GUI stack.
        matplotlib.use("TkAgg", force=True)
        import matplotlib.pyplot as plt
        return _PlotWindow(plt, window_seconds)
    except Exception as exc:
        logger_mp.warning(f"Head-relative monitor: Matplotlib viewer unavailable ({exc}); continuing with CSV only.")
        return None


def _monitor_worker(queue, status_queue, model, left_frame_id, right_frame_id, out_dir, metadata,
                    robot_head_in_waist, window_seconds, plot_rate_hz, show_plot):
    """Child-process entry point: FK, CSV and GUI are all owned here."""
    import pinocchio as pin

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "head_relative.csv")
    meta_path = os.path.join(out_dir, "run_meta.json")
    run_meta = dict(metadata)
    run_meta.update({
        "robot_head_in_waist_m": robot_head_in_waist.tolist(),
        "csv_columns": _CSV_HEADER,
        "started_at_unix": time.time(),
    })
    with open(meta_path, "w", encoding="utf-8") as meta_file:
        json.dump(run_meta, meta_file, indent=2)

    fk_data = model.createData()
    plot = _open_plot(window_seconds, out_dir) if show_plot else None
    redraw_period = 1.0 / plot_rate_hz
    next_redraw = time.monotonic() + redraw_period
    rows_written = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(_CSV_HEADER)
        csv_file.flush()
        status_queue.put(("ready", "matplotlib" if plot is not None else "csv_only"))
        while True:
            timeout = max(0.0, min(0.1, next_redraw - time.monotonic())) if plot else 0.1
            try:
                payload = queue.get(timeout=timeout)
            except Empty:
                payload = "timeout"

            if payload is None:
                break
            if payload != "timeout":
                seq, t_wall, t_rel, human_left, human_right, measured_q, ik_ok = payload
                try:
                    q = np.asarray(measured_q, dtype=float)
                    pin.framesForwardKinematics(model, fk_data, q)
                    robot_left = fk_data.oMf[left_frame_id].translation.copy()
                    robot_right = fk_data.oMf[right_frame_id].translation.copy()
                    left = relative_measurement(human_left, robot_left, robot_head_in_waist)
                    right = relative_measurement(human_right, robot_right, robot_head_in_waist)
                    writer.writerow(csv_row(seq, t_wall, t_rel, ik_ok, left, right))
                    rows_written += 1
                    if rows_written % 30 == 0:
                        csv_file.flush()
                    if plot is not None:
                        plot.append(t_rel, left, right)
                except Exception as exc:
                    logger_mp.error(f"Head-relative monitor dropped invalid sample {seq}: {exc}")

            if plot is not None and time.monotonic() >= next_redraw:
                if not plot.redraw():
                    logger_mp.info("Head-relative monitor: plot window closed; CSV logging continues.")
                    plot = None
                next_redraw = time.monotonic() + redraw_period

        csv_file.flush()

    if plot is not None:
        plot.close()
    logger_mp.info(f"Head-relative monitor closed ({rows_written} rows) -> {csv_path}")


def _monitor_worker_entry(*args):
    """Report Python startup failures to the parent before exiting."""
    status_queue = args[1]
    try:
        _monitor_worker(*args)
    except BaseException as exc:
        try:
            status_queue.put(("error", repr(exc)))
        except Exception:
            pass
        raise


class HeadRelativeMonitor:
    """Non-blocking parent-side handle for the independent monitor process."""

    def __init__(self, model, left_frame_id, right_frame_id, out_dir, metadata,
                 window_seconds=20.0, plot_rate_hz=10.0, show_plot=True,
                 queue_size=512, robot_head_in_waist=ROBOT_HEAD_IN_WAIST_M):
        if window_seconds <= 0.0:
            raise ValueError("window_seconds must be positive")
        if plot_rate_hz <= 0.0:
            raise ValueError("plot_rate_hz must be positive")
        if queue_size < 2:
            raise ValueError("queue_size must be at least 2")
        robot_head_in_waist = np.asarray(robot_head_in_waist, dtype=float)
        if robot_head_in_waist.shape != (3,) or not np.all(np.isfinite(robot_head_in_waist)):
            raise ValueError("robot_head_in_waist must be a finite xyz vector")
        robot_head_in_waist = robot_head_in_waist.copy()

        self.out_dir = out_dir
        self.csv_path = os.path.join(out_dir, "head_relative.csv")
        self._warned_dead = False
        self._warned_bad_sample = False
        self._dropped = 0
        self._closed = False
        context = mp.get_context("spawn")
        self._queue = context.Queue(maxsize=queue_size)
        self._status_queue = context.Queue(maxsize=2)
        self._process = context.Process(
            target=_monitor_worker_entry,
            name="head-relative-monitor",
            args=(self._queue, self._status_queue, model, left_frame_id, right_frame_id, out_dir, metadata,
                  robot_head_in_waist, float(window_seconds), float(plot_rate_hz), bool(show_plot)),
        )
        self._process.start()
        try:
            status, detail = self._status_queue.get(timeout=15.0)
        except Empty as exc:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=1.0)
            self._close_queues()
            raise RuntimeError("monitor process did not become ready within 15 seconds") from exc
        if status != "ready":
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            self._close_queues()
            raise RuntimeError(f"monitor process failed to start: {detail}")
        logger_mp.info(
            f"==> Head-relative monitor ready ({detail}), writing to {self.csv_path}"
        )

    @property
    def dropped_samples(self):
        return self._dropped

    def log(self, seq, t_rel, desired_left, desired_right, measured_q, ik_ok):
        """Submit one sample without ever waiting on disk, FK or Matplotlib."""
        try:
            if self._closed:
                return
            if not self._process.is_alive():
                if not self._warned_dead:
                    logger_mp.error("Head-relative monitor process stopped; teleoperation will continue.")
                    self._warned_dead = True
                return
            desired_left = np.asarray(desired_left, dtype=float)
            desired_right = np.asarray(desired_right, dtype=float)
            measured_q = np.asarray(measured_q, dtype=float)
            if desired_left.shape != (4, 4) or desired_right.shape != (4, 4):
                raise ValueError("desired wrist poses must be 4x4 matrices")
            payload = (
                int(seq),
                time.time(),
                float(t_rel),
                tuple(desired_left[:3, 3]),
                tuple(desired_right[:3, 3]),
                tuple(measured_q),
                bool(ik_ok),
            )
            try:
                self._queue.put_nowait(payload)
            except Full:
                # Prefer the newest control state for a live monitor. This is intentionally
                # lossy under backpressure and is recorded in the shutdown log.
                self._dropped += 1
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(payload)
                except (Empty, Full):
                    pass
        except Exception as exc:
            if not self._warned_bad_sample:
                logger_mp.error(
                    f"Head-relative monitor rejected a sample ({exc}); teleoperation will continue."
                )
                self._warned_bad_sample = True

    def close(self, timeout=5.0):
        if self._closed:
            return
        self._closed = True
        if self._process.is_alive():
            while True:
                try:
                    self._queue.put_nowait(None)
                    break
                except Full:
                    self._dropped += 1
                    try:
                        self._queue.get_nowait()
                    except Empty:
                        pass
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                logger_mp.warning("Head-relative monitor did not stop in time; terminating it.")
                self._process.terminate()
                self._process.join(timeout=1.0)
        self._close_queues()
        if self._dropped:
            logger_mp.warning(f"Head-relative monitor dropped {self._dropped} samples due to backpressure.")

    def _close_queues(self):
        # Never let a broken pipe or feeder thread hold up robot shutdown.
        self._queue.cancel_join_thread()
        self._queue.close()
        self._status_queue.cancel_join_thread()
        self._status_queue.close()
