"""Per-tick logger of the four teleoperation wrist-pose stages: desired (human), IK (achievable),
commanded (post velocity-clip), actual (measured). Writes a flat CSV plus a run_meta.json,
everything the offline analysis script needs, with no coupling back to it.

Design notes:
  - Reads only attributes every *ArmIK class already exposes generically
    (reduced_robot.model, L_hand_id, R_hand_id), so it needs no per-variant code.
  - Owns its own pin.Data for forward kinematics -- solve_ik() runs pin.rnea on
    arm_ik.reduced_robot.data, so reusing that data here would race/clobber oMf.
  - Writes happen on a background thread via a Queue (same pattern as EpisodeWriter), so a slow
    disk never stalls the 30 Hz control loop. The whole log() body is wrapped in try/except: a
    logging fault must never be able to take down a live teleop session.
"""

import csv
import json
import os
import subprocess
import time
from datetime import datetime
from queue import Empty, Queue
from threading import Thread

import numpy as np
import pinocchio as pin

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

_FLUSH_EVERY = 30  # ~1s of rows at 30 Hz


def _quat_wxyz(rotation_matrix):
    """(w, x, y, z) quaternion for a 3x3 rotation matrix."""
    q = pin.Quaternion(rotation_matrix)
    q.normalize()
    return float(q.w), float(q.x), float(q.y), float(q.z)


def _pose_row(translation, rotation_matrix):
    qw, qx, qy, qz = _quat_wxyz(rotation_matrix)
    return [float(translation[0]), float(translation[1]), float(translation[2]), qw, qx, qy, qz]


def _git_sha(repo_dir):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


_POSE_COLS = ("px", "py", "pz", "qw", "qx", "qy", "qz")
_STAGES = ("des", "ik", "cmd", "act")
_ARMS = ("l", "r")


class PoseErrorLogger:
    def __init__(self, arm_ik, out_dir, run_meta):
        """
        arm_ik: a live *_ArmIK instance (any variant) -- used only for its Pinocchio model and
                the L_ee/R_ee frame ids, which every variant defines identically.
        out_dir: directory for this run; created if missing.
        run_meta: dict of CLI args / context to persist alongside the data (see run_meta.json
                  below). Anything that changes the meaning of "desired pose" -- notably
                  --arm-reference-mode -- must be included so later comparisons aren't silently
                  invalid.
        """
        self._model = arm_ik.reduced_robot.model
        self._fk_data = self._model.createData()  # NOT arm_ik.reduced_robot.data -- see module docstring
        self._l_id = arm_ik.L_hand_id
        self._r_id = arm_ik.R_hand_id
        self._nq = self._model.nq

        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.csv_path = os.path.join(out_dir, "pose_error.csv")

        meta = dict(run_meta)
        meta.setdefault("started_at", datetime.now().isoformat())
        meta["nq"] = self._nq
        meta["joint_names"] = [self._model.names[i] for i in range(1, self._model.njoints)]
        meta["xr_teleoperate_git_sha"] = _git_sha(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        header = ["seq", "t_wall", "t_rel", "ik_ms", "ik_ok", "episode_id", "record_state"]
        for arm in _ARMS:
            for stage in _STAGES:
                header += [f"{arm}_{stage}_{c}" for c in _POSE_COLS]
        header += [f"q_sol_{i}" for i in range(self._nq)]
        header += [f"q_cmd_{i}" for i in range(self._nq)]
        header += [f"q_act_{i}" for i in range(self._nq)]
        self._header = header

        self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(header)
        self._rows_written = 0

        self._queue = Queue(-1)
        self._stop = False
        self._thread = Thread(target=self._process_queue, daemon=True)
        self._thread.start()

        logger_mp.info(f"==> PoseErrorLogger writing to {self.csv_path}")

    def _fk_pair(self, q):
        # Copy translation/rotation out immediately: oMf entries alias internal Pinocchio buffers
        # that the *next* framesForwardKinematics() call (for the next stage) overwrites in place.
        pin.framesForwardKinematics(self._model, self._fk_data, np.asarray(q, dtype=float))
        l = self._fk_data.oMf[self._l_id]
        r = self._fk_data.oMf[self._r_id]
        return (l.translation.copy(), l.rotation.copy()), (r.translation.copy(), r.rotation.copy())

    def log(self, seq, t_rel, desired_l, desired_r, sol_q, clipped_q, measured_q,
            ik_ms, ik_ok, episode_id=-1, record_state=""):
        """Log one control tick. desired_l/desired_r are 4x4 SE(3) matrices
        (tele_data.left/right_wrist_pose). sol_q, clipped_q, measured_q are dual-arm joint
        vectors of length nq (clipped_q may equal sol_q, e.g. in simulation). Never raises."""
        try:
            if len(sol_q) != self._nq or len(clipped_q) != self._nq or len(measured_q) != self._nq:
                logger_mp.error(
                    f"PoseErrorLogger: joint vector length mismatch (nq={self._nq}, "
                    f"sol={len(sol_q)}, cmd={len(clipped_q)}, act={len(measured_q)}) -- dropping row"
                )
                return

            (ik_l_t, ik_l_r), (ik_r_t, ik_r_r) = self._fk_pair(sol_q)
            (cmd_l_t, cmd_l_r), (cmd_r_t, cmd_r_r) = self._fk_pair(clipped_q)
            (act_l_t, act_l_r), (act_r_t, act_r_r) = self._fk_pair(measured_q)

            row = [seq, time.time(), t_rel, ik_ms, int(bool(ik_ok)), episode_id, record_state]
            row += _pose_row(desired_l[:3, 3], desired_l[:3, :3])
            row += _pose_row(ik_l_t, ik_l_r)
            row += _pose_row(cmd_l_t, cmd_l_r)
            row += _pose_row(act_l_t, act_l_r)
            row += _pose_row(desired_r[:3, 3], desired_r[:3, :3])
            row += _pose_row(ik_r_t, ik_r_r)
            row += _pose_row(cmd_r_t, cmd_r_r)
            row += _pose_row(act_r_t, act_r_r)
            row += list(np.asarray(sol_q, dtype=float))
            row += list(np.asarray(clipped_q, dtype=float))
            row += list(np.asarray(measured_q, dtype=float))

            self._queue.put(row)
        except Exception:
            import traceback
            logger_mp.error(f"PoseErrorLogger.log failed:\n{traceback.format_exc()}")

    def _process_queue(self):
        while not self._stop or not self._queue.empty():
            try:
                row = self._queue.get(timeout=1)
            except Empty:
                continue
            try:
                self._writer.writerow(row)
                self._rows_written += 1
                if self._rows_written % _FLUSH_EVERY == 0:
                    self._file.flush()
            except Exception as e:
                logger_mp.error(f"PoseErrorLogger: failed to write row: {e}")
            finally:
                self._queue.task_done()

    def close(self):
        self._queue.join()
        self._stop = True
        self._thread.join(timeout=5)
        self._file.flush()
        self._file.close()
        logger_mp.info(f"==> PoseErrorLogger closed ({self._rows_written} rows) -> {self.csv_path}")
