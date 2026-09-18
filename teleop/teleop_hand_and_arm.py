import time
import argparse
from multiprocessing import Value, Array, Lock
import threading
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController, H2_ArmController, R1_A5_ArmController, R1_A7_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK, H2_ArmIK, R1_A5_ArmIK, R1_A7_ArmIK
from teleimager.image_client import ImageClient
from datetime import datetime
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.pose_error_logger import PoseErrorLogger
from teleop.utils.head_relative_monitor import (
    G1_CALIBRATED_HEAD_IN_WAIST_M,
    HeadRelativeMonitor,
    apply_wrist_target_residual,
    g1_wrist_target_residuals,
    head_relative_monitor_run_name,
    retarget_wrist_pose_head_origin,
    robot_head_in_waist,
    validate_head_relative_monitor_config,
)
from teleop.utils.ipc import IPC_Server
from teleop.utils.camera_display import (
    CameraDisplay,
    DualRingPinchGesture,
    resolve_camera_names,
)
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from teleop.utils.locomotion_retarget import (HeadLocomotionRetargeter, LocoTuning, eval_deadman,
                                              deadman_is_available, joystick_twist, LatchingToggle)
from teleop.utils.locomotion_publisher import (LocomotionCommandPublisher, parse_sign,
                                               VX_LIMITS, VY_LIMITS, WZ_LIMITS)
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
CAMERA_DISPLAY = None   # Set after TeleImage configuration is received.
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

def on_press(key):
    global STOP, START, RECORD_TOGGLE, CAMERA_DISPLAY
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
    elif key == 'c' and CAMERA_DISPLAY is not None:
        camera_name = CAMERA_DISPLAY.cycle()
        logger_mp.info(f"📷 XR camera: {camera_name}")
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }


def update_camera_switch_gesture(gesture, camera_display, tele_data):
    """Cycle the XR camera when the deliberate hand gesture completes."""
    if gesture is None:
        return False
    if not tele_data.motion_data_ready:
        gesture.reset()
        return False
    if gesture.update(
        tele_data.left_hand_pos,
        tele_data.right_hand_pos,
        tele_data.left_wrist_pose,
        tele_data.right_wrist_pose,
    ):
        camera_name = camera_display.cycle()
        logger_mp.info(f"📷 XR camera gesture accepted: {camera_name}")
        return True
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--xr-camera-layout', type=str, choices=['single', 'side-by-side'], default='single',
                        help='Headset camera layout. "single" shows one selected camera and [c] cycles; '
                             '"side-by-side" shows every selected camera simultaneously.')
    parser.add_argument('--xr-cameras', type=str, default='auto',
                        help='Camera topics shown in XR: "auto" (head camera, or first enabled), "all", '
                             'or a comma-separated list such as head_camera,left_wrist_camera.')
    parser.add_argument('--xr-camera-gesture', type=str, choices=['dual-ring-pinch', 'off'],
                        default='dual-ring-pinch',
                        help='Meta Quest hand gesture for cycling cameras in single layout. '
                             '"dual-ring-pinch" requires both thumb/ring fingertips touching, '
                             'other fingers straight, and palms facing the headset for three seconds; '
                             '"off" disables gesture switching.')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1', 'H2', 'R1_A5', 'R1_A7'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    parser.add_argument('--arm-reference-mode', type=str, choices=['head_yaw', 'head_position'], default='head_yaw',
                        help='Frame the wrist targets are expressed in. '
                             '"head_yaw" (upstream v1.6 default) rotates targets by the operator\'s '
                             'yaw, so the arms follow the direction you face. "head_position" is the '
                             'pre-v1.6 behaviour: translation relative to the head only, orientation '
                             'left in world frame.')
    # network parameters
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--debug-mode', action = 'store_true',
                        help = 'Explicitly allow Debug Mode (raw rt/lowcmd, balance controller '
                               'disabled) on hardware when --motion is not set. Required to '
                               'start without --motion on hardware; has no effect with --sim.')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    # head/body-driven locomotion (third mode). All opt-in: defaults preserve existing behaviour.
    parser.add_argument('--head-loco', action='store_true', help='Enable head/body-driven locomotion (walk to walk, stand still to stop; speed-matched)')
    parser.add_argument('--loco-yaw-source', type=str, choices=['roll', 'off'], default='roll', help='How to steer: head roll (tilt) or no turning at all')
    parser.add_argument('--loco-deadman', type=str, default='left_fist',
                        choices=['left_fist', 'right_fist', 'both_fist', 'left_pinch', 'right_pinch', 'left_trigger', 'right_trigger', 'none'],
                        help='Hold-to-walk gesture, used by --loco-gate=hold. Releasing it stops the robot immediately.')
    parser.add_argument('--loco-gate', type=str, choices=['auto', 'hold', 'toggle'], default='auto',
                        help='How head-driven walking is armed. "hold" = hold --loco-deadman the whole '
                             'time. "toggle" = latch on/off with --loco-toggle-button. "auto" (default) '
                             'picks hold for --input-mode hand and toggle for controller, where the '
                             'thumbsticks are the primary way to drive and every hold gesture collides '
                             'with end-effector control.')
    parser.add_argument('--loco-toggle-button', type=str, choices=['x', 'y', 'b'], default='x',
                        help='Face button that latches head-driven walking when --loco-gate=toggle. '
                             '"A" is not offered: it quits teleoperation.')
    parser.add_argument('--loco-stick-deadzone', type=float, default=0.08,
                        help='Thumbstick deflection below which the stick is treated as centred. '
                             'Beyond it the stick overrides head-driven walking.')
    parser.add_argument('--loco-max-vx', type=float, default=0.40, help='Max forward/back speed (m/s)')
    parser.add_argument('--loco-max-vy', type=float, default=0.25, help='Max lateral speed (m/s)')
    parser.add_argument('--loco-max-wz', type=float, default=0.60, help='Max yaw rate (rad/s)')
    parser.add_argument('--loco-walk-deadzone', type=float, default=0.12, help='Operator walking speed ignored before the robot starts (m/s)')
    parser.add_argument('--loco-walk-full', type=float, default=0.80,
                        help='Operator walking speed that commands FULL robot speed (m/s). This is '
                             'the gain knob: LOWER it to make the robot faster for the same walking '
                             'speed (robot m/s per operator m/s = max_vx / (walk_full - deadzone)). '
                             'Raising --loco-max-vx alone only raises the ceiling, not the gain.')
    parser.add_argument('--loco-neck-offset', type=float, default=0.10, help='Camera-to-neck-pivot distance (m)')
    parser.add_argument('--loco-accel-xy', type=float, default=1.0,
                        help='Translation ramp-up limit (m/s^2). Room-scale walking bursts are ~1-2 s, '
                             'so at the default 1.0 the robot spends most of a burst still ramping and '
                             'covers far less ground than you. Raise to ~2.5-3 to keep up. Deceleration '
                             'is never ramped.')
    parser.add_argument('--loco-accel-yaw', type=float, default=2.0, help='Yaw ramp-up limit (rad/s^2). Deceleration is never ramped.')
    parser.add_argument('--loco-rate', type=float, default=None, help='Publisher rate (Hz). Default 100 in sim, 30 on hardware.')
    parser.add_argument('--loco-sign', type=str, default='1,1,1', help='Wire sign per axis "vx,vy,wz". Measured on G129 Inspire wholebody; re-verify per policy with loco_sign_probe.py.')
    parser.add_argument('--loco-height', type=float, default=0.8, help='Base height command (4th element; G123 ignores it)')
    parser.add_argument('--loco-debug', action='store_true', help='Log locomotion status at 2 Hz')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')
    # pose-error logging: desired (human) vs achievable (IK) vs commanded (post velocity-clip) vs
    # actual (measured) wrist pose, per tick, for offline analysis. Independent of --record.
    parser.add_argument('--log-pose-error', action = 'store_true',
                        help = 'Log per-tick wrist pose error (desired/IK/commanded/actual) to CSV for offline analysis. G1_23 only.')
    # Deliberately outside --task-dir: analysis output never lands in the episode data tree.
    parser.add_argument('--pose-log-dir', type = str, default = './utils/pose_logs/',
                        help = 'Directory for --log-pose-error runs (a timestamped subdirectory is created per run).')
    # Independent live comparison of unscaled human and measured robot head-relative wrists.
    parser.add_argument('--head-relative-monitor', action='store_true',
                        help='Log head-relative human/robot wrists to standalone CSV and show live Matplotlib graphs (G1_23 and G1_29 only; all supported end effectors, input modes, and arm reference modes).')
    parser.add_argument('--head-relative-monitor-dir', type=str, default='./utils/head_relative_logs/',
                        help='Standalone output directory for --head-relative-monitor runs.')
    parser.add_argument('--head-relative-monitor-window', type=float, default=20.0,
                        help='Rolling Matplotlib time window in seconds.')
    parser.add_argument('--head-relative-monitor-rate', type=float, default=10.0,
                        help='Matplotlib redraw rate in Hz; CSV still receives every submitted control sample.')
    parser.add_argument('--head-relative-monitor-no-viewer', action='store_true',
                        help='Write standalone head-relative CSV without opening a Matplotlib window.')
    parser.add_argument('--g1-head-origin-calibration', action='store_true',
                        help='Opt in to the G1 CAD head-reference-to-waist calibration for arm targets. G1_23 also applies the calibrated Y/Z wrist residuals; G1_29 keeps zero residuals. Legacy target geometry remains the default.')

    args = parser.parse_args()

    # --- head-loco target normalisation -------------------------------------------------
    # The correct arm topic differs by target and getting it wrong is silent and fatal, so
    # --head-loco resolves it here rather than trusting the operator to remember --motion.
    #   sim      : --motion OFF -> rt/lowcmd  (the only topic the Isaac sim subscribes to)
    #   hardware : --motion ON  -> rt/arm_sdk (the only arm interface that coexists with the
    #              balance controller) AND skips Enter_Debug_Mode(), which would otherwise
    #              disable balance and make walking impossible.
    if args.head_loco:
        if args.sim:
            if args.motion:
                parser.error(
                    "--head-loco --sim is incompatible with --motion: --motion routes arm "
                    "commands to rt/arm_sdk, which the Isaac Lab sim does not subscribe to "
                    "(it listens on rt/lowcmd), so the arms would go dead. Drop --motion for sim.")
        else:
            if not args.motion:
                logger_mp.warning("--head-loco on hardware: forcing --motion so arms use "
                                  "rt/arm_sdk and the balance controller stays up.")
            args.motion = True
        # Resolve the gate. Controllers default to a latching toggle: the thumbsticks are the
        # natural way to drive, and every hold-to-walk gesture doubles as an end-effector
        # input (with --ee brainco, squeeze drives the index finger and the trigger drives
        # the other four), so holding one to walk would clench the hand.
        if args.loco_gate == 'auto':
            args.loco_gate = 'hold' if args.input_mode == 'hand' else 'toggle'
        # eval_deadman() runs inside the control loop, so validate here instead of letting an
        # illegal pairing raise on the first iteration with the robot already live.
        if args.loco_gate == 'hold' and not deadman_is_available(args.loco_deadman, args.input_mode):
            parser.error(
                f"--loco-deadman={args.loco_deadman} is not available in "
                f"--input-mode {args.input_mode}. Hand mode offers left/right/both_fist and "
                f"left/right_pinch; controller mode offers left/right/both_fist and "
                f"left/right_trigger.")
        try:
            loco_sign = parse_sign(args.loco_sign)
        except ValueError as e:
            parser.error(str(e))

    # --- hardware debug-mode guard -------------------------------------------------------
    # Without --motion, arm commands go out on raw rt/lowcmd via Enter_Debug_Mode(), which
    # disables the robot's internal balance controller entirely. On hardware this is one
    # missing flag away from happening by accident, and disastrous if the robot isn't
    # supported by a gantry. Require an explicit opt-in before allowing that path; refuse to
    # start rather than silently dropping into Debug Mode. --sim has no physical robot to
    # endanger and its normal workflow already omits --motion, so this does not apply there.
    #
    # Must run AFTER the head-loco block above: on hardware that block forces args.motion
    # True, which legitimately satisfies this guard. Ordering it first would reject
    # --head-loco hardware runs that were never going to enter Debug Mode.
    if not args.sim and not args.motion and not args.debug_mode:
        parser.error(
            "Refusing to start: --motion is not set. Without it, this program enters the "
            "robot's raw Debug Mode (rt/lowcmd), which disables the internal balance "
            "controller -- the robot will not hold itself up if unsupported. Pass --motion "
            "for normal operation, or --debug-mode to explicitly acknowledge and allow "
            "Debug Mode.")

    # R1 arms have no motion mode. Checked after the head-loco block above, which can
    # force args.motion True on hardware -- ordering it earlier would let that slip through.
    if args.arm in ("R1_A5", "R1_A7") and args.motion:
        parser.error(f"{args.arm} does not support motion mode (--motion).")

    # Pose-error logging is wired for G1_23 only: it needs both the post-clip command readback
    # (G1_23_ArmController.get_last_clipped_q_target) and the IK convergence flag. G1_29 now
    # exposes convergence for the head-relative monitor, but still has no post-clip readback.
    # Refuse to write a CSV whose 'cmd' columns would silently duplicate 'ik'.
    if args.log_pose_error and args.arm != "G1_23":
        parser.error(
            f"--log-pose-error is only supported on --arm=G1_23 (got {args.arm}). The other arm "
            "variants do not expose the required post-velocity-clip command readback, so the "
            "commanded columns would be fabricated.")

    try:
        active_robot_head_in_waist = robot_head_in_waist(
            args.arm, use_g1_calibration=args.g1_head_origin_calibration
        )
        left_wrist_target_residual, right_wrist_target_residual = (
            g1_wrist_target_residuals(
                args.arm, use_g1_calibration=args.g1_head_origin_calibration
            )
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.head_relative_monitor:
        try:
            validate_head_relative_monitor_config(
                arm=args.arm,
                arm_reference_mode=args.arm_reference_mode,
                window_seconds=args.head_relative_monitor_window,
                plot_rate_hz=args.head_relative_monitor_rate,
            )
        except ValueError as exc:
            parser.error(str(exc))

    logger_mp.debug(f"args: {args}")

    # Defined before the try so the finally block can always reference them.
    loco_wrapper = None
    loco_retarget = None
    loco_pub = None
    head_relative_monitor = None

    try:
        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")
        selected_camera_names = resolve_camera_names(camera_config, args.xr_cameras)
        camera_display = CameraDisplay(
            camera_config,
            camera_names=selected_camera_names,
            layout=args.xr_camera_layout,
        )
        CAMERA_DISPLAY = camera_display
        reference_camera = camera_config[selected_camera_names[0]]
        xr_need_local_img = (
            args.display_mode != 'pass-through'
            and (
                camera_display.requires_local_zmq()
                or not reference_camera.get('enable_webrtc', False)
            )
        )
        if xr_need_local_img:
            camera_display.validate_local_zmq()
        use_direct_webrtc = (
            args.display_mode != 'pass-through'
            and not xr_need_local_img
            and reference_camera.get('enable_webrtc', False)
        )
        logger_mp.info(
            f"📷 XR cameras: {', '.join(selected_camera_names)}; "
            f"layout={args.xr_camera_layout}; "
            f"transport={'ZMQ composite' if xr_need_local_img else ('WebRTC' if use_direct_webrtc else 'off')}"
        )
        if args.xr_camera_layout == 'single' and len(selected_camera_names) > 1:
            logger_mp.info("📷 Press [c] (or send IPC command 'c') to cycle XR cameras.")

        camera_switch_gesture = None
        gesture_requested = args.xr_camera_gesture == 'dual-ring-pinch'
        if (
            gesture_requested
            and args.input_mode == 'hand'
            and args.display_mode != 'pass-through'
            and args.xr_camera_layout == 'single'
            and len(selected_camera_names) > 1
        ):
            camera_switch_gesture = DualRingPinchGesture()
            logger_mp.info(
                "📷 Quest camera gesture enabled: touch thumb to ring fingertip on both "
                "hands, keep the other fingers straight, face both palms toward the "
                "headset, and hold for three seconds."
            )

        # televuer_wrapper: obtain XR poses and display either one direct stream or a
        # locally composed canvas containing the selected camera streams.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=camera_display.binocular,
                                     img_shape=(camera_display.height, camera_display.width),
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=xr_need_local_img,
                                     webrtc=use_direct_webrtc,
                                     webrtc_url=(f"https://{args.img_server_ip}:{reference_camera['webrtc_port']}/offer"
                                                 if use_direct_webrtc else None),
                                     arm_reference_mode=args.arm_reference_mode
                                     )
        logger_mp.info(f"🖐️  arm reference mode: {args.arm_reference_mode}")
        if args.g1_head_origin_calibration:
            logger_mp.info(
                "🎯 G1 head-origin calibration enabled: "
                f"{G1_CALIBRATED_HEAD_IN_WAIST_M.tolist()} m"
            )
            if args.arm == "G1_23":
                logger_mp.info(
                    "🎯 G1_23 wrist residual calibration enabled: "
                    f"left={left_wrist_target_residual.tolist()} m, "
                    f"right={right_wrist_target_residual.tolist()} m"
                )
        
        
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            # On hardware, head-loco also needs a LocoClient. In sim there is no locomotion
            # service at all -- the twist goes out over the run_command DDS topic instead --
            # so never construct one there.
            if args.input_mode == "controller" or (args.head_loco and not args.sim):
                loco_wrapper = LocoClientWrapper()
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        # arm
        if args.arm == "G1_29":
            arm_ik = G1_29_ArmIK()
            arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "G1_23":
            arm_ik = G1_23_ArmIK()
            arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim,
                                           log_pose_error=args.log_pose_error)
        elif args.arm == "H1_2":
            arm_ik = H1_2_ArmIK()
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1":
            arm_ik = H1_ArmIK()
            arm_ctrl = H1_ArmController(simulation_mode=args.sim)
        elif args.arm == "H2":
            arm_ik = H2_ArmIK()
            arm_ctrl = H2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "R1_A5":
            arm_ik = R1_A5_ArmIK()
            arm_ctrl = R1_A5_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "R1_A7":
            arm_ik = R1_A7_ArmIK()
            arm_ctrl = R1_A7_ArmController(motion_mode=args.motion, simulation_mode=args.sim)

        # end-effector
        xr_motion_data_ready = Value('b', False, lock=True)        # [input] whether XR hand/controller motion data has arrived
        if args.ee in ("dex3", "inspire_ftp", "inspire_dfx") and args.input_mode == "controller":
            raise ValueError(f"{args.ee} does not support controller input mode.")
        elif args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "brainco" and args.input_mode == "hand":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller_hand
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_hand(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock,
                                                dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "brainco" and args.input_mode == "controller":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller_ctrl
            left_gripper_trigger_in = Value('d', 10.0, lock=True)  # [input]
            left_gripper_squeeze_in = Value('d', 0.0, lock=True)   # [input]
            right_gripper_trigger_in = Value('d', 10.0, lock=True) # [input]
            right_gripper_squeeze_in = Value('d', 0.0, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_ctrl(left_gripper_trigger_in, left_gripper_squeeze_in, right_gripper_trigger_in, right_gripper_squeeze_in,
                                                dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        else:
            pass
        
        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)           # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # head/body-driven locomotion. Constructed after ChannelFactoryInitialize so the
        # publisher binds the right DDS domain (1 in sim).
        if args.head_loco:
            loco_tuning = LocoTuning(max_vx=args.loco_max_vx,
                                     max_vy=args.loco_max_vy,
                                     max_wz=args.loco_max_wz,
                                     walk_deadzone_ms=args.loco_walk_deadzone,
                                     walk_full_ms=args.loco_walk_full,
                                     neck_offset_m=args.loco_neck_offset,
                                     height=args.loco_height)
            loco_retarget = HeadLocomotionRetargeter(loco_tuning, yaw_mode=args.loco_yaw_source)
            # Latched off at startup: arming walking must always be a deliberate press.
            loco_toggle = LatchingToggle(args.loco_toggle_button) if args.loco_gate == 'toggle' else None
            loco_pub = LocomotionCommandPublisher(sim=args.sim,
                                                  loco_wrapper=loco_wrapper,
                                                  rate_hz=args.loco_rate,
                                                  sign=loco_sign,
                                                  accel_xy=args.loco_accel_xy,
                                                  accel_yaw=args.loco_accel_yaw,
                                                  height=args.loco_height)

        # Standalone from EpisodeWriter: its own process, queue, CSV, metadata and GUI.
        if args.head_relative_monitor:
            monitor_run_dir = os.path.join(
                args.head_relative_monitor_dir,
                head_relative_monitor_run_name(
                    sim=args.sim,
                    arm=args.arm,
                    ee=args.ee,
                    input_mode=args.input_mode,
                    arm_reference_mode=args.arm_reference_mode,
                    timestamp=datetime.now().strftime('%Y%m%d_%H%M%S'),
                ),
            )
            try:
                head_relative_monitor = HeadRelativeMonitor(
                    model=arm_ik.reduced_robot.model,
                    left_frame_id=arm_ik.L_hand_id,
                    right_frame_id=arm_ik.R_hand_id,
                    out_dir=monitor_run_dir,
                    metadata={
                        "arm": args.arm,
                        "ee": args.ee,
                        "input_mode": args.input_mode,
                        "arm_reference_mode": args.arm_reference_mode,
                        "frequency_hz": args.frequency,
                        "sim": args.sim,
                        "mapping": "unscaled_head_relative_xyz",
                        "head_origin_calibration": (
                            "g1_cad_head_midline"
                            if args.g1_head_origin_calibration
                            else "legacy_virtual_head"
                        ),
                        "monitor_target": "pre_residual_human_wrist_target",
                        "left_wrist_target_residual_m": left_wrist_target_residual.tolist(),
                        "right_wrist_target_residual_m": right_wrist_target_residual.tolist(),
                    },
                    window_seconds=args.head_relative_monitor_window,
                    plot_rate_hz=args.head_relative_monitor_rate,
                    show_plot=not (args.headless or args.head_relative_monitor_no_viewer),
                    robot_head_in_waist=active_robot_head_in_waist,
                )
            except Exception as exc:
                logger_mp.error(
                    f"Head-relative monitor failed to start ({exc}); teleoperation will continue."
                )

        # record + headless / non-headless mode
        if args.record:
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = not args.headless)

        # pose-error logging (independent of --record)
        if args.log_pose_error:
            pose_log_run_dir = os.path.join(
                args.pose_log_dir,
                f"{'sim' if args.sim else 'real'}_{args.arm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            pose_logger = PoseErrorLogger(
                arm_ik=arm_ik,
                out_dir=pose_log_run_dir,
                run_meta={
                    "arm": args.arm,
                    "ee": args.ee,
                    "sim": args.sim,
                    "motion": args.motion,
                    "debug_mode": args.debug_mode,
                    "arm_reference_mode": args.arm_reference_mode,
                    "frequency": args.frequency,
                    "input_mode": args.input_mode,
                    "head_origin_calibration": (
                        "g1_cad_head_midline"
                        if args.g1_head_origin_calibration
                        else "legacy_virtual_head"
                    ),
                    "robot_head_in_waist_m": active_robot_head_in_waist.tolist(),
                    "left_wrist_target_residual_m": left_wrist_target_residual.tolist(),
                    "right_wrist_target_residual_m": right_wrist_target_residual.tolist(),
                },
            )

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        if args.head_loco:
            arm_topic = "rt/arm_sdk" if args.motion else "rt/lowcmd"
            logger_mp.info("----------------------------------------------------------------")
            logger_mp.info("🚶  HEAD/BODY LOCOMOTION ENABLED")
            logger_mp.info(f"    target        : {'SIM' if args.sim else 'HARDWARE'}")
            logger_mp.info(f"    arm topic     : {arm_topic}")
            logger_mp.info(f"    locomotion    : {loco_pub.transport_description}")
            logger_mp.info(f"    wire sign     : {loco_sign}  (vx,vy,wz)")
            if loco_toggle is not None:
                logger_mp.info(f"    walk gate     : press [{loco_toggle.label}] to ARM/DISARM "
                               f"head-driven walking (starts DISARMED)")
            else:
                logger_mp.info(f"    walk gate     : hold [{args.loco_deadman}] to walk, release to STOP")
            logger_mp.info(f"    steering      : {args.loco_yaw_source}"
                           f"{'  (tilt head left/right)' if args.loco_yaw_source == 'roll' else ''}")
            logger_mp.info(f"    limits        : vx<={args.loco_max_vx} vy<={args.loco_max_vy} "
                           f"wz<={args.loco_max_wz}")
            # The speed-match gain is not any single flag -- it is max / (walk_full - deadzone),
            # so raising --loco-max-vx alone silently changes nothing below walk_full. Print the
            # gain and a worked example so a too-slow run is diagnosable from the banner.
            span = max(args.loco_walk_full - args.loco_walk_deadzone, 1e-6)
            gain_x, gain_y = args.loco_max_vx / span, args.loco_max_vy / span
            probe = 0.5   # a relaxed indoor walking pace
            logger_mp.info(f"    speed match   : gain {gain_x:.2f}x fwd / {gain_y:.2f}x lat above a "
                           f"{args.loco_walk_deadzone} m/s deadzone")
            logger_mp.info(f"                    you at {probe} m/s -> robot "
                           f"{min(max(probe - args.loco_walk_deadzone, 0.0) * gain_x, args.loco_max_vx):.2f} m/s "
                           f"(lower --loco-walk-full to go faster)")
            logger_mp.info(f"    accel         : {args.loco_accel_xy} m/s^2 xy, {args.loco_accel_yaw} rad/s^2 yaw "
                           f"({args.loco_max_vx / max(args.loco_accel_xy, 1e-6):.1f}s to reach max vx)")
            # Trained-envelope clamps in the publisher are applied last, so a CLI max above them
            # is silently truncated on the wire.
            for name, value, (lo, hi) in (("vx", args.loco_max_vx, VX_LIMITS),
                                          ("vy", args.loco_max_vy, VY_LIMITS),
                                          ("wz", args.loco_max_wz, WZ_LIMITS)):
                if value > hi:
                    logger_mp.warning(f"    ⚠️  --loco-max-{name}={value} exceeds the wire clamp "
                                      f"{hi}; forward/left commands are truncated to {hi}")
                if -value < lo:
                    logger_mp.warning(f"    ⚠️  --loco-max-{name}={value} exceeds the wire clamp "
                                      f"{lo}; backward/right commands are truncated to {lo}")
            if args.input_mode == "controller":
                logger_mp.info("    thumbsticks   : ALWAYS live, and override head-driven walking "
                               "while deflected")
            logger_mp.info("    walk to walk, stand still to stop (speed-matched);")
            logger_mp.info("    to cover more ground: release, walk back, re-engage.")
            logger_mp.info("⚠️  YOU ARE BLIND TO YOUR REAL SURROUNDINGS WHILE WALKING.")
            logger_mp.info("⚠️  Clear your space, set your guardian, and have a spotter.")
        READY = True                  # now ready to (1) enter START state
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)
            if xr_need_local_img:
                display_frames = {
                    name: img_client.get_frame(name)
                    for name in camera_display.displayed_camera_names
                }
                tv_wrapper.render_to_xr(camera_display.compose(display_frames))
            if camera_switch_gesture is not None:
                preview_tele_data = tv_wrapper.get_tele_data()
                update_camera_switch_gesture(
                    camera_switch_gesture, camera_display, preview_tele_data
                )

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        arm_ctrl.speed_gradual_max()
        if args.log_pose_error:
            pose_log_start_time = time.time()
            pose_log_seq = 0
        if head_relative_monitor is not None:
            head_relative_monitor_start_time = time.monotonic()
            head_relative_monitor_seq = 0
        # Started only after [r], so nothing can move the robot before the operator is ready.
        if loco_pub is not None:
            loco_pub.start()
        loco_dbg_i = 0

        head_img = None
        left_wrist_img = None
        right_wrist_img = None

        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            # Fetch each needed stream once. Display selection is generic; the three legacy
            # assignments below remain for the existing episode-recording schema.
            frame_names = list(camera_display.displayed_camera_names) if xr_need_local_img else []
            if args.record:
                for name in ('head_camera', 'left_wrist_camera', 'right_wrist_camera'):
                    if camera_config.get(name, {}).get('enable_zmq', False) and name not in frame_names:
                        frame_names.append(name)
            camera_frames = {name: img_client.get_frame(name) for name in frame_names}
            if xr_need_local_img:
                tv_wrapper.render_to_xr(camera_display.compose(camera_frames))
            head_img = camera_frames.get('head_camera')
            left_wrist_img = camera_frames.get('left_wrist_camera')
            right_wrist_img = camera_frames.get('right_wrist_camera')

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            update_camera_switch_gesture(camera_switch_gesture, camera_display, tele_data)
            if args.g1_head_origin_calibration:
                tele_data.left_wrist_pose = retarget_wrist_pose_head_origin(
                    tele_data.left_wrist_pose, active_robot_head_in_waist
                )
                tele_data.right_wrist_pose = retarget_wrist_pose_head_origin(
                    tele_data.right_wrist_pose, active_robot_head_in_waist
                )

            # The monitor retains the calibrated human target before applying
            # empirical robot corrections.  This keeps future CSV error values
            # honest: they measure the robot against the human, not against its
            # corrected IK command.
            monitor_left_wrist_pose = tele_data.left_wrist_pose.copy()
            monitor_right_wrist_pose = tele_data.right_wrist_pose.copy()
            tele_data.left_wrist_pose = apply_wrist_target_residual(
                tele_data.left_wrist_pose, left_wrist_target_residual
            )
            tele_data.right_wrist_pose = apply_wrist_target_residual(
                tele_data.right_wrist_pose, right_wrist_target_residual
            )

            # head/body-driven locomotion. Updated here, immediately after the tele data
            # arrives and before IK, so solver latency never delays the setpoint. The
            # publisher thread re-sends it at its own rate.
            if args.head_loco:
                # Arm the head-driven source: a held gesture in hand mode, a latched button in
                # controller mode. The toggle is sampled every frame so its rising edge is
                # never missed.
                if loco_toggle is not None:
                    was_armed = loco_toggle.state
                    walk_enable = loco_toggle.update(tele_data)
                    if walk_enable != was_armed:
                        logger_mp.info(f"[loco] head-driven walking "
                                       f"{'ARMED' if walk_enable else 'DISARMED'} "
                                       f"([{loco_toggle.label}])")
                else:
                    walk_enable = eval_deadman(tele_data, args.loco_deadman, args.input_mode)

                # The head retargeter is updated every frame regardless of who wins below, so
                # its speed estimate and engagement heading stay continuous -- otherwise
                # releasing the stick would hand over a stale setpoint.
                head_twist = loco_retarget.update(tele_data.head_pose, walk_enable, time.monotonic())

                # Arbitration: a deflected thumbstick is an explicit command and outranks the
                # head. Centred sticks return None, which is what separates "stick idle" from
                # "stick commanding a stop".
                stick_twist = (joystick_twist(tele_data, loco_tuning, args.loco_stick_deadzone)
                               if args.input_mode == "controller" else None)
                if stick_twist is not None:
                    twist, twist_src = stick_twist, "stick"
                else:
                    twist, twist_src = head_twist, "head"
                loco_pub.set_twist(twist)

                if args.loco_debug:
                    loco_dbg_i += 1
                    if loco_dbg_i % max(1, int(args.frequency // 2)) == 0:   # ~2 Hz
                        st = loco_retarget.status
                        gate = ('ARMED' if walk_enable else 'disarmed') if loco_toggle is not None \
                               else ('HELD' if walk_enable else 'open')
                        logger_mp.info(
                            f"[loco] src={twist_src} gate={gate} "
                            f"fwd={st['fwd_ms']:+.2f}m/s lat={st['lat_ms']:+.2f}m/s "
                            f"roll={st['roll_rad']:+.2f}rad -> "
                            f"vx={twist.vx:+.3f} vy={twist.vy:+.3f} wz={twist.wz:+.3f} "
                            f"({st['reason']})")

            if args.ee in ("dex3", "inspire_ftp", "inspire_dfx", "brainco") and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "brainco" and args.input_mode == "controller":
                with left_gripper_trigger_in.get_lock():
                    left_gripper_trigger_in.value = tele_data.left_ctrl_triggerValue
                with left_gripper_squeeze_in.get_lock():
                    left_gripper_squeeze_in.value = tele_data.left_ctrl_squeezeValue
                with right_gripper_trigger_in.get_lock():
                    right_gripper_trigger_in.value = tele_data.right_ctrl_triggerValue
                with right_gripper_squeeze_in.get_lock():
                    right_gripper_squeeze_in.value = tele_data.right_ctrl_squeezeValue
            elif args.ee == "dex1" and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee == "dex1" and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            with xr_motion_data_ready.get_lock():
                xr_motion_data_ready.value = tele_data.motion_data_ready
            
            # high level control
            if args.head_loco:
                # loco_pub is the single writer to the wire: the thumbsticks were already
                # arbitrated against the head source above and published through it, so the
                # direct loco_wrapper.Move() call below is deliberately skipped. That also
                # makes stick-driven walking work in sim, where there is no LocoClient at all.
                if args.input_mode == "controller":
                    # quit teleoperate
                    if tele_data.right_ctrl_aButton:
                        START = False
                        STOP = True
                    # DELIBERATELY DISABLED -- do not re-enable without asking.
                    # Upstream damps the robot when both thumbsticks are clicked. Damping
                    # cuts the motors, so the robot drops where it stands; the click is far
                    # too easy to trigger by accident while driving with the sticks, and the
                    # cost of a false positive is a fall. Stop with [q] or the A button, or
                    # use the hardware e-stop.
                    #
                    # if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    #     loco_pub.zero()
                    #     if loco_toggle is not None:
                    #         loco_toggle.state = False
                    #     if loco_wrapper is not None:
                    #         loco_wrapper.Enter_Damp_Mode()
            elif args.input_mode == "controller" and args.motion:
                # quit teleoperater
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
                # DELIBERATELY DISABLED -- do not re-enable without asking.
                # Upstream damps the robot when both thumbsticks are clicked. Damping cuts
                # the motors, so the robot drops where it stands; the click is far too easy
                # to trigger by accident while driving with the sticks, and the cost of a
                # false positive is a fall. Stop with [q] or the A button, or use the
                # hardware e-stop.
                #
                # (Upstream's own call is loco_wrapper.Damp(), which would raise
                # AttributeError anyway -- LocoClientWrapper only exposes Enter_Damp_Mode.)
                #
                # if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                #     loco_wrapper.Enter_Damp_Mode()
                # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
                loco_wrapper.Move(-tele_data.left_ctrl_thumbstickValue[1] * 0.3,
                                  -tele_data.left_ctrl_thumbstickValue[0] * 0.3,
                                  -tele_data.right_ctrl_thumbstickValue[0]* 0.3)

            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q, current_lr_arm_dq)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

            if head_relative_monitor is not None:
                head_relative_monitor.log(
                    seq=head_relative_monitor_seq,
                    t_rel=time.monotonic() - head_relative_monitor_start_time,
                    desired_left=monitor_left_wrist_pose,
                    desired_right=monitor_right_wrist_pose,
                    measured_q=current_lr_arm_q,
                    ik_ok=arm_ik.last_solve_ok,
                )
                head_relative_monitor_seq += 1

            # pose-error logging (independent of --record; see PoseErrorLogger)
            if args.log_pose_error:
                # G1_23 only (enforced at startup), so both readbacks are guaranteed present.
                pose_logger.log(
                    seq=pose_log_seq,
                    t_rel=start_time - pose_log_start_time,
                    desired_l=tele_data.left_wrist_pose,
                    desired_r=tele_data.right_wrist_pose,
                    sol_q=sol_q,
                    clipped_q=arm_ctrl.get_last_clipped_q_target(),
                    measured_q=current_lr_arm_q,
                    ik_ms=(time_ik_end - time_ik_start) * 1000.0,
                    ik_ok=arm_ik.last_solve_ok,
                    episode_id=(recorder.episode_id if args.record else -1),
                    record_state=('RECORDING' if RECORD_RUNNING else 'IDLE') if args.record else '',
                )
                pose_log_seq += 1

            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                elif (args.ee == "brainco" and args.input_mode == "controller"):
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # arm state and action (split into left/right halves by the arm's own DOF, so it works for any variant: H1/G1_23/R1_A5 = 4/5 per arm, G1_29/R1_A7 = 7)
                half = len(current_lr_arm_q) // 2
                left_arm_state,  right_arm_state  = current_lr_arm_q[:half], current_lr_arm_q[half:]
                left_arm_action, right_arm_action = sol_q[:half], sol_q[half:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr[:, :camera_config['head_camera']['image_shape'][1]//2]
                            colors[f"color_{1}"] = head_img.bgr[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{2}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{3}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    else:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{1}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{2}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        # FIRST: stop the robot walking. This outranks homing the arms, and covers every
        # exit path -- [q], right_ctrl_aButton, KeyboardInterrupt and unhandled exceptions.
        try:
            if loco_pub is not None:
                loco_pub.stop()
        except Exception as e:
            logger_mp.error(f"Failed to stop locomotion publisher: {e}")

        try:
            arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        
        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            if img_client is not None:
                img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if not args.motion:
                pass
                # status, result = motion_switcher.Exit_Debug_Mode()
                # logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if head_relative_monitor is not None:
                head_relative_monitor.close()
        except Exception as e:
            logger_mp.error(f"Failed to close head-relative monitor: {e}")

        try:
            if args.record:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")

        try:
            if args.log_pose_error:
                pose_logger.close()
        except Exception as e:
            logger_mp.error(f"Failed to close pose_logger: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
