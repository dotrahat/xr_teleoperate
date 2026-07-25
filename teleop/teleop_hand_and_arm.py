import time
import argparse
from multiprocessing import Value, Array, Lock
import threading
import logging_mp
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from teleop.utils.locomotion_retarget import HeadLocomotionRetargeter, LocoTuning, eval_deadman
from teleop.utils.locomotion_publisher import LocomotionCommandPublisher, parse_sign
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
    global STOP, START, RECORD_TOGGLE
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    # parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--img-server-ip', type=str, default='0.0.0.0', help='IP address of image server, used by teleimager and televuer')
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
                        help='Hold-to-walk gesture. Releasing it stops the robot immediately.')
    parser.add_argument('--loco-max-vx', type=float, default=0.40, help='Max forward/back speed (m/s)')
    parser.add_argument('--loco-max-vy', type=float, default=0.25, help='Max lateral speed (m/s)')
    parser.add_argument('--loco-max-wz', type=float, default=0.60, help='Max yaw rate (rad/s)')
    parser.add_argument('--loco-walk-deadzone', type=float, default=0.12, help='Operator walking speed ignored before the robot starts (m/s)')
    parser.add_argument('--loco-walk-full', type=float, default=0.80, help='Operator walking speed giving full robot speed (m/s)')
    parser.add_argument('--loco-neck-offset', type=float, default=0.10, help='Camera-to-neck-pivot distance (m)')
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

    logger_mp.info(f"args: {args}")

    # Defined before the try so the finally block can always reference them.
    loco_wrapper = None
    loco_retarget = None
    loco_pub = None

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
        # request_port must match ZMQ_Responser's bind port on the server (60000).
        # ImageClient's default of 60001 collides with the head camera's WebRTC port.
        img_client = ImageClient(host=args.img_server_ip, request_port=60000)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")
        xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=camera_config['head_camera']['enable_zmq'],
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
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
            arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1_2":
            arm_ik = H1_2_ArmIK()
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1":
            arm_ik = H1_ArmIK()
            arm_ctrl = H1_ArmController(simulation_mode=args.sim)

        # end-effector
        if args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "brainco":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock,
                                           dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim,
                                           input_mode=args.input_mode)
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
            loco_retarget = HeadLocomotionRetargeter(
                LocoTuning(max_vx=args.loco_max_vx,
                           max_vy=args.loco_max_vy,
                           max_wz=args.loco_max_wz,
                           walk_deadzone_ms=args.loco_walk_deadzone,
                           walk_full_ms=args.loco_walk_full,
                           neck_offset_m=args.loco_neck_offset,
                           height=args.loco_height),
                yaw_mode=args.loco_yaw_source)
            loco_pub = LocomotionCommandPublisher(sim=args.sim,
                                                  loco_wrapper=loco_wrapper,
                                                  rate_hz=args.loco_rate,
                                                  sign=loco_sign,
                                                  height=args.loco_height)

        # record + headless / non-headless mode
        if args.record:
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = not args.headless)

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
            logger_mp.info(f"    deadman       : hold [{args.loco_deadman}] to walk, release to STOP")
            logger_mp.info(f"    steering      : {args.loco_yaw_source}"
                           f"{'  (tilt head left/right)' if args.loco_yaw_source == 'roll' else ''}")
            logger_mp.info(f"    limits        : vx<={args.loco_max_vx} vy<={args.loco_max_vy} "
                           f"wz<={args.loco_max_wz}")
            logger_mp.info("    walk to walk, stand still to stop (speed-matched);")
            logger_mp.info("    to cover more ground: release, walk back, re-engage.")
            logger_mp.info("⚠️  YOU ARE BLIND TO YOUR REAL SURROUNDINGS WHILE WALKING.")
            logger_mp.info("⚠️  Clear your space, set your guardian, and have a spotter.")
        READY = True                  # now ready to (1) enter START state
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)
            if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                head_img, _ = img_client.get_head_frame()
                tv_wrapper.render_to_xr(head_img)

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        arm_ctrl.speed_gradual_max()
        # Started only after [r], so nothing can move the robot before the operator is ready.
        if loco_pub is not None:
            loco_pub.start()
        loco_dbg_i = 0
        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img:
                    head_img, head_img_fps = img_client.get_head_frame()
                if xr_need_local_img:
                    tv_wrapper.render_to_xr(head_img)
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record:
                    left_wrist_img, _ = img_client.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record:
                    right_wrist_img, _ = img_client.get_right_wrist_frame()

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

            # head/body-driven locomotion. Updated here, immediately after the tele data
            # arrives and before IK, so solver latency never delays the setpoint. The
            # publisher thread re-sends it at its own rate.
            if args.head_loco:
                walk_enable = eval_deadman(tele_data, args.loco_deadman, args.input_mode)
                twist = loco_retarget.update(tele_data.head_pose, walk_enable, time.monotonic())
                loco_pub.set_twist(twist)
                if args.loco_debug:
                    loco_dbg_i += 1
                    if loco_dbg_i % max(1, int(args.frequency // 2)) == 0:   # ~2 Hz
                        st = loco_retarget.status
                        logger_mp.info(
                            f"[loco] deadman={'HELD' if walk_enable else 'open'} "
                            f"fwd={st['fwd_ms']:+.2f}m/s lat={st['lat_ms']:+.2f}m/s "
                            f"roll={st['roll_rad']:+.2f}rad -> "
                            f"vx={twist.vx:+.3f} vy={twist.vy:+.3f} wz={twist.wz:+.3f} "
                            f"({st['reason']})")

            if (args.ee == "dex3" or args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "brainco" and args.input_mode == "controller":
                # Controller mode: trigger drives the four fingers, squeeze drives the thumb.
                # Element 0 = finger closure, element 1 = thumb closure; both [0,1] (0 open, 1 closed).
                # triggerValue is 10.0 (released) → 0.0 (fully pressed); squeezeValue is 0.0 → 1.0.
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[0] = 1.0 - tele_data.left_ctrl_triggerValue / 10.0
                    left_hand_pos_array[1] = tele_data.left_ctrl_squeezeValue
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[0] = 1.0 - tele_data.right_ctrl_triggerValue / 10.0
                    right_hand_pos_array[1] = tele_data.right_ctrl_squeezeValue
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
            
            # high level control
            if args.head_loco:
                # Head-loco owns the twist (published above), so the thumbstick path is
                # skipped -- two sources must never fight over the same command.
                if args.input_mode == "controller" and tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
            elif args.input_mode == "controller" and args.motion:
                # quit teleoperater
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    # loco_wrapper.Enter_Damp_Mode() # Damp()
                    print('no debug')
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

            # [DBG-G123] trace vertical tracking through the whole chain
            _dbg_i = locals().get('_dbg_i', 0) + 1
            if _dbg_i % 15 == 0:
                logger_mp.info(
                    f"[DBG-G123] L_tgt_z={tele_data.left_wrist_pose[2,3]:+.3f} "
                    f"R_tgt_z={tele_data.right_wrist_pose[2,3]:+.3f} | "
                    f"curr arm q[0,3,5,8]={current_lr_arm_q[0]:+.3f},{current_lr_arm_q[3]:+.3f},{current_lr_arm_q[5]:+.3f},{current_lr_arm_q[8]:+.3f} | "
                    f"sol L_sh_pitch={sol_q[0]:+.3f} L_elbow={sol_q[3]:+.3f} R_sh_pitch={sol_q[5]:+.3f} R_elbow={sol_q[8]:+.3f}"
                )

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
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img[:, :camera_config['head_camera']['image_shape'][1]//2]
                            colors[f"color_{1}"] = head_img[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{2}"] = left_wrist_img
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{3}"] = right_wrist_img
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    else:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{1}"] = left_wrist_img
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{2}"] = right_wrist_img
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
            if args.record:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)