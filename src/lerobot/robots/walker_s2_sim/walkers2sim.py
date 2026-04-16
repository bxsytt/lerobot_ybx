"""
Isaac Sim 环境的 walkerS2 双臂机器人控制类 (LeRobot V3 修复版)

修复列表:
  1. 移除重复的 is_connected 属性 (原第~840行), 保留唯一定义并在 disconnect() 中正确清理
  2. 移除重复的 cameras 属性 (原第~579行)
  3. 移除错误的 observation_features 定义 (原第~963行, 引用不存在的 self.config.state_names)
  4. 修复 disconnect() 中 _ros2_teleop 双重 None 逻辑错误
  5. connect() 末尾添加 _register_world_callbacks() 调用
  6. 修复 _init_keyboard_listener() 中未定义的 EVDEV_AVAILABLE / EvdevKeyboardListener
  7. IsaacSimRobotInterface 补充三个缺失方法:
       set_arm_joint_positions(), set_finger_positions(), apply_finger_efforts()
  8. 修复 _on_key_release 中 switch_control_arm key 不一致问题
"""

from __future__ import annotations
import logging
import time
import os
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path
import yaml
import numpy as np
import torch
import threading
import collections
from functools import cached_property

from lerobot.robots.config import RobotConfig
from .walkers2simConfig import WalkerS2Config
from lerobot.robots.robot import Robot

# pynput 按需导入
try:
    from pynput import keyboard as pynput_keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    pynput_keyboard = None
    PYNPUT_AVAILABLE = False

# evdev 按需导入 (Bug修复#6: 原代码直接使用 EVDEV_AVAILABLE 未定义就报 NameError)
try:
    import evdev
    EVDEV_AVAILABLE = True
except ImportError:
    evdev = None
    EVDEV_AVAILABLE = False

logger = logging.getLogger(__name__)


class RobotDeviceNotConnectedError(Exception):
    pass


# ---------------------------------------------------------------------------
# EvdevKeyboardListener: 仅在 evdev 可用时使用
# ---------------------------------------------------------------------------
if EVDEV_AVAILABLE:
    class EvdevKeyboardListener:
        """通过 /dev/input 直接读取键盘事件，绕过 X11，适合 Docker/RustDesk/VNC 场景。"""

        # evdev key code → 字符的最简映射（数字行 + 常用字母）
        _EVDEV_KEY_CHAR_MAP: dict[str, str] = {
            "KEY_1": "1", "KEY_2": "2", "KEY_3": "3", "KEY_4": "4",
            "KEY_5": "5", "KEY_6": "6", "KEY_7": "7", "KEY_8": "8",
            "KEY_9": "9", "KEY_0": "0", "KEY_MINUS": "-", "KEY_EQUAL": "=",
            "KEY_Q": "q", "KEY_W": "w", "KEY_E": "e", "KEY_R": "r",
            "KEY_T": "t", "KEY_Y": "y", "KEY_U": "u", "KEY_I": "i",
            "KEY_O": "o", "KEY_P": "p", "KEY_A": "a", "KEY_S": "s",
            "KEY_D": "d", "KEY_F": "f", "KEY_G": "g", "KEY_H": "h",
            "KEY_J": "j", "KEY_K": "k", "KEY_L": "l", "KEY_Z": "z",
            "KEY_X": "x", "KEY_C": "c", "KEY_V": "v", "KEY_B": "b",
            "KEY_N": "n", "KEY_M": "m",
        }

        def __init__(self, on_press, on_release):
            self._on_press = on_press
            self._on_release = on_release
            self._thread: Optional[threading.Thread] = None
            self._running = False
            self._device = None

        def start(self):
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            keyboards = [d for d in devices if evdev.ecodes.EV_KEY in d.capabilities()]
            if not keyboards:
                raise RuntimeError("未找到 evdev 键盘设备")
            self._device = keyboards[0]
            logger.info(f"evdev 使用设备: {self._device.name}")
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

        def _loop(self):
            for event in self._device.read_loop():
                if not self._running:
                    break
                if event.type == evdev.ecodes.EV_KEY:
                    key_event = evdev.categorize(event)
                    char = self._EVDEV_KEY_CHAR_MAP.get(key_event.keycode)
                    if char is None:
                        continue

                    class _FakeKey:
                        pass

                    fake = _FakeKey()
                    fake.char = char
                    if key_event.keystate == evdev.events.KeyEvent.key_down:
                        self._on_press(fake)
                    elif key_event.keystate == evdev.events.KeyEvent.key_up:
                        self._on_release(fake)

        def stop(self):
            self._running = False


@dataclass
class TimingMetric:
    count: int = 0
    total_s: float = 0.0
    min_s: float = field(default_factory=lambda: float("inf"))
    max_s: float = 0.0

    def update(self, duration_s: float) -> None:
        duration_s = max(float(duration_s), 0.0)
        self.count += 1
        self.total_s += duration_s
        self.min_s = min(self.min_s, duration_s)
        self.max_s = max(self.max_s, duration_s)

    def as_dict(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"count": 0, "avg_s": 0.0, "min_s": 0.0, "max_s": 0.0}
        return {
            "count": self.count,
            "avg_s": self.total_s / self.count,
            "min_s": self.min_s,
            "max_s": self.max_s,
        }


def load_config(config_path: str) -> dict:
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    yaml_dir = os.path.dirname(config_path)
    if "root_path" in cfg:
        cfg["root_path"] = os.path.abspath(os.path.join(yaml_dir, cfg["root_path"]))
    return cfg


def load_isaac_sim_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    root_dir = Path(config_path).parent.parent  # 假设 config 在 ecbg/config 下，root 在 ecbg 下

    def fix_paths(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, str) and (v.startswith("../") or v.startswith("./")):
                    d[k] = str((root_dir / v).resolve())
                else:
                    fix_paths(v)

    fix_paths(cfg)
    return cfg


# ---------------------------------------------------------------------------
# ROS2 适配层
# ---------------------------------------------------------------------------
ROS2_AVAILABLE = False
try:
    import rclpy
    ROS2_AVAILABLE = True
except ImportError:
    rclpy = None


class ROS2TeleopSubscriber:
    def __init__(self, arm_joint_names: list[str], finger_joint_names: list[str], topic: str):
        self._arm_joint_names = arm_joint_names
        self._finger_joint_names = finger_joint_names
        self._topic = topic
        self._lock = threading.Lock()
        self._latest_arm_positions: Optional[np.ndarray] = None
        self._latest_finger_positions: Optional[np.ndarray] = None
        self._latest_timestamp: Optional[float] = None
        self._node = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> bool:
        if not ROS2_AVAILABLE:
            return False
        try:
            if not rclpy.ok():
                rclpy.init()
            from sensor_msgs.msg import JointState
            self._node = rclpy.create_node("isaac_pico4_teleop_subscriber")
            self._node.create_subscription(JointState, self._topic, self._on_joint_state, 10)
            self._running = True
            self._thread = threading.Thread(target=self._spin_loop, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            print(f"ROS2 启动失败: {e}")
            return False

    def _spin_loop(self):
        while self._running and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.01)

    def _on_joint_state(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        arm_positions = [name_to_pos[n] for n in self._arm_joint_names if n in name_to_pos]
        finger_positions = [name_to_pos[n] for n in self._finger_joint_names if n in name_to_pos]

        if len(arm_positions) == len(self._arm_joint_names):
            with self._lock:
                self._latest_arm_positions = np.array(arm_positions, dtype=np.float32)
                if len(finger_positions) == len(self._finger_joint_names):
                    self._latest_finger_positions = np.array(finger_positions, dtype=np.float32)
                self._latest_timestamp = time.time()

    def get_latest(self, max_age_s: float = 0.5):
        with self._lock:
            if self._latest_arm_positions is None or (
                time.time() - (self._latest_timestamp or 0) > max_age_s
            ):
                return None
            return {
                "arm_positions": self._latest_arm_positions.copy(),
                "finger_positions": (
                    self._latest_finger_positions.copy()
                    if self._latest_finger_positions is not None
                    else None
                ),
            }

    def stop(self):
        self._running = False
        if self._node:
            self._node.destroy_node()


# ---------------------------------------------------------------------------
# IsaacSimRobotInterface
# ---------------------------------------------------------------------------
class IsaacSimRobotInterface:
    """与 Isaac Sim 的底层接口交互类"""

    arm_joint_names: list[str] = [
        "L_shoulder_pitch_joint", "L_shoulder_roll_joint", "L_shoulder_yaw_joint",
        "L_elbow_roll_joint", "L_elbow_yaw_joint", "L_wrist_pitch_joint", "L_wrist_roll_joint",
        "R_shoulder_pitch_joint", "R_shoulder_roll_joint", "R_shoulder_yaw_joint",
        "R_elbow_roll_joint", "R_elbow_yaw_joint", "R_wrist_pitch_joint", "R_wrist_roll_joint",
    ]
    finger_joint_names: list[str] = [
        "L_finger1_joint", "L_finger2_joint", "R_finger1_joint", "R_finger2_joint"
    ]
    gripper_open_width: float = 0.06
    gripper_close_width: float = 0.02
    gripper_close_tau: float = 100.0   
    sixforce_joint_names: list[str] = ["L_sixforce_joint", "R_sixforce_joint"]

    def __init__(
        self,
        prim_path: str,
        name: str = "walkerS2",
        world: Any = None,
        urdf_path: Optional[str] = None,
    ):
        self.prim_path = prim_path
        self.name = name
        self._world = world
        self._articulation = None
        self.time = 0.0
        self.urdf_path = urdf_path
        self.arm_joint_indices: list[int] = []
        self.finger_joint_indices: list[int] = []
        self.cameras = {}
        self._camera_prim_paths = {
            "head_left": f"{prim_path}/head_pitch_link/head_stereo_left/head_stereo_left_Camera_01",
            "head_right": f"{prim_path}/head_pitch_link/head_stereo_right/head_stereo_right_Camera_01",
            "wrist_left": f"{prim_path}/L_camera_link/L_camera_link/L_wrist_camera/L_wrist_Camera",
            "wrist_right": f"{prim_path}/R_camera_link/R_camera_link/R_wrist_camera/R_wrist_Camera",
        }
        self.initial_joint_positions = None
        self.ik_solver: Optional[Any] = None
        self._left_arm_isaac_indices = None
        self._right_arm_isaac_indices = None
        self._waist_isaac_indices = None
        self._waist_init_positions = None
        self._ik_warn_counter = 0
        self._smooth_alpha = 0.3
        self._last_arm_positions = {}
        self._joint_value_map = {
            "L_elbow_roll_joint": -1.8963565338596158,
            "L_elbow_yaw_joint": 1.4000461262831179,
            "L_shoulder_pitch_joint": 0.09322471888572098,
            "L_shoulder_roll_joint": -0.5933223843430208,
            "L_shoulder_yaw_joint": -1.595878574835185,
            "L_wrist_pitch_joint": -0.00048740902645395785,
            "L_wrist_roll_joint": 0.0998718010009366,
            "R_elbow_roll_joint": -1.8963607249359917,
            "R_elbow_yaw_joint": -1.4000874256427638,
            "R_shoulder_pitch_joint": -0.09321727661087699,
            "R_shoulder_roll_joint": -0.5933455607833843,
            "R_shoulder_yaw_joint": 1.595869459316937,
            "R_wrist_pitch_joint": 0.00048144049606466176,
            "R_wrist_roll_joint": 0.09985407619802703,
            "head_pitch_joint": -0.600945933438922,
            "head_yaw_joint": 1.9677590016147396e-07,
        }

    def initialize(self):
        from isaacsim.core.prims import Articulation

        logger.info(f"连接到 Articulation: {self.prim_path}")
        self._articulation = Articulation(
            prim_paths_expr=self.prim_path,
            name=self.name,
        )
        self._articulation.initialize()

        all_joint_names = self._articulation.dof_names
        logger.info(f"机器人总关节数: {len(all_joint_names)}")

        self.arm_joint_indices = []
        for arm_joint in self.arm_joint_names:
            if arm_joint in all_joint_names:
                self.arm_joint_indices.append(all_joint_names.index(arm_joint))

        self.finger_joint_indices = []
        for finger_joint in self.finger_joint_names:
            if finger_joint in all_joint_names:
                self.finger_joint_indices.append(all_joint_names.index(finger_joint))

        self.initial_joint_positions = [0.0] * len(all_joint_names)
        for i, joint_name in enumerate(all_joint_names):
            if joint_name in self._joint_value_map:
                self.initial_joint_positions[i] = self._joint_value_map[joint_name]

        s2_joint_indices = [self._articulation.get_dof_index(name) for name in all_joint_names]
        self._articulation.set_joint_positions(
            torch.tensor(self.initial_joint_positions),
            joint_indices=torch.tensor(s2_joint_indices),
        )

        self._setup_cameras()
        self.initialize_ik(urdf_path=self.urdf_path)

    def _setup_cameras(self):
        from isaacsim.sensors.camera import Camera

        for cam_name, prim_path in self._camera_prim_paths.items():
            try:
                self.cameras[cam_name] = Camera(prim_path=prim_path, resolution=(640, 480))
                self.cameras[cam_name].initialize()
                self.cameras[cam_name].add_distance_to_image_plane_to_frame()
            except Exception as e:
                print(f"相机 {cam_name} 初始化跳过: {e}")

        try:
            self.cameras["dummy_camera_top"] = Camera(
                prim_path="/Root/Ref_Xform/Ref/head_pitch_link/head_stereo_left/dummy_camera_top",
                translation=np.array([-2.54145, -0.06363, 2.4821]),
                orientation=np.array([0.915, -0.008, 0.403, 0.006]),
            )
            self.cameras["dummy_camera_top"].initialize()
        except Exception as e:
            print(f"虚拟顶部相机创建失败: {e}")

        try:
            self.cameras["dummy_camera_side"] = Camera(
                prim_path="/Replicator/Ref_Xform/Ref/dummy_camera_side",
                translation=np.array([2.06555, -0.02631, 0.95453]),
                orientation=np.array([-5.943e-03, -3.2476e-02, -2.01e-04, 9.99455e-01]),
            )
            self.cameras["dummy_camera_side"].initialize()
        except Exception as e:
            print(f"虚拟侧面相机创建失败: {e}")

    def reset(self):
        if self._articulation is None:
            return
        if self.initial_joint_positions is not None:
            pos_tensor = torch.tensor(self.initial_joint_positions, dtype=torch.float32)
            all_indices = torch.arange(self._articulation.num_dof, dtype=torch.int32)
            self._articulation.set_joint_positions(pos_tensor, joint_indices=all_indices)
            vel_zero = torch.zeros(self._articulation.num_dof, dtype=torch.float32)
            self._articulation.set_joint_velocities(vel_zero, joint_indices=all_indices)

        self.time = 0.0
        self._ik_warn_counter = 0
        self._last_arm_positions.clear()

        if self.ik_solver is not None:
            self.ik_solver.reset_runtime_state()
            all_names = self._articulation.dof_names
            curr_pos = self._articulation.get_joint_positions().flatten().tolist()
            self.ik_solver.sync_joint_positions(all_names, curr_pos)
            self.ik_solver.save_initial_q()

    def cleanup(self):
        self._articulation = None
        self.cameras.clear()

    def get_joint_states(self):
        if self._articulation is None:
            return None
        try:
            all_names = self._articulation.dof_names
            all_joint_positions = self._articulation.get_joint_positions().flatten()
            all_joint_velocities = self._articulation.get_joint_velocities().flatten()
            all_joint_efforts = self._articulation.get_measured_joint_efforts().flatten()

            all_joint_positions = torch.tensor(all_joint_positions, dtype=torch.float32)
            all_joint_velocities = torch.tensor(all_joint_velocities, dtype=torch.float32)
            all_joint_efforts = torch.tensor(all_joint_efforts, dtype=torch.float32)

            arm_indices = torch.tensor(self.arm_joint_indices, dtype=torch.long)
            finger_indices = torch.tensor(self.finger_joint_indices, dtype=torch.long)

            return {
                "all_names": all_names,
                "all_positions": all_joint_positions.tolist(),
                "arm_names": self.arm_joint_names,
                "arm_positions": all_joint_positions[arm_indices].tolist(),
                "arm_velocities": all_joint_velocities[arm_indices].tolist(),
                "arm_torques": all_joint_efforts[arm_indices].tolist(),
                "finger_names": self.finger_joint_names,
                "finger_positions": all_joint_positions[finger_indices].tolist(),
            }
        except Exception:
            return None

    # Bug修复#7: 补充 _robot_control_callback 中调用但缺失的三个方法 -------

    def set_arm_joint_positions(
        self, target_arm_positions: list[float], task_num: int = 1
    ) -> None:
        """将 14 个手臂关节设置到目标位置（位置控制）。"""
        if self._articulation is None:
            return
        from isaacsim.core.utils.types import ArticulationActions

        positions = torch.tensor([target_arm_positions], dtype=torch.float32)
        indices = torch.tensor(self.arm_joint_indices, dtype=torch.int32)
        self._articulation.apply_action(
            ArticulationActions(joint_positions=positions, joint_indices=indices)
        )

    def set_finger_positions(
        self, target_fingers: list[float], task_num: int = 1
    ) -> None:
        """将 4 个夹爪关节设置到目标位置（位置控制）。

        target_fingers 中为 NaN 的关节跳过（夹持模式下由 apply_finger_efforts 接管）。
        """
        if self._articulation is None:
            return
        from isaacsim.core.utils.types import ArticulationActions

        valid_positions, valid_indices = [], []
        for i, (pos, idx) in enumerate(zip(target_fingers, self.finger_joint_indices)):
            if not (pos != pos):  # NaN 检测
                valid_positions.append(pos)
                valid_indices.append(idx)

        if not valid_positions:
            return

        positions = torch.tensor([valid_positions], dtype=torch.float32)
        indices = torch.tensor(valid_indices, dtype=torch.int32)
        self._articulation.apply_action(
            ArticulationActions(joint_positions=positions, joint_indices=indices)
        )

    def apply_finger_efforts(self, efforts: list[float]) -> None:
        """对 4 个夹爪关节施加力矩（力控模式，夹持物体时使用）。

        efforts 中为 0.0 的关节不施力。
        """
        if self._articulation is None:
            return
        from isaacsim.core.utils.types import ArticulationActions

        nonzero_efforts, nonzero_indices = [], []
        for effort, idx in zip(efforts, self.finger_joint_indices):
            if abs(effort) > 1e-6:
                nonzero_efforts.append(effort)
                nonzero_indices.append(idx)

        if not nonzero_efforts:
            return

        effort_tensor = torch.tensor([nonzero_efforts], dtype=torch.float32)
        indices = torch.tensor(nonzero_indices, dtype=torch.int32)
        self._articulation.apply_action(
            ArticulationActions(joint_efforts=effort_tensor, joint_indices=indices)
        )

    # -----------------------------------------------------------------------

    def close_gripper(self, side: Optional[str] = None):
        target = [self.gripper_close_width] * 2
        indices = (
            self.finger_joint_indices[:2] if side == "left" else self.finger_joint_indices[2:4]
        )
        self._articulation.set_joint_positions(
            target, joint_indices=torch.tensor(indices, dtype=torch.int32)
        )

    def open_gripper(self, side: Optional[str] = None):
        target = [self.gripper_open_width] * 2
        indices = (
            self.finger_joint_indices[:2] if side == "left" else self.finger_joint_indices[2:4]
        )
        self._articulation.set_joint_positions(
            target, joint_indices=torch.tensor(indices, dtype=torch.int32)
        )

    def get_camera_rgbd(self, camera_name):
        return {
            "rgb": self.cameras[camera_name].get_rgb() if camera_name in self.cameras else None,
            "depth": (
                self.cameras[camera_name].get_depth() if camera_name in self.cameras else None
            ),
            "camera_name": camera_name,
        }

    def get_sixforce(self):
        wrench_data_list = []
        sensor_joint_forces = self._articulation.get_measured_joint_forces()[0]
        for joint_name in self.sixforce_joint_names:
            idx = self._articulation.get_joint_index(joint_name)
            sixforce_data = sensor_joint_forces[idx + 1]
            wrench_data_list.append(
                {
                    "frame_id": joint_name,
                    "force": sixforce_data[:3].tolist(),
                    "torque": sixforce_data[3:].tolist(),
                }
            )
        return wrench_data_list

    def initialize_ik(self, urdf_path: str):
        from Ubtech_sim.source.DualArmIK import DualArmIK

        self.ik_solver = DualArmIK(urdf_path)
        dof_names = self._articulation.dof_names
        self._left_arm_isaac_indices = [
            self._articulation.get_dof_index(j)
            for j in DualArmIK.LEFT_ARM_JOINTS
            if j in dof_names
        ]
        self._right_arm_isaac_indices = [
            self._articulation.get_dof_index(j)
            for j in DualArmIK.RIGHT_ARM_JOINTS
            if j in dof_names
        ]
        self._waist_isaac_indices, self._waist_init_positions = [], []
        for jname in ["waist_yaw_joint", "waist_pitch_joint"]:
            if jname in dof_names:
                self._waist_isaac_indices.append(self._articulation.get_dof_index(jname))
                self._waist_init_positions.append(0.0)

        if self._waist_isaac_indices:
            self._articulation.set_joint_positions(
                torch.tensor(self._waist_init_positions, dtype=torch.float32),
                joint_indices=torch.tensor(self._waist_isaac_indices, dtype=torch.int32),
            )

        joints = self.get_joint_states()
        if joints:
            self.ik_solver.sync_joint_positions(joints["all_names"], joints["all_positions"])
        self.ik_solver.save_initial_q()
        self.ik_solver.set_neutral_config(
            [self._joint_value_map.get(j, 0.0) for j in DualArmIK.LEFT_ARM_JOINTS],
            [self._joint_value_map.get(j, 0.0) for j in DualArmIK.RIGHT_ARM_JOINTS],
        )

    def get_ee_poses(self):
        joints = self.get_joint_states()
        if not joints or not self.ik_solver:
            return None
        self.ik_solver.sync_joint_positions(joints["all_names"], joints["all_positions"])
        return self.ik_solver.get_both_ee_poses()

    def control_dual_arm_ik(
        self,
        step_size: float,
        left_target_xyzrpy=None,
        right_target_xyzrpy=None,
        **ik_kwargs,
    ):
        from isaacsim.core.utils.types import ArticulationActions

        self.time += step_size
        joints = self.get_joint_states()
        if not joints:
            return
        ik_result = self.ik_solver.solve_dual_arm(
            left_target_xyzrpy=left_target_xyzrpy,
            right_target_xyzrpy=right_target_xyzrpy,
            isaac_joint_names=joints["all_names"],
            isaac_joint_positions=joints["all_positions"],
            **ik_kwargs,
        )

        all_indices, all_positions = [], []
        if "left_joint_positions" in ik_result:
            all_indices.extend(self._left_arm_isaac_indices)
            all_positions.extend(
                self._smooth_joints("left", ik_result["left_joint_positions"]).tolist()
            )
        if "right_joint_positions" in ik_result:
            all_indices.extend(self._right_arm_isaac_indices)
            all_positions.extend(
                self._smooth_joints("right", ik_result["right_joint_positions"]).tolist()
            )

        if self._waist_isaac_indices:
            self._articulation.set_joint_positions(
                torch.tensor(self._waist_init_positions, dtype=torch.float32),
                joint_indices=torch.tensor(self._waist_isaac_indices, dtype=torch.int32),
            )

        if all_indices:
            self._articulation.apply_action(
                ArticulationActions(
                    joint_positions=torch.tensor([all_positions], dtype=torch.float32),
                    joint_indices=torch.tensor(all_indices, dtype=torch.int32),
                )
            )
        # 修复 2: 将 all_positions 存入返回结果中
        # 否则上层的 "smoothed_positions" 检查永远进不去，
        # 导致机械臂每帧都被强行锁定在原地。
        # ==========================================
        if ik_result is not None:
            ik_result["smoothed_positions"] = all_positions

        return ik_result

    def _smooth_joints(self, side: str, ik_positions: np.ndarray) -> np.ndarray:
        ik_positions = np.asarray(ik_positions, dtype=float)
        if side not in self._last_arm_positions:
            self._last_arm_positions[side] = ik_positions.copy()
            return ik_positions
        smoothed = self._last_arm_positions[side] + self._smooth_alpha * (
            ik_positions - self._last_arm_positions[side]
        )
        self._last_arm_positions[side] = smoothed.copy()
        return smoothed


# ---------------------------------------------------------------------------
# WalkerS2sim
# ---------------------------------------------------------------------------
class WalkerS2sim(Robot):
    """walkerS2 双臂机器人控制类 - LeRobot V3 修复版"""

    robot_type: str = "walker_s2_sim"
    name: str = "walkerS2"
    CAMERA_NAMES = ["head_left", "head_right", "wrist_left", "wrist_right"]

    STATE_DIM = 20
    STATE_NAMES = [
        "L_shoulder_pitch_joint.pos", "L_shoulder_roll_joint.pos", "L_shoulder_yaw_joint.pos",
        "L_elbow_roll_joint.pos", "L_elbow_yaw_joint.pos", "L_wrist_pitch_joint.pos",
        "L_wrist_roll_joint.pos",
        "R_shoulder_pitch_joint.pos", "R_shoulder_roll_joint.pos", "R_shoulder_yaw_joint.pos",
        "R_elbow_roll_joint.pos", "R_elbow_yaw_joint.pos", "R_wrist_pitch_joint.pos",
        "R_wrist_roll_joint.pos",
        "L_finger1_joint.pos", "L_finger2_joint.pos", "R_finger1_joint.pos", "R_finger2_joint.pos",
        "left_gripper_control", "right_gripper_control",
    ]

    ACTION_DIM = 20
    ACTION_NAMES = STATE_NAMES  # action 与 state 维度相同

    MOTION_AXES = ("x", "y", "z", "rx", "ry", "rz")
    BIMANUAL_MIRROR_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float32)

    enable_ros2_teleop: bool = True
    ros2_joint_commands_topic: str = "/isaac/joint_position_commands"

    def __init__(self, config: WalkerS2Config | dict | None = None):
        if isinstance(config, dict):
            config = WalkerS2Config(**config)
        elif config is None:
            config = WalkerS2Config()

        super().__init__(config)
        self.config = config
        self.logs = {}

        # self.current_control_arm = "left" # "left" or "right"
        # self.bimanual_control_enabled = False # 默认单臂模式

        # Bug修复#1: 只保留一个 _is_connected 标志，所有状态检查都用它
        self._is_connected = False

        self._timing_metrics: dict[str, TimingMetric] = {
            "send_action": TimingMetric(),
            "get_observation": TimingMetric(),
            "dt_s": TimingMetric(),
        }

        if os.path.exists(self.config.task_cfg_path):
            self.config.task_cfg = load_isaac_sim_config(self.config.task_cfg_path)
            logger.info(f"成功加载并转换 YAML 配置: {self.config.task_cfg_path}")

        self.current_control_arm: str = "left"
        self.gripper_control_mode: str = "position"
        self.bimanual_control_enabled: bool = False

        if not hasattr(self.config, "camera_width"):
            self.config.camera_width = 640
        if not hasattr(self.config, "camera_height"):
            self.config.camera_height = 480

        self._kit = None
        self._world = None
        self._scene_builder = None
        self._data_logger = None
        self._robot_interface: Optional[IsaacSimRobotInterface] = None

        self._arm_joint_indices: list[int] = []
        self._current_positions: torch.Tensor | None = None

        self._speed_index = self.config.default_speed_index
        self._pressed_keys: dict[str, bool] = {}
        self._keyboard_listener = None

        self._render_every_n = 1
        self._send_action_step_idx = 0

        self._callback_lock = threading.Lock()
        self._callbacks_registered = False
        self._pending_absolute_action: Optional[np.ndarray] = None
        self._last_keyboard_frame_id: int = -1
        self._latest_camera_rgb: dict[str, np.ndarray] = {}
        self._hold_arm_positions: Optional[np.ndarray] = None
        self._hold_finger_positions: Optional[np.ndarray] = None
        self._left_gripping: bool = False
        self._right_gripping: bool = False

        self._keyboard_cmd_queue: collections.deque[dict[str, bool]] = collections.deque(
            maxlen=256
        )
        self._current_frame_keys: Optional[dict[str, bool]] = None

        self._ros2_teleop: Optional[ROS2TeleopSubscriber] = None

    # Bug修复#1+#2: 只保留一个 is_connected 属性，移除重复定义
    @property
    def is_connected(self) -> bool:
        return self._is_connected

    # Bug修复#2: 只保留一个 cameras 属性，返回 dict（兼容 lerobot v3 框架）
    @property
    def cameras(self) -> dict:
        return self.config.cameras if hasattr(self.config, "cameras") else {}

    def step(self, render: bool = True):
        if not self.is_connected or self._world is None:
            raise RobotDeviceNotConnectedError("未连接")
        self._world.step(render=render)
        self._send_action_step_idx += 1

    def toggle_bimanual_mode(self):
        """由 teleop 调用，切换模式"""
        self.bimanual_control_enabled = not self.bimanual_control_enabled
        mode_str = "双臂同步" if self.bimanual_control_enabled else "单臂独立"
        logging.info(f"控制模式已切换: {mode_str}")

    # def toggle_bimanual_mode(self):
    #     """由 Teleop 调用的切换接口"""
    #     self.bimanual_control_enabled = not self.bimanual_control_enabled
    #     status = "开启" if self.bimanual_control_enabled else "关闭"
    #     logging.info(f"【控制模式】双臂同步控制已 {status}")

    # ---- 回调注册 / 注销 ----
    def _register_world_callbacks(self) -> None:
        if self._world is None or self._callbacks_registered:
            return
        self._world.add_physics_callback("robot_control", self._robot_control_callback)
        self._world.add_physics_callback("score_input_record", self._score_input_record_callback)
        self._world.add_physics_callback("foam_sync", self._foam_sync_callback)
        self._world.add_render_callback("camera_images", self._camera_images_callback)
        self._callbacks_registered = True
        logger.info("已注册物理/渲染回调")

    def _unregister_world_callbacks(self) -> None:
        if self._world is None or not self._callbacks_registered:
            return
        remove_physics = getattr(self._world, "remove_physics_callback", None)
        remove_render = getattr(self._world, "remove_render_callback", None)
        if callable(remove_physics):
            for cb_name in ["robot_control", "score_input_record", "foam_sync"]:
                try:
                    remove_physics(cb_name)
                except Exception:
                    pass
        if callable(remove_render):
            try:
                remove_render("camera_images")
            except Exception:
                pass
        self._callbacks_registered = False

    # ---- 回调实现 ----
    def _robot_control_callback(self, step_size: float) -> None:
        if not self.is_connected:
            return

        if self._hold_arm_positions is None:
            states = self._robot_interface.get_joint_states()
            if states:
                self._hold_arm_positions = np.array(states["arm_positions"], dtype=np.float32)
                self._hold_finger_positions = np.array(states["finger_positions"], dtype=np.float32)
                logger.info("[callback] 已快照初始关节状态作为保持目标")
            else:
                return

        with self._callback_lock:
            abs_action = self._pending_absolute_action
            if abs_action is not None:
                abs_action = abs_action.copy()
                self._pending_absolute_action = None

        if abs_action is not None:
            # 推理/回放模式
            self._hold_arm_positions = abs_action[:14].copy()
            if abs_action.shape[0] >= 18:
                self._hold_finger_positions = abs_action[14:18].copy()
            if abs_action.shape[0] >= 20:
                self._left_gripping = float(abs_action[18]) > 0
                self._right_gripping = float(abs_action[19]) > 0
        else:
            # 遥操作模式
            current_frame = self._send_action_step_idx
            if current_frame != self._last_keyboard_frame_id:
                self._last_keyboard_frame_id = current_frame

                key_snapshot: dict[str, bool] = {}
                while self._keyboard_cmd_queue:
                    snap = self._keyboard_cmd_queue.popleft()
                    for k, v in snap.items():
                        if v:
                            key_snapshot[k] = True
                for k, v in self._pressed_keys.items():
                    if v:
                        key_snapshot[k] = True

                self._current_frame_keys = key_snapshot

                ros2_data = None
                if self._ros2_teleop is not None:
                    ros2_data = self._ros2_teleop.get_latest(max_age_s=0.5)

                if ros2_data is not None:
                    self._hold_arm_positions = np.asarray(
                        ros2_data["arm_positions"], dtype=np.float32
                    ).copy()
                    if ros2_data["finger_positions"] is not None:
                        self._hold_finger_positions = np.asarray(
                            ros2_data["finger_positions"], dtype=np.float32
                        ).copy()
                    _, _, left_gripper, right_gripper, _ = self._compute_keyboard_delta(
                        key_snapshot
                    )
                    if ros2_data["finger_positions"] is None:
                        gripper_step = 0.002
                        g_open = self._robot_interface.gripper_open_width
                        g_close = self._robot_interface.gripper_close_width
                        g_lo, g_hi = min(g_open, g_close), max(g_open, g_close)
                        if abs(left_gripper) > 0.01:
                            # 【Bug修复】：当 left_gripper 为 1.0 (闭合)时，应该减小宽度趋近 g_close (0.02)
                            # 所以这里用减号：- (1.0 * 0.002) = 变小
                            self._hold_finger_positions[:2] = np.clip(
                                self._hold_finger_positions[:2] - left_gripper * gripper_step,
                                g_lo,
                                g_hi,
                            )
                            self._left_gripping = left_gripper > 0
                        if abs(right_gripper) > 0.01:
                            self._hold_finger_positions[2:4] = np.clip(
                                self._hold_finger_positions[2:4] - right_gripper * gripper_step,
                                g_lo,
                                g_hi,
                            )
                            self._right_gripping = right_gripper > 0
                else:
                    (
                        left_delta,
                        right_delta,
                        left_gripper,
                        right_gripper,
                        _,
                    ) = self._compute_keyboard_delta(key_snapshot)
                    has_left_input = np.linalg.norm(left_delta) > 1e-8
                    has_right_input = np.linalg.norm(right_delta) > 1e-8

                    if has_left_input or has_right_input:
                        ee_poses = self._robot_interface.get_ee_poses()
                        if ee_poses is not None:
                            left_target = (
                                np.asarray(ee_poses["left"][:6] + left_delta, dtype=np.float32)
                                if has_left_input
                                else None
                            )
                            right_target = (
                                np.asarray(ee_poses["right"][:6] + right_delta, dtype=np.float32)
                                if has_right_input
                                else None
                            )
                            ik_result = self._robot_interface.control_dual_arm_ik(
                                step_size=step_size,
                                left_target_xyzrpy=left_target,
                                right_target_xyzrpy=right_target,
                            )
                            if ik_result and "smoothed_positions" in ik_result:
                                sp = ik_result["smoothed_positions"]
                                offset = 0
                                if "left_joint_positions" in ik_result:
                                    self._hold_arm_positions[:7] = np.array(
                                        sp[offset : offset + 7], dtype=np.float32
                                    )
                                    offset += 7
                                if "right_joint_positions" in ik_result:
                                    self._hold_arm_positions[7:14] = np.array(
                                        sp[offset : offset + 7], dtype=np.float32
                                    )

                    gripper_step = 0.002
                    g_open = self._robot_interface.gripper_open_width
                    g_close = self._robot_interface.gripper_close_width
                    g_lo, g_hi = min(g_open, g_close), max(g_open, g_close)
                    if abs(left_gripper) > 0.01:
                        # 【Bug修复】：同理，键盘控制也必须用减号
                        self._hold_finger_positions[:2] = np.clip(
                            self._hold_finger_positions[:2] - left_gripper * gripper_step,
                            g_lo,
                            g_hi,
                        )
                        self._left_gripping = left_gripper > 0
                    if abs(right_gripper) > 0.01:
                        self._hold_finger_positions[2:4] = np.clip(
                            self._hold_finger_positions[2:4] - right_gripper * gripper_step,
                            g_lo,
                            g_hi,
                        )
                        self._right_gripping = right_gripper > 0

        # 统一下发保持目标
        self._robot_interface.set_arm_joint_positions(
            target_arm_positions=self._hold_arm_positions.tolist(),
            task_num=self.config.task_cfg.get("task_number", 1),
        )

        close_tau = self._robot_interface.gripper_close_tau
        efforts = [0.0, 0.0, 0.0, 0.0]
        finger_pos = self._hold_finger_positions.copy()

        # 这里的 _left_gripping 此时就对应了正确的意思：当按下 'l' 键 (1.0 > 0) 时触发力控
        if self._left_gripping:
            efforts[0] = close_tau
            efforts[1] = close_tau
            finger_pos[0] = float("nan")
            finger_pos[1] = float("nan")
        if self._right_gripping:
            efforts[2] = close_tau
            efforts[3] = close_tau
            finger_pos[2] = float("nan")
            finger_pos[3] = float("nan")

        self._robot_interface.set_finger_positions(
            target_fingers=finger_pos.tolist(),
            task_num=self.config.task_cfg.get("task_number", 1),
        )
        self._robot_interface.apply_finger_efforts(efforts)

    def _score_input_record_callback(self, step_size: float) -> None:
        if self._scene_builder is None:
            return
        get_transforms = getattr(self._scene_builder, "get_target_object_transforms", None)
        if callable(get_transforms):
            get_transforms(step_size)

    def _foam_sync_callback(self, _step_size: float) -> None:
        if self._scene_builder is None:
            return
        sync_foam = getattr(self._scene_builder, "sync_foam_to_box", None)
        if callable(sync_foam):
            sync_foam()

    def _camera_images_callback(self, _step: float) -> None:
        if not self.is_connected:
            return
        camera_data: dict[str, np.ndarray] = {}
        for cam_name in self.CAMERA_NAMES:
            try:
                cam_res = self._robot_interface.get_camera_rgbd(cam_name)
                rgb = cam_res.get("rgb") if cam_res else None
                if rgb is not None:
                    camera_data[cam_name] = rgb
            except Exception:
                continue
        if camera_data:
            with self._callback_lock:
                self._latest_camera_rgb = {
                    name: frame.copy() for name, frame in camera_data.items()
                }

    # ---- 特征属性 ----
    @property
    def camera_features(self) -> dict[str, Any]:
        features = {}
        for cam_name in self.CAMERA_NAMES:
            features[f"observation.images.{cam_name}"] = {
                "shape": (3, self.config.camera_height, self.config.camera_width),
                "names": ["channels", "height", "width"],
                "dtype": "video",
            }
        return features

    @property
    def env_state_dim(self) -> int:
        task_cfg = getattr(self.config, "task_cfg", {})
        if not task_cfg:
            return 0
        task = task_cfg.get("task_number", 0)
        if task == 1:
            n = task_cfg.get("part", {}).get("num_parts", 2) * 2
        elif task == 2:
            n = task_cfg.get("part", {}).get("num_parts", 5) * 2
        elif task == 3:
            num_boxes = len(task_cfg.get("box", {}).get("box_position", []))
            num_parts = task_cfg.get("part", {}).get("num_parts", 3)
            n = num_boxes * num_parts
        elif task == 4:
            n = 0
        else:
            n = 0
        return n * 7

    @property
    def motor_features(self) -> dict[str, Any]:
        feats = {
            "observation.state": {
                "dtype": "float32",
                "shape": (self.STATE_DIM,),
                "names": self.STATE_NAMES,
            },
            "action": {
                "dtype": "float32",
                "shape": (self.ACTION_DIM,),
                "names": self.ACTION_NAMES,
            },
        }
        dim = self.env_state_dim
        if dim > 0:
            num_obj = dim // 7
            names = []
            for i in range(num_obj):
                names.extend([f"obj{i}_{c}" for c in ["x", "y", "z", "qx", "qy", "qz", "qw"]])
            feats["observation.environment_state"] = {
                "dtype": "float32",
                "shape": (dim,),
                "names": names,
            }
        return feats

    @property
    def features(self) -> dict[str, Any]:
        return {**self.motor_features, **self.camera_features}

    # Bug修复#3: 只保留一个正确的 observation_features（使用 self.STATE_NAMES 而非不存在的 self.config.state_names）
    @property
    def observation_features(self) -> dict:
        feats = {
            "observation.state": {
                "dtype": "float32",
                "shape": (self.STATE_DIM,),
                "names": self.STATE_NAMES,
            }
        }
        for cam_name in self.CAMERA_NAMES:
            feats[f"observation.images.{cam_name}"] = {
                "shape": (3, self.config.camera_height, self.config.camera_width),
                "names": ["channels", "height", "width"],
                "dtype": "video",
                "info": {
                    "video.fps": 30.0,
                    "video.height": self.config.camera_height,
                    "video.width": self.config.camera_width,
                    "video.channels": 3,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
        dim = self.env_state_dim
        if dim > 0:
            num_obj = dim // 7
            names = []
            for i in range(num_obj):
                names.extend([f"obj{i}_{c}" for c in ["x", "y", "z", "qx", "qy", "qz", "qw"]])
            feats["observation.environment_state"] = {
                "dtype": "float32",
                "shape": (dim,),
                "names": names,
            }
        return feats

    @property
    def action_features(self) -> dict:
        return {
            "action": {
                "dtype": "float32",
                "shape": (self.ACTION_DIM,),
                "names": self.ACTION_NAMES,
            }
        }

    @property
    def has_camera(self) -> bool:
        return True

    @property
    def num_cameras(self) -> int:
        return len(self.CAMERA_NAMES)

    @property
    def available_arms(self) -> list[str]:
        return ["left", "right"]

    @property
    def leader_arms(self) -> dict:
        return {}

    @property
    def follower_arms(self) -> dict:
        return {}

    @property
    def is_calibrated(self) -> bool:
        return True

    def configure(self):
        pass

    def calibrate(self, arm_name: str | None = None) -> None:
        pass

    # ---- connect / disconnect / reset ----
    def connect(self) -> None:
        if self.is_connected:
            logger.info("已经连接")
            return

        if not self.config.task_cfg_path:
            raise ValueError("必须提供 task_cfg_path 以加载场景")

        # 步骤 1: 创建 SimulationApp
        from isaacsim import SimulationApp
        self._kit = SimulationApp({
            "width": self.config.sim_width,
            "height": self.config.sim_height,
            "headless": self.config.headless,
        })

        # 步骤 2: 加载场景 USD
        from isaacsim.core.api import World
        import omni.usd as omni_usd

        scene_path = os.path.join(
            self.config.task_cfg.get("root_path", ""),
            self.config.task_cfg.get("scene_usd", ""),
        )
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"场景 USD 文件不存在: {scene_path}")
        omni_usd.get_context().open_stage(scene_path)

        # 步骤 3: 创建 World
        self._world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.config.physics_dt,
            rendering_dt=self.config.rendering_dt,
        )
        self._world.initialize_physics()

        # 步骤 4: SceneBuilder 构建场景
        try:
            # 获取当前工作空间的根目录 (假设是 /workspace/lerobot_0.5.1)
            workspace_root = Path(__file__).parent.parent.parent.parent.parent.parent
            
            # 这里的路径改为你存放新代码的 Ubtech_sim
            ubtech_root = workspace_root / "Ubtech_sim"
            
            if str(ubtech_root) not in os.sys.path:
                # 将 Ubtech_sim 加入路径，且放在最前面防止被旧路径截胡
                os.sys.path.insert(0, str(ubtech_root))
            
            # 修改导入语句，从你的 source 目录导入
            # 注意：根据你的目录结构，如果 Ubtech_sim/source 是个 package，则：
            # from lerobot.ecbg.source.SceneBuilder import SceneBuilder
            # from lerobot.ecbg.source.DataLogger import DataLogger
            from Ubtech_sim.source.SceneBuilder import SceneBuilder
            from Ubtech_sim.source.DataLogger import DataLogger
            
            print(f"[WalkerS2sim] 成功从 {ubtech_root} 加载 SceneBuilder")

            data_logger = DataLogger(
                enabled=False, csv_path="", camera_enabled=False, camera_hdf5_path=""
            )
            self._scene_builder = SceneBuilder(self.config.task_cfg, data_logger=data_logger)
            self._scene_builder.build_all()
            self._scene_builder.build_robot()

            self._world.play()
        except ImportError as e:
            logger.error(f"无法导入 SceneBuilder: {e}")
            raise
        except Exception as e:
            logger.error(f"场景构建失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

        # 步骤 5: 创建机器人接口
        self._robot_interface = IsaacSimRobotInterface(
            prim_path=self.config.prim_path,
            name=self.config.robot_name,
            world=self._world,
            urdf_path=self.config.urdf_path,
        )
        self._robot_interface.initialize()
        self._setup_joints()

        # 快照初始关节状态
        states = self._robot_interface.get_joint_states()
        if states:
            self._hold_arm_positions = np.array(states["arm_positions"], dtype=np.float32)
            self._hold_finger_positions = np.array(states["finger_positions"], dtype=np.float32)

        self._is_connected = True

        # Bug修复#5: connect() 末尾注册物理/渲染回调，缺少此行机器人永远不会动
        self._register_world_callbacks()

        logger.info(f"连接成功，控制 {len(self._arm_joint_indices)} 个手臂关节")

    def disconnect(self) -> None:
        if not self.is_connected:
            return

        self._stop_keyboard_listener()

        # Bug修复#4: 修复双重 None 逻辑错误（先 stop 再置 None）
        if self._ros2_teleop is not None:
            self._ros2_teleop.stop()
            self._ros2_teleop = None

        self._unregister_world_callbacks()

        if self._robot_interface:
            self._robot_interface.cleanup()
            self._robot_interface = None
        self._data_logger = None

        if self._world:
            try:
                self._world.stop()
            except Exception:
                pass
            self._world = None

        if self._kit:
            try:
                self._kit.close()
            except Exception:
                pass
            self._kit = None

        # Bug修复#1: disconnect() 必须将 _is_connected 设为 False
        self._is_connected = False
        logger.info("已断开连接")

    def reset(self) -> None:
        self._unregister_world_callbacks()

        self._scene_builder.reset()

        task_num = self.config.task_cfg.get("task_number", 0)
        if task_num in (1, 3):
            self._world.reset()
            self._scene_builder.scatter_after_reset()

        self._robot_interface.reset()

        settle_steps = int(1.0 / self.config.physics_dt)
        for _ in range(settle_steps):
            self._robot_interface._world.step(render=False)

        self._send_action_step_idx = 0

        with self._callback_lock:
            self._pending_absolute_action = None
            self._latest_camera_rgb = {}
        self._last_keyboard_frame_id = -1
        self._keyboard_cmd_queue.clear()
        self._current_frame_keys = None
        self._pressed_keys = {}
        self._left_gripping = False
        self._right_gripping = False

        states = self._robot_interface.get_joint_states()
        if states:
            self._hold_arm_positions = np.array(states["arm_positions"], dtype=np.float32)
            self._hold_finger_positions = np.array(states["finger_positions"], dtype=np.float32)
        else:
            self._hold_arm_positions = None
            self._hold_finger_positions = None

        self._register_world_callbacks()
        logger.info("[IsaacSim] 环境已重置")

    # ---- send_action / get_observation ----
    def send_action(self, action: torch.Tensor | None = None) -> torch.Tensor:
        t0 = time.perf_counter()
        try:
            if not self.is_connected:
                raise RobotDeviceNotConnectedError("未连接")

            if action is not None:
                action_np = (
                    action.detach().cpu().numpy()
                    if isinstance(action, torch.Tensor)
                    else np.asarray(action)
                )
                action_np = action_np.reshape(-1)
                if action_np.shape[0] != self.ACTION_DIM:
                    raise ValueError(
                        f"推理动作维度错误: 期望 {self.ACTION_DIM}，得到 {action_np.shape[0]}"
                    )
                with self._callback_lock:
                    self._pending_absolute_action = action_np.copy()
                return torch.tensor(action_np, dtype=torch.float32)
            else:
                joints_states = self._robot_interface.get_joint_states()
                if joints_states and "arm_positions" in joints_states:
                    arm_pos = torch.tensor(joints_states["arm_positions"], dtype=torch.float32)
                    gripper_pos = torch.tensor(
                        joints_states.get("finger_positions", [0.0] * 4), dtype=torch.float32
                    )
                    lg = torch.tensor(1.0 if self._left_gripping else -1.0, dtype=torch.float32)
                    rg = torch.tensor(1.0 if self._right_gripping else -1.0, dtype=torch.float32)
                    return torch.cat([arm_pos, gripper_pos, lg.unsqueeze(0), rg.unsqueeze(0)])
                else:
                    return torch.zeros(self.ACTION_DIM, dtype=torch.float32)
        finally:
            self.record_timing("send_action", time.perf_counter() - t0)

    def get_observation(self) -> dict[str, torch.Tensor]:
        t0 = time.perf_counter()
        try:
            if not self.is_connected:
                raise RobotDeviceNotConnectedError("未连接")

            obs = {}

            try:
                joints_states = self._robot_interface.get_joint_states()
                if joints_states and "arm_positions" in joints_states:
                    arm_pos = torch.tensor(joints_states["arm_positions"], dtype=torch.float32)
                    gripper_pos = torch.tensor(
                        joints_states.get("finger_positions", [0.0] * 4), dtype=torch.float32
                    )
                    lg = torch.tensor(1.0 if self._left_gripping else -1.0, dtype=torch.float32)
                    rg = torch.tensor(1.0 if self._right_gripping else -1.0, dtype=torch.float32)
                    state = torch.cat([arm_pos, gripper_pos, lg.unsqueeze(0), rg.unsqueeze(0)])
                    obs["observation.state"] = state
                    self._current_positions = arm_pos.clone()
                else:
                    raise RuntimeError("无法获取关节状态")
            except Exception as e:
                logger.warning(f"[observation] 获取关节状态失败: {e}")
                obs["observation.state"] = torch.zeros(self.STATE_DIM, dtype=torch.float32)
                self._current_positions = torch.zeros(14, dtype=torch.float32)

            for cam_name in self.CAMERA_NAMES:
                img_key = f"observation.images.{cam_name}"
                try:
                    with self._callback_lock:
                        rgb_cached = self._latest_camera_rgb.get(cam_name)
                    rgb = rgb_cached.copy() if rgb_cached is not None else None
                    if rgb is None:
                        cam_res = self._robot_interface.get_camera_rgbd(cam_name)
                        rgb = cam_res.get("rgb") if cam_res else None
                    if rgb is None:
                        obs[img_key] = torch.zeros(
                            (3, self.config.camera_height, self.config.camera_width)
                        )
                        continue
                    if rgb.shape[-1] == 4:
                        rgb = rgb[..., :3]
                    obs[img_key] = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                except Exception as e:
                    logger.warning(f"相机 {cam_name} 获取失败: {e}")
                    obs[img_key] = torch.zeros(
                        (3, self.config.camera_height, self.config.camera_width)
                    )

            dim = self.env_state_dim
            if dim > 0:
                try:
                    if self._scene_builder is not None:
                        env_flat = self._scene_builder.get_object_poses_flat()
                        if env_flat.shape[0] == dim:
                            obs["observation.environment_state"] = torch.from_numpy(env_flat)
                        else:
                            obs["observation.environment_state"] = torch.zeros(
                                dim, dtype=torch.float32
                            )
                    else:
                        obs["observation.environment_state"] = torch.zeros(
                            dim, dtype=torch.float32
                        )
                except Exception as e:
                    logger.warning(f"[observation] 获取物体位姿失败: {e}")
                    obs["observation.environment_state"] = torch.zeros(dim, dtype=torch.float32)

            return obs
        finally:
            self.record_timing("get_observation", time.perf_counter() - t0)

    def teleop_step(self, record_data: bool = False) -> tuple[dict, dict] | None:
        action_tensor = self.send_action(None)
        render_now = (self._send_action_step_idx % self._render_every_n) == 0
        self.step(render=render_now)
        obs = self.get_observation()
        if not record_data:
            return None
        return (obs, {"action": action_tensor})

    # ---- 关节辅助 ----
    def _setup_joints(self) -> None:
        self._arm_joint_indices = self._robot_interface.arm_joint_indices
        if len(self._arm_joint_indices) != 14:
            raise RuntimeError(
                f"关节索引数量错误: 期望14，得到 {len(self._arm_joint_indices)}"
            )
        states = self._robot_interface.get_joint_states()
        if states is None or "arm_positions" not in states:
            raise RuntimeError("无法获取关节状态或缺少 arm_positions")
        logger.info(f"14个手臂关节已设置，初始位置: {states['arm_positions']}")

    # ---- 键盘控制 ----
    def _init_keyboard_listener(self) -> None:
        """初始化键盘监听（优先 evdev，回退 pynput）。"""
        self._pressed_keys = {
            "speed_up": False, "speed_down": False, "quit": False,
            "switch_control_arm": False, "toggle_bimanual_mode": False,
            "toggle_gripper_mode": False,
            "x_up": False, "x_down": False, "y_up": False, "y_down": False,
            "z_up": False, "z_down": False,
            "rx_up": False, "rx_down": False, "ry_up": False, "ry_down": False,
            "rz_up": False, "rz_down": False,
            "gripper_open": False, "gripper_close": False,
        }

        # Bug修复#6: EVDEV_AVAILABLE 和 EvdevKeyboardListener 现在在模块顶部正确定义
        if EVDEV_AVAILABLE:
            try:
                listener = EvdevKeyboardListener(
                    on_press=self._on_key_press,
                    on_release=self._on_key_release,
                )
                listener.start()
                self._keyboard_listener = listener
                logger.info("键盘监听已启动 (evdev)")
                return
            except Exception as e:
                logger.warning(f"evdev 启动失败: {e}，尝试 pynput")

        if PYNPUT_AVAILABLE:
            try:
                self._keyboard_listener = pynput_keyboard.Listener(
                    on_press=self._on_key_press,
                    on_release=self._on_key_release,
                )
                self._keyboard_listener.start()
                logger.info("键盘监听已启动 (pynput)")
                return
            except Exception as e:
                logger.warning(f"pynput 启动失败: {e}")

        logger.warning("evdev 和 pynput 均不可用，键盘控制功能被禁用")

    def _stop_keyboard_listener(self) -> None:
        if self._keyboard_listener:
            self._keyboard_listener.stop()
            self._keyboard_listener = None

    def _teleop_key(self, action_name: str, default: str) -> str:
        keymap = getattr(self.config, "teleop_keymap", {})
        return str(keymap.get(action_name, default)).lower()

    def _resolve_key_action(self, char: str | None) -> str | None:
        if char is None:
            return None
        key_to_action = {
            self._teleop_key("x_up", "1"): "x_up",
            self._teleop_key("x_down", "3"): "x_down",
            self._teleop_key("y_up", "4"): "y_up",
            self._teleop_key("y_down", "6"): "y_down",
            self._teleop_key("z_up", "7"): "z_up",
            self._teleop_key("z_down", "9"): "z_down",
            self._teleop_key("rx_up", "y"): "rx_up",
            self._teleop_key("rx_down", "u"): "rx_down",
            self._teleop_key("ry_up", "v"): "ry_up",
            self._teleop_key("ry_down", "b"): "ry_down",
            self._teleop_key("rz_up", "n"): "rz_up",
            self._teleop_key("rz_down", "m"): "rz_down",
            self._teleop_key("gripper_open", "k"): "gripper_open",
            self._teleop_key("gripper_close", "l"): "gripper_close",
        }
        return key_to_action.get(char)

    def _apply_active_arm_delta(
        self, active_delta: np.ndarray, active_gripper: float
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        left_delta = np.zeros(6, dtype=np.float32)
        right_delta = np.zeros(6, dtype=np.float32)
        left_gripper = 0.0
        right_gripper = 0.0

        if self.current_control_arm == "left":
            left_delta = active_delta.copy()
            left_gripper = active_gripper
            if self.bimanual_control_enabled:
                right_delta = active_delta * self.BIMANUAL_MIRROR_SIGNS
                right_gripper = active_gripper
        else:
            right_delta = active_delta.copy()
            right_gripper = active_gripper
            if self.bimanual_control_enabled:
                left_delta = active_delta * self.BIMANUAL_MIRROR_SIGNS
                left_gripper = active_gripper

        return left_delta, right_delta, left_gripper, right_gripper

    def _enqueue_keyboard_snapshot(self) -> None:
        self._keyboard_cmd_queue.append(dict(self._pressed_keys))

    def _on_key_press(self, key) -> bool | None:
        try:
            char = getattr(key, "char", None)
            char = char.lower() if char is not None else None

            if char == self._teleop_key("quit", "q"):
                self._pressed_keys["quit"] = True
                return False

            if char == self._teleop_key("switch_control_arm", "o"):
                self.current_control_arm = (
                    "right" if self.current_control_arm == "left" else "left"
                )
                self._enqueue_keyboard_snapshot()
                return None

            if char == self._teleop_key("toggle_bimanual_mode", "0"):
                if not self._pressed_keys.get("toggle_bimanual_mode", False):
                    self._pressed_keys["toggle_bimanual_mode"] = True
                    self.bimanual_control_enabled = not self.bimanual_control_enabled
                return None

            if char == self._teleop_key("toggle_gripper_mode", "2"):
                if not self._pressed_keys.get("toggle_gripper_mode", False):
                    self._pressed_keys["toggle_gripper_mode"] = True
                    self.gripper_control_mode = (
                        "effort" if self.gripper_control_mode == "position" else "position"
                    )
                return None

            if char in (self._teleop_key("speed_up", "+"), "="):
                self._speed_index = min(
                    self._speed_index + 1, len(self.config.speed_levels) - 1
                )
                return None
            elif char == self._teleop_key("speed_down", "-"):
                self._speed_index = max(self._speed_index - 1, 0)
                return None

            action_name = self._resolve_key_action(char)
            if action_name is not None:
                self._pressed_keys[action_name] = True
                self._enqueue_keyboard_snapshot()

        except Exception as e:
            print(f"[keyboard] 处理错误: {e}", flush=True)
        return None

    def _on_key_release(self, key) -> None:
        try:
            char = getattr(key, "char", None)
            char = char.lower() if char is not None else None

            # Bug修复#8: 按压时用 "o" 作为 switch_control_arm，释放时也应该用 "o"，原来错误地用了 "j"
            if char == self._teleop_key("switch_control_arm", "o"):
                self._pressed_keys["switch_control_arm"] = False
                return

            if char == self._teleop_key("toggle_bimanual_mode", "0"):
                self._pressed_keys["toggle_bimanual_mode"] = False
                return

            if char == self._teleop_key("toggle_gripper_mode", "2"):
                self._pressed_keys["toggle_gripper_mode"] = False
                return

            action_name = self._resolve_key_action(char)
            if action_name is not None:
                self._pressed_keys[action_name] = False
                self._enqueue_keyboard_snapshot()

        except Exception:
            pass

    def _compute_keyboard_delta(
        self, key_snapshot: dict[str, bool] | None = None
    ) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
        keys = key_snapshot if key_snapshot is not None else self._pressed_keys
        step = self.config.speed_levels[self._speed_index]

        active_delta = np.zeros(6, dtype=np.float32)
        active_gripper = 0.0

        for index, axis in enumerate(self.MOTION_AXES):
            if keys.get(f"{axis}_up"):
                active_delta[index] += step
            if keys.get(f"{axis}_down"):
                active_delta[index] -= step

        # k: gripper_open -> -1.0 (非力矩模式)
        # l: gripper_close -> 1.0 (触发后续的力矩模式)
        if keys.get("gripper_open"):
            active_gripper = -1.0
        elif keys.get("gripper_close"):
            active_gripper = 1.0

        left_delta, right_delta, left_gripper, right_gripper = self._apply_active_arm_delta(
            active_delta, active_gripper
        )
        action = np.concatenate([left_delta, right_delta])
        return left_delta, right_delta, left_gripper, right_gripper, action

    # ---- 工具方法 ----
    def record_timing(self, metric_name: str, duration_s: float) -> None:
        metric = self._timing_metrics.setdefault(metric_name, TimingMetric())
        metric.update(duration_s)

    def get_timing_stats(self) -> dict[str, dict[str, float | int]]:
        return {n: m.as_dict() for n, m in self._timing_metrics.items() if m.count > 0}

    def print_timing_stats(self) -> None:
        stats = self.get_timing_stats()
        if not stats:
            return
        lines = ["=== 耗时统计 ==="]
        for name in ["send_action", "get_observation", "dt_s"]:
            if name not in stats:
                continue
            s = stats[name]
            lines.append(
                f"{name}: count={s['count']}, avg={s['avg_s']*1000:.2f}ms, "
                f"min={s['min_s']*1000:.2f}ms, max={s['max_s']*1000:.2f}ms"
            )
        print("\n".join(lines))

    def print_logs(self) -> None:
        if not self.is_connected:
            print("未连接")
            return
        print(f"速度等级: {self._speed_index} ({self.config.speed_levels[self._speed_index]:.3f})")
        print(f"当前控制臂: {self.current_control_arm}")
        print(f"控制模式: {'双臂' if self.bimanual_control_enabled else '单臂'}")
        if self._current_positions is not None:
            print(f"当前位置: {self._current_positions.tolist()}")

    def get_box_joints(self):
        return self._scene_builder.box_articulation.get_joint_positions()

    def reset_body(self):
        if self._robot_interface._articulation is None:
            raise RuntimeError("Articulation 未初始化")
        body_indices = [
            idx
            for idx in range(self._robot_interface._articulation.num_dof)
            if idx
            not in (
                self._robot_interface.arm_joint_indices
                + self._robot_interface.finger_joint_indices
            )
        ]
        body_initial_positions = [
            self._robot_interface.initial_joint_positions[idx] for idx in body_indices
        ]
        self._robot_interface._articulation.set_joint_positions(
            torch.tensor(body_initial_positions, dtype=torch.float32),
            joint_indices=torch.tensor(body_indices, dtype=torch.int32),
        )
        self._robot_interface._articulation.set_joint_velocities(
            torch.zeros(len(body_indices), dtype=torch.float32),
            joint_indices=torch.tensor(body_indices, dtype=torch.int32),
        )

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass