import torch
import numpy as np
from typing import Any
from lerobot.teleoperators.teleoperator import Teleoperator
from .teleop_config import WalkerS2KeyboardTeleopConfig

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


class WalkerS2KeyboardTeleop(Teleoperator):
    """
    键盘遥操作器 - 负责监听按键，向机器人提供控制信号。

    设计说明:
      - 本类只负责键盘状态管理（_pressed_keys, current_control_arm, _speed_index）
      - 实际的 IK 计算和关节控制由机器人类（WalkerS2sim）的物理回调完成
      - teleop_and_record.py 通过 sync_to_robot() 或手动赋值将状态同步给机器人
    """

    name = "walker_s2_keyboard"
    config_class = WalkerS2KeyboardTeleopConfig

    def __init__(self, config: WalkerS2KeyboardTeleopConfig | None = None):
        self.config = config if config is not None else WalkerS2KeyboardTeleopConfig()
        super().__init__(self.config)

        # 初始化按键状态字典（包含所有 keymap 中的动作 + 特殊动作）
        self._pressed_keys: dict[str, bool] = {
            action: False for action in self.config.keymap.values()
        }
        # toggle_arm 是瞬时切换，不加入持久状态；quit 需要单独加
        self._pressed_keys.pop("toggle_arm", None)
        # self._pressed_keys.pop("toggle_bimanual", None)
        self._pressed_keys["quit"] = False

        self.current_control_arm: str = self.config.initial_control_arm
        self._speed_index: int = self.config.default_speed_index
        self.listener = None
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def configure(self):
        pass

    def calibrate(self):
        return True

    def send_feedback(self, feedback: dict):
        pass

    @property
    def action_features(self) -> dict[str, dict]:
        state_names = [
            "L_shoulder_pitch_joint.pos", "L_shoulder_roll_joint.pos",
            "L_shoulder_yaw_joint.pos",
            "L_elbow_roll_joint.pos", "L_elbow_yaw_joint.pos",
            "L_wrist_pitch_joint.pos", "L_wrist_roll_joint.pos",
            "R_shoulder_pitch_joint.pos", "R_shoulder_roll_joint.pos",
            "R_shoulder_yaw_joint.pos",
            "R_elbow_roll_joint.pos", "R_elbow_yaw_joint.pos",
            "R_wrist_pitch_joint.pos", "R_wrist_roll_joint.pos",
            "L_finger1_joint.pos", "L_finger2_joint.pos",
            "R_finger1_joint.pos", "R_finger2_joint.pos",
            "left_gripper_control", "right_gripper_control",
        ]
        return {
            "action": {
                "dtype": "float32",
                "shape": (20,),
                "names": state_names,
            }
        }

    @property
    def feedback_features(self) -> dict[str, dict]:
        return {}

    def connect(self):
        if not PYNPUT_AVAILABLE:
            raise ImportError("未安装 pynput，请运行: pip install pynput")
        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self.listener.start()
        self._is_connected = True

    def disconnect(self):
        if self.listener:
            self.listener.stop()
            self.listener = None
        self._is_connected = False

    def _on_press(self, key):
        try:
            if hasattr(key, "char") and key.char in self.config.keymap:
                cmd = self.config.keymap[key.char]
                if cmd == "toggle_arm":
                    # 瞬时切换，不进入持久状态
                    arms = ["left", "right", "both"]
                    idx = arms.index(self.current_control_arm)
                    self.current_control_arm = arms[(idx + 1) % len(arms)]
                # # 处理切换逻辑
                # if cmd == "toggle_arm" and hasattr(self, "_robot"):
                #     self._robot.switch_control_arm()
                # elif cmd == "toggle_bimanual" and hasattr(self, "_robot"):
                #     # 直接调用机器人的切换方法
                #     self._robot.toggle_bimanual_mode()
                elif cmd == "quit":
                    self._pressed_keys["quit"] = True
                else:
                    self._pressed_keys[cmd] = True
            elif key == keyboard.Key.esc:
                self._pressed_keys["quit"] = True
        except Exception:
            pass
        

    def _on_release(self, key):
        try:
            if hasattr(key, "char") and key.char in self.config.keymap:
                cmd = self.config.keymap[key.char]
                # toggle_arm 是瞬时动作，没有持久状态，跳过
                if cmd != "toggle_arm" and cmd in self._pressed_keys:
                    self._pressed_keys[cmd] = False
                # if cmd in self._pressed_keys:
                #     self._pressed_keys[cmd] = True
        except Exception:
            pass

    def get_action(self) -> torch.Tensor:
        """
        在新框架下，返回 20 维 Action。
        """
        # 如果是录制模式，我们希望返回的是包含 IK 结果和夹爪位置的完整向量
        # 此时 teleop_and_record.py 会调用 robot.send_action(None)
        # 我们在这里返回一个占位符，但要确保维度是 20
        action = torch.zeros(20, dtype=torch.float32)
        
        # 记录当前的开关指令（用于有些框架可能需要的 delta 控制）
        if self._pressed_keys.get("gripper_open"):
            action[-2:] = -1.0 
        elif self._pressed_keys.get("gripper_close"):
            action[-2:] = 1.0
            
        return action
    
    
    def sync_to_robot(self, robot) -> None:
        """
        将键盘状态同步到机器人实例。

        在 teleop_and_record.py 的控制循环中调用此方法，
        确保机器人的 _pressed_keys 反映最新的键盘输入。

        Args:
            robot: WalkerS2sim 实例
        """
        robot.current_control_arm = self.current_control_arm
        robot._speed_index = self._speed_index
        # Bug修复(teleop_and_record): 关键同步 - 将 teleop 的按键状态复制给机器人
        # 原代码中 robot._pressed_keys 始终为空 {}，导致机器人从不响应键盘
        robot._pressed_keys = dict(self._pressed_keys)
        robot._enqueue_keyboard_snapshot()