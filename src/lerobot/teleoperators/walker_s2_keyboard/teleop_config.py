from dataclasses import dataclass, field
from lerobot.teleoperators.teleoperator import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("walker_s2_keyboard")
@dataclass
class WalkerS2KeyboardTeleopConfig(TeleoperatorConfig):
    speed_levels: list[float] = field(default_factory=lambda: [0.015, 0.035, 0.05])
    default_speed_index: int = 1
    initial_control_arm: str = "left"

    # 键位映射: 字符 → 动作名
    # 动作名与 WalkerS2sim._resolve_key_action 中的映射保持一致
    keymap: dict[str, str] = field(
        default_factory=lambda: {
            "1": "x_up",
            "3": "x_down",
            "4": "y_up",
            "6": "y_down",
            "7": "z_up",
            "9": "z_down",
            "y": "rx_up",
            "u": "rx_down",
            "v": "ry_up",
            "b": "ry_down",
            "n": "rz_up",
            "m": "rz_down",
            "k": "gripper_open",
            "l": "gripper_close",
            "g": "toggle_arm",   # 瞬时切换控制臂，不进入持久状态
            "q": "quit",
            "0": "toggle_bimanual",
        }
    )