from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from dataclasses import dataclass, field
from lerobot.robots.config import RobotConfig


PHYSICS_DT = 1.0 / 60.0
RENDERING_DT = 1.0 / 60.0
CAMERA_FPS = 30  # Bug修复: 原来写 fps=RENDERING_DT (≈0.0167)，fps是帧率整数，不是周期浮点数
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
SIM_WIDTH = 1280
SIM_HEIGHT = 720


@RobotConfig.register_subclass("walker_s2_sim")
@dataclass
class WalkerS2Config(RobotConfig):
    # Bug修复: 继承 RobotConfig 而不是只加装饰器（装饰器注册需要父类是 RobotConfig）
    type: str = "walker_s2_sim"
    id: str = "walker_s2_default"
    calibration_dir: str | None = None

    use_async_state: bool = False
    task_cfg_path: str = "/workspace/lerobot_0.5.1/src/lerobot/ecbg/config/task4.yaml"
    urdf_path: str = "/workspace/lerobot_0.5.1/src/lerobot/ecbg/assets/s2.urdf"
    task_cfg: dict = field(default_factory=dict)
    prim_path: str = "/Root/Ref_Xform/Ref"
    robot_name: str = "walkerS2"
    sim_width: int = SIM_WIDTH
    sim_height: int = SIM_HEIGHT
    headless: bool = False
    physics_dt: float = PHYSICS_DT
    rendering_dt: float = RENDERING_DT
    camera_width: int = CAMERA_WIDTH
    camera_height: int = CAMERA_HEIGHT

    speed_levels: list[float] = field(default_factory=lambda: [0.015, 0.035, 0.05])
    default_speed_index: int = 1

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "head_left": OpenCVCameraConfig(
                index_or_path=0,
                fps=CAMERA_FPS,      # Bug修复: 原来是 fps=RENDERING_DT ≈ 0.0167，改为正确的帧率30
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
            ),
            "head_right": OpenCVCameraConfig(
                index_or_path=1,
                fps=CAMERA_FPS,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
            ),
            "wrist_left": OpenCVCameraConfig(
                index_or_path=2,
                fps=CAMERA_FPS,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
            ),
            "wrist_right": OpenCVCameraConfig(
                index_or_path=3,
                fps=CAMERA_FPS,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
            ),
        }
    )