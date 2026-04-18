"""
teleop_and_record.py - lerobot v3 (0.5.1) 下的遥操作数据采集脚本

核心修复:
  1. 将 dataset.consolidate() 替换为官方 0.5.1 标准的 dataset.finalize()，解决 Parquet 损坏问题。
  2. 引入官方的 VideoEncodingManager 上下文管理器，确保视频帧与动作数据安全对齐并落盘。
  3. 保留了终端实时打印 Action 和 Observation，便于检查数据。
  4. 交互逻辑（回车开始，左键重录，右键保存，Q键退出）保持不变。
"""

import argparse
import logging
import time
from pathlib import Path

import torch

# 引入 pynput 用于全局按键和鼠标监听
from pynput import keyboard, mouse

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.walker_s2_sim.walkers2sim import WalkerS2sim
from lerobot.robots.walker_s2_sim.walkers2simConfig import WalkerS2Config
from lerobot.teleoperators.walker_s2_keyboard.teleop import WalkerS2KeyboardTeleop
from lerobot.teleoperators.walker_s2_keyboard.teleop_config import WalkerS2KeyboardTeleopConfig

# 核心修复：引入官方的视频编码管理器
from lerobot.datasets.video_utils import VideoEncodingManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ==========================================
# 状态机：用于在后台线程和主线程间传递交互信号
# ==========================================
class UIState:
    start_recording = False
    end_episode_early = False
    redo_episode = False
    quit_program = False

    @classmethod
    def reset_episode_flags(cls):
        cls.start_recording = False
        cls.end_episode_early = False
        cls.redo_episode = False

ui_state = UIState()

def on_key_press(key):
    if key == keyboard.Key.enter:
        ui_state.start_recording = True
    elif hasattr(key, 'char') and key.char and key.char.lower() == 'q':
        ui_state.quit_program = True

    # 同步给 teleop 实例
    if hasattr(main, 'teleop'):
        main.teleop._on_press(key)

def on_key_release(key):
    if hasattr(main, 'teleop'):
        main.teleop._on_release(key)

def on_mouse_click(x, y, button, pressed):
    if pressed:
        if button == mouse.Button.left:
            ui_state.redo_episode = True
        elif button == mouse.Button.right:
            ui_state.end_episode_early = True

def parse_args():
    parser = argparse.ArgumentParser(description="WalkerS2 遥操作数据采集")
    parser.add_argument("--num_episodes", type=int, default=5, help="采集 episode 数量")
    parser.add_argument("--fps", type=int, default=30, help="采集帧率")
    parser.add_argument("--task", type=str, default="packing_box", help="任务描述文字")
    parser.add_argument("--repo_id", type=str, default="sjj/test", help="HuggingFace dataset repo_id")
    parser.add_argument("--save_path", type=str, default="datasets/task4/v1", help="本地保存路径")
    parser.add_argument(
        "--task_cfg",
        type=str,
        default="src/lerobot/ecbg/config/task4.yaml",
        # default="Ubtech_sim/config/Packing_Box.yaml",
        help="Isaac Sim 场景配置 YAML 路径",
    )
    parser.add_argument("--headless", action="store_true", help="无头模式运行仿真")
    parser.add_argument("--episode_time_s", type=float, default=60.0, help="每个 episode 最长时间(秒)")
    parser.add_argument("--resume", action="store_true", help="从已有数据集继续采集")
    return parser.parse_args()

def wait_for_start_signal(teleop: WalkerS2KeyboardTeleop, robot: WalkerS2sim):
    """等待操作者按下 Enter 键后才开始录制"""
    logger.info("*" * 50)
    logger.info("等待中: 请按下 [Enter] 键开始录制当前 Episode。")
    logger.info("随时可按 [Q] 键退出整个程序。")
    logger.info("*" * 50)
    
    UIState.reset_episode_flags()
    
    while not ui_state.start_recording and not ui_state.quit_program:
        teleop.sync_to_robot(robot)
        robot.step(render=True)
        time.sleep(0.01)

    if ui_state.quit_program:
        return False
    return True

def record_episode(
    teleop: WalkerS2KeyboardTeleop,
    robot: WalkerS2sim,
    dataset: LeRobotDataset,
    task_description: str,
    fps: int,
    episode_time_s: float,
) -> str:
    """
    录制一个 episode。
    Returns 状态字符串: "success", "redo", "quit"
    """
    dt = 1.0 / fps
    max_steps = int(episode_time_s * fps)
    step_count = 0

    UIState.reset_episode_flags()

    logger.info(f"开始录制（最多 {max_steps} 步 / {episode_time_s:.0f}s）")
    logger.info("操作提示: [鼠标右键]->保存并进入下一集 | [鼠标左键]->重新录制本集 | [Q]->退出程序")

    while step_count < max_steps:
        t_start = time.perf_counter()

        if ui_state.quit_program:
            logger.info("检测到退出信号 [Q]，准备退出程序。")
            return "quit"
        if ui_state.redo_episode:
            ui_state.redo_episode = False
            logger.warning("检测到 [鼠标左键]，丢弃当前数据，重新录制本集！")
            return "redo"
        if ui_state.end_episode_early:
            ui_state.end_episode_early = False
            logger.info("检测到 [鼠标右键]，提前结束并保存当前 Episode。")
            return "success"
        if teleop._pressed_keys.get("quit", False):
            return "quit"

        teleop.sync_to_robot(robot)
        
        # 1. 获取 Action
        record_action = robot.send_action(None)
        if isinstance(record_action, torch.Tensor):
            record_action = record_action.to(torch.float32)
        else:
            record_action = torch.tensor(record_action, dtype=torch.float32)

        # 2. 仿真步进
        robot.step(render=True)
        
        # 3. 获取 Observation
        obs = robot.get_observation()
        # 强制转换 obs 中的所有浮点数组
        processed_obs = {}
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                processed_obs[k] = v.to(dtype=torch.float32)
            else:
                processed_obs[k] = torch.tensor(v, dtype=torch.float32)
                
        # 每秒打印 1 次，防止刷屏，便于比对数据
        if step_count % fps == 0:
            logger.info(f"--- 录制中 | 第 {step_count} 帧 ---")
            action_list = [round(x, 3) for x in record_action.tolist()]
            logger.info(f"Action (20维): {action_list}")
            
            if "observation.state" in obs:
                obs_list = [round(x, 3) for x in obs["observation.state"].tolist()]
                logger.info(f"observation State (20维): {obs_list}")

        frame_data = {
            "action": record_action,
            "task": task_description,
            **obs,
        }
        dataset.add_frame(frame_data)

        # 维持目标帧率
        elapsed = time.perf_counter() - t_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

        step_count += 1

    logger.info("时间到，当前 Episode 录制完成。")
    return "success"

def main():
    args = parse_args()

    robot_cfg = WalkerS2Config(task_cfg_path=args.task_cfg, headless=args.headless)
    robot = WalkerS2sim(robot_cfg)
    teleop_cfg = WalkerS2KeyboardTeleopConfig(speed_levels=[0.015, 0.035, 0.05], default_speed_index=1)
    main.teleop = WalkerS2KeyboardTeleop(teleop_cfg)

    # 启动监听器
    k_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    m_listener = mouse.Listener(on_click=on_mouse_click)
    k_listener.start()
    m_listener.start()

    dataset = None 

    try:
        logger.info("正在连接机器人...")
        robot.connect()
        logger.info("正在连接键盘遥操作器...")
        main.teleop.connect()

        dataset_features = {}
        dataset_features.update(robot.observation_features)
        dataset_features.update(robot.action_features)

        save_path = Path(args.save_path)
        if args.resume and save_path.exists():
            dataset = LeRobotDataset(repo_id=args.repo_id, root=save_path)
            # 兼容恢复录制时的图片写入器
            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer(num_processes=0, num_threads=4 * len(robot.cameras))
        else:
            dataset = LeRobotDataset.create(
                repo_id=args.repo_id,
                root=save_path,
                fps=args.fps,
                robot_type="walker_s2_sim",
                features=dataset_features,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=4 * len(robot.cameras),
            )

        logger.info(f"开始采集 {args.num_episodes} 个 episodes，保存路径: {save_path}")

        episode_idx = 0
        
        # 核心修复：使用 VideoEncodingManager 包装循环，安全处理视频帧缓冲
        with VideoEncodingManager(dataset):
            while episode_idx < args.num_episodes:
                logger.info(f"\n========== Episode {episode_idx + 1}/{args.num_episodes} ==========")

                if not wait_for_start_signal(main.teleop, robot):
                    break 

                status = record_episode(
                    teleop=main.teleop,
                    robot=robot,
                    dataset=dataset,
                    task_description=args.task,
                    fps=args.fps,
                    episode_time_s=args.episode_time_s,
                )

                if status == "quit":
                    break
                elif status == "redo":
                    dataset.clear_episode_buffer()
                    logger.info("正在重置场景以备重新采集...")
                    robot.reset()
                    continue 
                elif status == "success":
                    dataset.save_episode()
                    episode_idx += 1
                    logger.info(f"Episode {episode_idx} 已成功保存到缓冲区 (累计 {len(dataset)} 帧)")
                    
                    if episode_idx < args.num_episodes:
                        logger.info("正在重置场景，准备下一集...")
                        robot.reset()

    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，正在退出...")
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)
    finally:
        logger.info("正在关闭系统并保存数据...")
        k_listener.stop()
        m_listener.stop()
        
        # 核心修复：使用 dataset.finalize() 写出 Parquet 的 metadata
        if dataset is not None:
            logger.info("正在执行 dataset.finalize() 闭合 Parquet 文件并写入元数据...")
            try:
                dataset.finalize()
                logger.info("数据集元数据合并完成！Parquet 文件安全！")
            except Exception as e:
                logger.error(f"关闭数据集时出错: {e}")

        try:
            main.teleop.disconnect()
            robot.disconnect()
        except Exception:
            pass
        logger.info(f"数据采集结束！数据集最终位置: {args.save_path}")

if __name__ == "__main__":
    main()