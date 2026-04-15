"""
teleop_and_record.py - lerobot v3 (0.5.x) 下的遥操作数据采集脚本

运行方式:
    PYTHONPATH=$PYTHONPATH:$(pwd)/src /isaac-sim/python.sh \
        src/lerobot/scripts/teleop_and_record.py \
        [--num_episodes 5] [--fps 30] [--task packing_box] \
        [--repo_id sjj/test] [--save_path datasets/task4/v1] \
        [--task_cfg src/lerobot/ecbg/config/task4.yaml]

主要修复（相对于原 teleop_and_record.py）:
  1. 修复键盘状态不同步问题: 通过 teleop.sync_to_robot(robot) 将 teleop 的
     _pressed_keys 同步给 robot，原代码 robot._pressed_keys 始终为空 {}
  2. 使用 argparse 支持命令行参数，不用每次改代码
  3. 整合 warmup / episode_time_s / reset_time_s 等与旧版一致的参数
  4. 正确处理 LeRobotDataset.create() 的 features 参数
  5. 退出信号检测: 同时支持 teleop quit 键和 robot quit 键
"""

import argparse
import logging
import time
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.walker_s2_sim.walkers2sim import WalkerS2sim
from lerobot.robots.walker_s2_sim.walkers2simConfig import WalkerS2Config
from lerobot.teleoperators.walker_s2_keyboard.teleop import WalkerS2KeyboardTeleop
from lerobot.teleoperators.walker_s2_keyboard.teleop_config import WalkerS2KeyboardTeleopConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# 修复 1: 强制关闭 PIL 等底层库的烦人调试日志
# ==========================================
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


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
        help="Isaac Sim 场景配置 YAML 路径",
    )
    parser.add_argument("--headless", action="store_true", help="无头模式运行仿真")
    parser.add_argument("--warmup_time_s", type=float, default=5.0, help="每个 episode 开始前等待时间(秒)")
    parser.add_argument("--episode_time_s", type=float, default=60.0, help="每个 episode 最长时间(秒)")
    parser.add_argument("--reset_time_s", type=float, default=5.0, help="重置后等待时间(秒)")
    parser.add_argument("--resume", action="store_true", help="从已有数据集继续采集")
    return parser.parse_args()


def wait_for_start_signal(teleop: WalkerS2KeyboardTeleop, robot: WalkerS2sim, wait_s: float):
    """在 warmup 期间持续渲染仿真，等待操作者准备好。"""
    logger.info(f"Warmup: 等待 {wait_s:.1f}s...")
    deadline = time.perf_counter() + wait_s
    while time.perf_counter() < deadline:
        # 持续渲染，让操作者能看到场景，同步状态但不记录
        teleop.sync_to_robot(robot)
        robot.step(render=True)
        time.sleep(0.01)
    # 清空 quit 信号，防止 warmup 期间按键漏入录制
    teleop._pressed_keys["quit"] = False
    robot._pressed_keys = {}


def record_episode(
    teleop: WalkerS2KeyboardTeleop,
    robot: WalkerS2sim,
    dataset: LeRobotDataset,
    task_description: str,
    fps: int,
    episode_time_s: float,
) -> bool:
    """
    录制一个 episode。

    Returns:
        True  - episode 正常录制完毕（时间到或手动结束）
        False - 用户请求退出（quit 键）
    """
    dt = 1.0 / fps
    max_steps = int(episode_time_s * fps)
    step_count = 0

    logger.info(f"开始录制（最多 {max_steps} 步 / {episode_time_s:.0f}s）")
    logger.info("按 q 结束当前 episode，再按 q 退出程序")

    while step_count < max_steps:
        t_start = time.perf_counter()

        # ===== 关键修复: 同步键盘状态到机器人 =====
        # 原代码中 robot._pressed_keys 始终为 {}，导致机器人不响应键盘输入
        # sync_to_robot() 会:
        #   1. 复制 teleop._pressed_keys → robot._pressed_keys
        #   2. 同步 current_control_arm 和 _speed_index
        #   3. 调用 robot._enqueue_keyboard_snapshot() 入队信号
        teleop.sync_to_robot(robot)
        # ===========================================

        # send_action(None) = 遥操作模式，从当前关节状态构造记录用 action
        # 实际的 IK 控制由物理回调 _robot_control_callback 完成
        record_action = robot.send_action(None)

        # 推进物理仿真（触发 _robot_control_callback 执行 IK 和关节控制）
        robot.step(render=True)

        # 获取观测
        obs = robot.get_observation()

        # 存入数据集
        frame_data = {
            "action": record_action,
            "task": task_description,
            **obs,
        }
        dataset.add_frame(frame_data)

        # 检查退出信号（teleop 或 robot 的 quit 键）
        if teleop._pressed_keys.get("quit", False) or robot._pressed_keys.get("quit", False):
            logger.info("检测到 quit 信号，结束当前 episode")
            teleop._pressed_keys["quit"] = False
            return False

        # 维持目标帧率
        elapsed = time.perf_counter() - t_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

        step_count += 1

    return True


def main():
    args = parse_args()

    # ---- 初始化机器人 ----
    robot_cfg = WalkerS2Config(
        task_cfg_path=args.task_cfg,
        headless=args.headless,
    )
    robot = WalkerS2sim(robot_cfg)

    # ---- 初始化遥操作器 ----
    teleop_cfg = WalkerS2KeyboardTeleopConfig(
        speed_levels=[0.015, 0.035, 0.05],
        default_speed_index=1,
    )
    teleop = WalkerS2KeyboardTeleop(teleop_cfg)

    try:
        logger.info("正在连接机器人...")
        robot.connect()
        logger.info("正在连接键盘遥操作器...")
        teleop.connect()

        # ---- 构建 dataset features ----
        # observation_features: observation.state + observation.images.* + observation.environment_state
        # action_features:      action (20维)
        dataset_features = {}
        dataset_features.update(robot.observation_features)
        dataset_features.update(robot.action_features)  # 使用机器人的 action_features 而非 teleop 的

        # ---- 创建数据集 ----
        save_path = Path(args.save_path)
        if args.resume and save_path.exists():
            logger.info(f"继续已有数据集: {save_path}")
            dataset = LeRobotDataset(
                repo_id=args.repo_id,
                root=save_path,
            )
        else:
            logger.info(f"创建新数据集: {save_path}")
            dataset = LeRobotDataset.create(
                repo_id=args.repo_id,
                root=save_path,
                fps=args.fps,
                robot_type="walker_s2_sim",
                features=dataset_features,
                use_videos=True,
            )

        logger.info(f"数据集特征: {list(dataset_features.keys())}")
        logger.info(f"开始采集 {args.num_episodes} 个 episodes，保存路径: {save_path}")

        episode_idx = 0
        while episode_idx < args.num_episodes:
            logger.info(f"\n========== Episode {episode_idx + 1}/{args.num_episodes} ==========")

            # Warmup: 等待操作者准备
            if args.warmup_time_s > 0:
                wait_for_start_signal(teleop, robot, args.warmup_time_s)

            # 录制 episode
            should_continue = record_episode(
                teleop=teleop,
                robot=robot,
                dataset=dataset,
                task_description=args.task,
                fps=args.fps,
                episode_time_s=args.episode_time_s,
            )

            # 保存 episode
            dataset.save_episode()
            episode_idx += 1
            logger.info(f"Episode {episode_idx} 已保存 (共 {len(dataset)} 帧)")

            if not should_continue:
                logger.info("收到退出信号，停止采集")
                break

            # Reset: 重置场景，等待稳定
            if episode_idx < args.num_episodes:
                logger.info("重置场景...")
                robot.reset()
                if args.reset_time_s > 0:
                    logger.info(f"等待 {args.reset_time_s:.1f}s 后开始下一 episode...")
                    time.sleep(args.reset_time_s)

    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，正在退出...")
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)
    finally:
        logger.info("正在关闭系统...")
        try:
            teleop.disconnect()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        logger.info(f"数据采集完成！数据集位置: {args.save_path}")


if __name__ == "__main__":
    main()