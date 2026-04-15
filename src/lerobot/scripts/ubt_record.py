from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.teleoperators.walker_s2_keyboard.teleop import WalkerS2KeyboardTeleop
from lerobot.teleoperators.walker_s2_keyboard.teleop_config import WalkerS2KeyboardTeleopConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.robots.walker_s2_sim.walkers2sim import WalkerS2sim
from lerobot.robots.walker_s2_sim.walkers2simConfig import WalkerS2Config
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop


current_file = Path(__file__).resolve()
DATASETS_ROOT = current_file.parent.parent.parent / "datasets"
DATASETS_ROOT_STR = str(DATASETS_ROOT)
# print("数据集根路径:", DATASETS_ROOT_STR)

NUM_EPISODES = 2
FPS = 30
EPISODE_TIME_SEC = 10000
RESET_TIME_SEC = 30
TASK_DESCRIPTION = "packing_box"
HF_REPO_ID = "sjj/lerobot-walker-s2-sim-recording-test"
SAVE_PATH = DATASETS_ROOT_STR + '/packing_box_v0'


def main():
    robot_config = WalkerS2Config()
    robot = WalkerS2sim(robot_config)

    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    # print("Action features:", action_features)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    # print("Observation features:", obs_features)
    dataset_features = {**action_features, **obs_features}
    # print("Combined dataset features:", dataset_features)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    keyboard_config = WalkerS2KeyboardTeleopConfig()
    keyboard = WalkerS2KeyboardTeleop(keyboard_config)


    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=dataset_features,
        root=SAVE_PATH,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    robot.connect()
    keyboard.connect()
    listener, events = init_keyboard_listener()
    # init_rerun(session_name="walkers2_record")

    try:
        if not robot.is_connected or not keyboard.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        log_flag = True
        while not events["start_record"]:
            robot._robot_interface._world.step(render=True) 
            if log_flag:
                log_say(text="调整好按Enter键开始录制...", play_sounds=False)
                log_flag = False

        print("Starting record loop...")
        episode_idx = 0
        while episode_idx < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

            # Main record loop
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop=keyboard,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=False,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

            # Reset the environment if not stopping or re-recording
            if not events["stop_recording"] and (
                episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                record_loop(
                    robot=robot ,
                    events=events,
                    fps=FPS,
                    teleop=keyboard,
                    control_time_s=RESET_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=False,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                )

            if events["rerecord_episode"]:
                log_say("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            # Save episode
            dataset.save_episode()
            episode_idx += 1

    finally:
        # Clean up
        log_say("Stop recording")
        robot.disconnect()
        keyboard.disconnect()
        listener.stop()

        dataset.finalize()
        dataset.push_to_hub()



if __name__ == "__main__":
    main()