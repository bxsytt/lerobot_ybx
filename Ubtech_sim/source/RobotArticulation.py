from typing import Optional, List
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.types import ArticulationActions
import torch
import numpy as np
from isaacsim.sensors.camera import Camera
from .DualArmIK import DualArmIK

class RobotArticulation:
    """具有articulation属性的机器人类，提供控制接口"""
    
    def __init__(
        self,
        prim_path: str,
        name: str = "robot",
        usd_path: Optional[str] = None,
    ):
        """
        初始化机器人
        """
        self.prim_path = prim_path
        self.name = name
        self.usd_path = usd_path
        self._articulation = None
        self.time = 0.0
        self.target_joints = [
            "L_shoulder_pitch_joint",
            "L_shoulder_roll_joint", 
            "L_shoulder_yaw_joint",
            "L_elbow_pitch_joint",
            "L_elbow_yaw_joint",
            "L_wrist_pitch_joint",
            "L_wrist_roll_joint",
            "L_finger1_joint",
            "L_finger2_joint",
        ]
        self.sixforce_joint_names = [
            "L_sixforce_joint",   # 左手
            "R_sixforce_joint",  # 右手
        ]
        self.inital_joint_positions = None
        self.cameras = {}
        self.ik_solver: Optional[DualArmIK] = None
        self._left_arm_isaac_indices = None
        self._right_arm_isaac_indices = None
        self._waist_isaac_indices = None
        self._waist_init_positions = None
        self._ik_warn_counter = 0
        self._smooth_alpha = 0.3  # EMA smoothing: 0=no change, 1=instant
        self._last_arm_positions = {}  # side -> np.array of last sent positions
    def initialize(self):
        """初始化articulation用于仿真"""
        self._articulation = Articulation(
            prim_paths_expr=self.prim_path,
            name=self.name,
        )
        self._articulation.initialize()
        s2_joint_names = self._articulation.dof_names
        self.inital_joint_positions = [0.0] * len(s2_joint_names)
        
        # Set specific values only for target joints that exist in s2_joint_names
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
        for i, joint_name in enumerate(s2_joint_names):
            if joint_name in self._joint_value_map:
                self.inital_joint_positions[i] = self._joint_value_map[joint_name]
        s2_joint_indices = [self._articulation.get_dof_index(name) for name in s2_joint_names]
        self._articulation.set_joint_positions(torch.tensor(self.inital_joint_positions), joint_indices=torch.tensor(s2_joint_indices))
    
        self._setup_cameras()
    def _setup_cameras(self):
        """Setup cameras with fixed prim paths"""
        # Head cameras
        self.cameras['head_left'] = Camera(
            prim_path="/Root/Ref_Xform/Ref/head_pitch_link/head_stereo_left/head_stereo_left_Camera_01"
        )
        self.cameras['head_left'].initialize()
        self.cameras['head_left'].add_distance_to_image_plane_to_frame()
        self.cameras['head_right'] = Camera(
            prim_path="/Root/Ref_Xform/Ref/head_pitch_link/head_stereo_right/head_stereo_right_Camera_01"
        )
        self.cameras['head_right'].initialize()
        self.cameras['head_right'].add_distance_to_image_plane_to_frame()
        # Wrist cameras
        self.cameras['wrist_left'] = Camera(
            prim_path="/Root/Ref_Xform/Ref/L_camera_link/L_camera_link/L_wrist_camera/L_wrist_Camera"
        )
        self.cameras['wrist_left'].initialize()
        self.cameras['wrist_left'].add_distance_to_image_plane_to_frame()
        self.cameras['wrist_right'] = Camera(
            prim_path="/Root/Ref_Xform/Ref/R_camera_link/R_camera_link/R_wrist_camera/R_wrist_Camera"
        )
        self.cameras['wrist_right'].initialize()
        self.cameras['wrist_right'].add_distance_to_image_plane_to_frame()
    def cleanup(self):
        """清理资源"""
        if self._articulation is not None:
            self._articulation.cleanup()

    def get_joint_states(self):
        if self._articulation is None:
            return None
            
        joint_names = self._articulation.dof_names
        joint_positions = self._articulation.get_joint_positions().tolist()
        joint_velocities = self._articulation.get_joint_velocities().tolist()
        joint_efforts = self._articulation.get_measured_joint_efforts().tolist()
        
        return {
            'names': joint_names,
            'positions': joint_positions,
            'velocities': joint_velocities,
            'efforts': joint_efforts
        }

    def get_sixforce(self):
        wrench_data_list = []
        sensor_joint_forces = self._articulation.get_measured_joint_forces()[0]

        for joint_name in self.sixforce_joint_names:
            joint_index = self._articulation.get_joint_index(joint_name)

            sixforce_data = sensor_joint_forces[joint_index + 1]
            wrench_data = {
                'frame_id': joint_name.replace("_joint", "_frame"),
                'force': sixforce_data[:3].tolist(),
                'torque': sixforce_data[3:].tolist()
            }
            wrench_data_list.append(wrench_data)
        return wrench_data_list
    
    def get_camera_rgb(self, camera_name):
        if camera_name not in self.cameras:
            return None
        try:
            return self.cameras[camera_name].get_rgb()
        except Exception as e:
            print(f"Failed to get RGB from camera {camera_name}: {e}")
            return None
    
    def get_camera_depth(self, camera_name):
        if camera_name not in self.cameras:
            return None
        try:
            return self.cameras[camera_name].get_depth()
        except Exception as e:
            print(f"Failed to get depth from camera {camera_name}: {e}")
            return None
    
    def get_camera_rgbd(self, camera_name):
        rgb = self.get_camera_rgb(camera_name)
        depth = self.get_camera_depth(camera_name)
        return {
            'rgb': rgb,
            'depth': depth,
            'camera_name': camera_name,
        }

    def get_head_left_camera_rgbd(self):
        """获取头部左相机的RGB-D数据"""
        return self.get_camera_rgbd('head_left')
    
    def get_head_right_camera_rgbd(self):
        """获取头部右相机的RGB-D数据"""
        return self.get_camera_rgbd('head_right')
    
    def get_wrist_left_camera_rgbd(self):
        """获取腕部左相机的RGB-D数据"""
        return self.get_camera_rgbd('wrist_left')
    
    def get_wrist_right_camera_rgbd(self):
        """获取腕部右相机的RGB-D数据"""
        return self.get_camera_rgbd('wrist_right')            

    def get_cameras_images(self, step):
        """获取所有相机的RGB-D数据"""
        camera_data = {}
        for camera_name in self.cameras.keys():
            camera_data[camera_name] = self.get_camera_rgbd(camera_name)
        return camera_data
    # ------------------------------------------------------------------
    # 双臂 IK 控制
    # ------------------------------------------------------------------

    def initialize_ik(self, urdf_path: str):
        """初始化 Pinocchio IK 求解器并建立关节索引映射"""
        self.ik_solver = DualArmIK(urdf_path)

        dof_names = self._articulation.dof_names
        current_positions = self._articulation.get_joint_positions().tolist()
        if current_positions and isinstance(current_positions[0], list):
            current_positions = current_positions[0]
        # 建立左/右臂关节名到 Isaac Sim dof index 的映射
        self._left_arm_isaac_indices = []
        for jname in DualArmIK.LEFT_ARM_JOINTS:
            if jname in dof_names:
                self._left_arm_isaac_indices.append(self._articulation.get_dof_index(jname))

        self._right_arm_isaac_indices = []
        for jname in DualArmIK.RIGHT_ARM_JOINTS:
            if jname in dof_names:
                self._right_arm_isaac_indices.append(self._articulation.get_dof_index(jname))

        # 腰部腿部关节：运行时锁定到初始位置，防止晃动
        WAIST_LEGS_JOINTS = ["waist_yaw_joint", "waist_pitch_joint",
                             "L_hip_pitch_joint", "L_hip_roll_joint", "L_hip_yaw_joint",
                             "R_hip_pitch_joint", "R_hip_roll_joint", "R_hip_yaw_joint",
                             "L_knee_pitch_joint", "R_knee_pitch_joint",
                             "L_ankle_pitch_joint", "L_ankle_roll_joint",
                             "R_ankle_pitch_joint", "R_ankle_roll_joint"]
        self._waist_legs_isaac_indices = []
        self._waist_legs_init_positions = []
        for jname in WAIST_LEGS_JOINTS:
            idx = self._articulation.get_dof_index(jname)
            self._waist_legs_isaac_indices.append(idx)
            # 锁定到当前姿态，避免每步被强拉回 0 导致抖动
            self._waist_legs_init_positions.append(float(0.0))

        # 先将腰部腿部关节设为目标值，再同步到 pinocchio
        if self._waist_legs_isaac_indices:
            self._articulation.set_joint_positions(
                torch.tensor(self._waist_legs_init_positions, dtype=torch.float32),
                joint_indices=torch.tensor(self._waist_legs_isaac_indices, dtype=torch.int32),
            )

        # 用初始关节值同步 pinocchio（首次同步，更新全部关节）
        joints = self.get_joint_states()
        if joints is not None:
            self.ik_solver.sync_joint_positions(
                joints['names'], joints['positions'][0]
            )

        # 保存为初始 q — 后续 sync 只更新手臂关节，其余锁定到此值
        self.ik_solver.save_initial_q()

        # 设置中性臂构型，用于 IK 零空间优化
        left_neutral = [self._joint_value_map.get(j, 0.0) for j in DualArmIK.LEFT_ARM_JOINTS]
        right_neutral = [self._joint_value_map.get(j, 0.0) for j in DualArmIK.RIGHT_ARM_JOINTS]
        self.ik_solver.set_neutral_config(left_neutral, right_neutral)

        print(f"[DualArmIK] 初始化完成  左臂 {len(self._left_arm_isaac_indices)} DOF, "
              f"右臂 {len(self._right_arm_isaac_indices)} DOF, "
              f"腰部腿部锁定 {len(self._waist_legs_isaac_indices)} DOF")

    def get_ee_poses(self):
        """获取双臂末端执行器当前 xyzrpy（先同步 Isaac Sim 关节状态到 pinocchio）"""
        joints = self.get_joint_states()
        if joints is None or self.ik_solver is None:
            return None
        self.ik_solver.sync_joint_positions(joints['names'], joints['positions'][0])
        return self.ik_solver.get_both_ee_poses()

    def control_dual_arm_ik(
        self,
        step_size: float,
        left_target_xyzrpy=None,
        right_target_xyzrpy=None,
        **ik_kwargs,
    ):
        """
        双臂 IK 控制：分别给左右臂末端目标 [x,y,z,roll,pitch,yaw]，同时下发。

        Args:
            step_size: 物理步长（秒）
            left_target_xyzrpy:  左臂目标位姿，None 则保持当前位置
            right_target_xyzrpy: 右臂目标位姿，None 则保持当前位置
            **ik_kwargs: 传给 IK 求解器的参数（max_iter, pos_tol, rot_tol, damping, dt）
        """
        self.time += step_size

        if self.ik_solver is None:
            print("[DualArmIK] IK 求解器未初始化，请先调用 initialize_ik()")
            return

        # 1. 同步当前关节状态到 pinocchio
        joints = self.get_joint_states()
        if joints is None:
            return

        isaac_names = joints['names']
        isaac_positions = joints['positions'][0]  # shape: (N,)

        # 2. 求解双臂 IK
        ik_result = self.ik_solver.solve_dual_arm(
            left_target_xyzrpy=left_target_xyzrpy,
            right_target_xyzrpy=right_target_xyzrpy,
            isaac_joint_names=isaac_names,
            isaac_joint_positions=isaac_positions,
            **ik_kwargs,
        )

        # 3. 合并目标关节角，平滑后下发
        all_indices = []
        all_positions = []

        warn_msg = []
        if 'left_joint_positions' in ik_result:
            smoothed = self._smooth_joints('left', ik_result['left_joint_positions'])
            all_indices.extend(self._left_arm_isaac_indices)
            all_positions.extend(smoothed.tolist())
            if not ik_result['left_success']:
                warn_msg.append("左臂")

        if 'right_joint_positions' in ik_result:
            smoothed = self._smooth_joints('right', ik_result['right_joint_positions'])
            all_indices.extend(self._right_arm_isaac_indices)
            all_positions.extend(smoothed.tolist())
            if not ik_result['right_success']:
                warn_msg.append("右臂")

        # 每 200 步（约 1 秒）只打印一次警告，避免刷屏
        if warn_msg:
            self._ik_warn_counter += 1
            if self._ik_warn_counter >= 200:
                diag = ""
                if hasattr(self.ik_solver, '_last_fail_info'):
                    info = self.ik_solver._last_fail_info
                    diag = (f" | pos_err={info['pos_err']:.4f}m"
                            f" rot_err={info['rot_err']:.4f}rad"
                            f" rot_tol={info['effective_rot_tol']:.4f}")
                print(f"[DualArmIK] 警告: {', '.join(warn_msg)} IK 未收敛{diag}")
                self._ik_warn_counter = 0
        else:
            self._ik_warn_counter = 0

        # 锁定腰部关节到初始位置（不用 apply_action，避免与手臂控制耦合）
        if self._waist_isaac_indices:
            self._articulation.set_joint_positions(
                torch.tensor(self._waist_init_positions, dtype=torch.float32),
                joint_indices=torch.tensor(self._waist_isaac_indices, dtype=torch.int32),
            )

        if len(all_indices) > 0:
            self._articulation.apply_action(
                ArticulationActions(
                    joint_positions=torch.tensor([all_positions], dtype=torch.float32),
                    joint_indices=torch.tensor(all_indices, dtype=torch.int32),
                )
            )

    def _smooth_joints(self, side: str, ik_positions: np.ndarray) -> np.ndarray:
        """EMA smoothing of IK joint output to reduce jitter.

        Args:
            side: 'left' or 'right'
            ik_positions: raw IK output joint angles

        Returns:
            Smoothed joint angles
        """
        ik_positions = np.asarray(ik_positions, dtype=float)
        if side not in self._last_arm_positions:
            self._last_arm_positions[side] = ik_positions.copy()
            return ik_positions

        prev = self._last_arm_positions[side]
        alpha = self._smooth_alpha
        smoothed = prev + alpha * (ik_positions - prev)
        self._last_arm_positions[side] = smoothed.copy()
        return smoothed

    def control_example(self, step_size):
        """
        Simple control function that sets target positions for the seven left arm joints.
        This function is called every physics step.
        """
        self.time += step_size
        joints_states = self.get_joint_states()
        sixforces = self.get_sixforce()

        target_positions = []
        