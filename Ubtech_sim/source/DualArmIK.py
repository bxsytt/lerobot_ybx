import pinocchio as pin
import numpy as np
from typing import Optional, Tuple, Dict, List


class DualArmIK:
    """基于Pinocchio的双臂逆运动学求解器
    
    左右臂分别求解IK，结果合并后同时下发到关节。
    目标位姿以 [x, y, z, roll, pitch, yaw] 形式给出，在机器人基座坐标系下。
    """

    LEFT_ARM_JOINTS = [
        "L_shoulder_pitch_joint",
        "L_shoulder_roll_joint",
        "L_shoulder_yaw_joint",
        "L_elbow_roll_joint",
        "L_elbow_yaw_joint",
        "L_wrist_pitch_joint",
        "L_wrist_roll_joint",
    ]

    RIGHT_ARM_JOINTS = [
        "R_shoulder_pitch_joint",
        "R_shoulder_roll_joint",
        "R_shoulder_yaw_joint",
        "R_elbow_roll_joint",
        "R_elbow_yaw_joint",
        "R_wrist_pitch_joint",
        "R_wrist_roll_joint",
    ]

    LEFT_EE_FRAME = "L_sixforce_link"
    RIGHT_EE_FRAME = "R_sixforce_link"

    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # 末端执行器 frame ID
        self.left_ee_id = self.model.getFrameId(self.LEFT_EE_FRAME)
        self.right_ee_id = self.model.getFrameId(self.RIGHT_EE_FRAME)

        # 左臂关节在 pinocchio 中的 q-index 和 v-index
        self.left_arm_q_indices = []
        self.left_arm_v_indices = []
        for name in self.LEFT_ARM_JOINTS:
            jid = self.model.getJointId(name)
            self.left_arm_q_indices.append(self.model.joints[jid].idx_q)
            self.left_arm_v_indices.append(self.model.joints[jid].idx_v)

        # 右臂关节在 pinocchio 中的 q-index 和 v-index
        self.right_arm_q_indices = []
        self.right_arm_v_indices = []
        for name in self.RIGHT_ARM_JOINTS:
            jid = self.model.getJointId(name)
            self.right_arm_q_indices.append(self.model.joints[jid].idx_q)
            self.right_arm_v_indices.append(self.model.joints[jid].idx_v)

        # 关节名 → pinocchio q-index 映射（用于从 Isaac Sim 同步关节状态）
        # 注意：不遍历 model.names（部分 pinocchio 版本 C++ binding 不可用），
        # 而是在首次 sync 时按需查找。
        self.joint_name_to_q_idx: Dict[str, int] = {}
        self._name_mapping_built = False

        # 内部 q 向量（全身）
        self.q = pin.neutral(self.model)

        # 中性臂构型 (用于零空间优化), 由外部 set_neutral_config 设置
        self.q_neutral_left: Optional[np.ndarray] = None
        self.q_neutral_right: Optional[np.ndarray] = None

        # 初始 q (锁定非手臂关节), 由外部 save_initial_q() 设置
        self.q_initial: Optional[np.ndarray] = None
        self._arm_joint_names: set = set(self.LEFT_ARM_JOINTS + self.RIGHT_ARM_JOINTS)

        # 连续 IK 失败计数（用于触发 warm-start 复位）
        self._left_fail_count: int = 0
        self._right_fail_count: int = 0
        self._fail_reset_threshold: int = 30  # 连续失败超过此值则复位

    # ------------------------------------------------------------------
    # 中性构型
    # ------------------------------------------------------------------

    def set_neutral_config(self, left_angles: List[float], right_angles: List[float]):
        """设置左右臂中性关节角 (7-DOF), 用于零空间优化."""
        self.q_neutral_left = np.asarray(left_angles, dtype=float)
        self.q_neutral_right = np.asarray(right_angles, dtype=float)
        print(f"[DualArmIK] 中性构型已设置  left={self.q_neutral_left}  right={self.q_neutral_right}")

    def save_initial_q(self):
        """保存当前 q 为初始参考, 后续 sync 时非手臂关节将锁定到此值."""
        self.q_initial = self.q.copy()
        print(f"[DualArmIK] 初始 q 已保存 (非手臂关节将锁定到此值)")

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------

    @staticmethod
    def xyzrpy_to_se3(xyzrpy) -> pin.SE3:
        """将 [x, y, z, roll, pitch, yaw] 转换为 pinocchio SE3"""
        pos = np.asarray(xyzrpy[:3], dtype=float)
        r, p, y = float(xyzrpy[3]), float(xyzrpy[4]), float(xyzrpy[5])
        rot = pin.rpy.rpyToMatrix(r, p, y)
        return pin.SE3(rot, pos)

    @staticmethod
    def se3_to_xyzrpy(se3: pin.SE3) -> np.ndarray:
        """将 pinocchio SE3 转换为 [x, y, z, roll, pitch, yaw]"""
        pos = se3.translation
        rpy = pin.rpy.matrixToRpy(se3.rotation)
        return np.concatenate([pos, rpy])

    # ------------------------------------------------------------------
    # 与 Isaac Sim 同步
    # ------------------------------------------------------------------

    def _build_name_mapping(self, isaac_joint_names: List[str]):
        """首次调用时，按 Isaac Sim 关节名逐个查找 pinocchio joint ID，构建映射"""
        for name in isaac_joint_names:
            if name not in self.joint_name_to_q_idx:
                if self.model.existJointName(name):
                    jid = self.model.getJointId(name)
                    self.joint_name_to_q_idx[name] = self.model.joints[jid].idx_q
        self._name_mapping_built = True

    def sync_joint_positions(self, isaac_joint_names: List[str],
                             isaac_joint_positions: List[float]):
        """从 Isaac Sim 的关节状态同步到 pinocchio 内部 q 向量.

        如果已调用 save_initial_q(), 则先将 q 重置为初始值,
        然后只同步手臂关节, 防止腿部等关节漂移导致发散.
        """
        if not self._name_mapping_built:
            self._build_name_mapping(isaac_joint_names)

        # 先重置为初始值 (锁定非手臂关节)
        if self.q_initial is not None:
            self.q = self.q_initial.copy()

        for name, pos in zip(isaac_joint_names, isaac_joint_positions):
            if name in self.joint_name_to_q_idx:
                # 有初始 q 时, 只同步手臂关节; 否则同步全部
                if self.q_initial is None or name in self._arm_joint_names:
                    self.q[self.joint_name_to_q_idx[name]] = pos

    # ------------------------------------------------------------------
    # 正运动学
    # ------------------------------------------------------------------

    def get_ee_pose(self, side: str) -> pin.SE3:
        """获取当前某臂末端执行器位姿（需先 sync_joint_positions）"""
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        if side == "left":
            return self.data.oMf[self.left_ee_id].copy()
        else:
            return self.data.oMf[self.right_ee_id].copy()

    def get_both_ee_poses(self) -> Dict[str, np.ndarray]:
        """获取双臂末端位姿 xyzrpy"""
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        return {
            "left": self.se3_to_xyzrpy(self.data.oMf[self.left_ee_id]),
            "right": self.se3_to_xyzrpy(self.data.oMf[self.right_ee_id]),
        }

    # ------------------------------------------------------------------
    # 单臂 IK 求解（阻尼最小二乘）
    # ------------------------------------------------------------------

    def _reset_arm_warmstart(self, side: str):
        """将指定臂的 warm-start 复位到 q_initial，用于逃出局部极小值。"""
        if self.q_initial is None:
            return
        indices = self.left_arm_q_indices if side == "left" else self.right_arm_q_indices
        for idx in indices:
            self.q[idx] = self.q_initial[idx]
        print(f"[DualArmIK] {side} 臂连续失败，warm-start 已复位到初始构型")

    def solve_ik_single_arm(
        self,
        target_se3: pin.SE3,
        side: str,
        max_iter: int = 150,
        pos_tol: float = 5e-3,
        rot_tol: float = 5e-3,
        damping: float = 1e-4,
        dt: float = 1.0,
        dq_max: float = 0.5,
        pos_weight: float = 1.0,
        rot_weight: float = 1.0,
        null_weight: float = 0.1,
    ) -> Tuple[np.ndarray, bool]:
        """
        单臂 IK 求解，使用加权阻尼最小二乘法（DLS）+ 零空间优化。
        
        Args:
            target_se3: 目标末端位姿 (SE3)
            side: "left" 或 "right"
            max_iter: 最大迭代次数
            pos_tol: 位置误差容差 (m)
            rot_tol: 姿态误差容差 (rad)
            damping: 阻尼系数
            dt: 步长
            pos_weight: 位置误差权重 (越大越优先)
            rot_weight: 姿态误差权重 (越小越放松姿态约束)
            null_weight: 零空间优化权重，将臂拉向中性构型 (0=禁用)

        Returns:
            (arm_joint_positions, success) — 7维关节角度数组，是否收敛
        """
        if side == "left":
            ee_id = self.left_ee_id
            q_indices = self.left_arm_q_indices
            v_indices = self.left_arm_v_indices
            q_neutral_arm = self.q_neutral_left
        else:
            ee_id = self.right_ee_id
            q_indices = self.right_arm_q_indices
            v_indices = self.right_arm_v_indices
            q_neutral_arm = self.q_neutral_right

        n_dof = len(v_indices)
        q = self.q.copy()
        W = np.diag([pos_weight]*3 + [rot_weight]*3)
        # 当姿态权重低时，自动放宽姿态容差
        effective_rot_tol = rot_tol / max(rot_weight, 0.01)
        use_null = (q_neutral_arm is not None and null_weight > 0)

        for _ in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            current_pose = self.data.oMf[ee_id]
            # 6D 误差向量（LOCAL 坐标系下的 log 映射）
            error = pin.log(current_pose.actInv(target_se3)).vector

            pos_err = np.linalg.norm(error[:3])
            rot_err = np.linalg.norm(error[3:])
            if pos_err < pos_tol and rot_err < effective_rot_tol:
                arm_q = np.array([q[i] for i in q_indices])
                for j, idx in enumerate(q_indices):
                    self.q[idx] = q[idx]
                return arm_q, True

            # 计算 LOCAL 坐标系下的 Jacobian，并施加权重
            J_full = pin.computeFrameJacobian(
                self.model, self.data, q, ee_id, pin.LOCAL
            )
            J_arm = J_full[:, v_indices]
            J_w = W @ J_arm
            e_w = W @ error

            # 加权阻尼最小二乘: dq = (Jw^T Jw + λI)^{-1} Jw^T ew
            JtJ = J_w.T @ J_w + damping * np.eye(n_dof)
            dq_task = np.linalg.solve(JtJ, J_w.T @ (dt * e_w))

            # 零空间优化：将臂推向中性构型，避免奇异/极限
            if use_null:
                J_w_pinv = np.linalg.solve(JtJ, J_w.T)      # n_dof × 6
                N = np.eye(n_dof) - J_w_pinv @ J_w           # n_dof × n_dof
                q_arm = np.array([q[idx] for idx in q_indices])
                dq_null = null_weight * N @ (q_neutral_arm - q_arm)
                dq_arm = dq_task + dq_null
            else:
                dq_arm = dq_task

            # 步长裁剪，防止振荡
            dq_norm = np.linalg.norm(dq_arm)
            if dq_norm > dq_max:
                dq_arm = dq_arm * (dq_max / dq_norm)

            # 仅更新手臂关节
            for j, idx in enumerate(q_indices):
                q[idx] += dq_arm[j]

            # 关节限位裁剪
            for j, idx in enumerate(q_indices):
                lo = self.model.lowerPositionLimit[idx]
                hi = self.model.upperPositionLimit[idx]
                q[idx] = np.clip(q[idx], lo, hi)

        # 未在 max_iter 内收敛，返回最后结果（仍然更新 self.q 用于 warm-start）
        arm_q = np.array([q[i] for i in q_indices])
        for j, idx in enumerate(q_indices):
            self.q[idx] = q[idx]
        self._last_fail_info = {
            'side': side, 'pos_err': pos_err, 'rot_err': rot_err,
            'effective_rot_tol': effective_rot_tol,
        }
        return arm_q, False

    def _update_fail_count(self, side: str, success: bool):
        """更新连续失败计数；失败超阈值则复位 warm-start。"""
        if side == "left":
            if success:
                self._left_fail_count = 0
            else:
                self._left_fail_count += 1
                if self._left_fail_count >= self._fail_reset_threshold:
                    self._reset_arm_warmstart("left")
                    self._left_fail_count = 0
        else:
            if success:
                self._right_fail_count = 0
            else:
                self._right_fail_count += 1
                if self._right_fail_count >= self._fail_reset_threshold:
                    self._reset_arm_warmstart("right")
                    self._right_fail_count = 0

    # ------------------------------------------------------------------
    # 双臂 IK 求解
    # ------------------------------------------------------------------

    def solve_dual_arm(
        self,
        left_target_xyzrpy=None,
        right_target_xyzrpy=None,
        isaac_joint_names: Optional[List[str]] = None,
        isaac_joint_positions: Optional[List[float]] = None,
        **ik_kwargs,
    ) -> Dict:
        """
        双臂 IK 求解入口。

        Args:
            left_target_xyzrpy:  左臂目标 [x, y, z, roll, pitch, yaw]，None 则跳过
            right_target_xyzrpy: 右臂目标 [x, y, z, roll, pitch, yaw]，None 则跳过
            isaac_joint_names:   Isaac Sim 当前关节名列表（用于同步）
            isaac_joint_positions: Isaac Sim 当前关节角列表
            **ik_kwargs: 传给 solve_ik_single_arm 的额外参数

        Returns:
            {
                'left_joint_names':  [...],   # 仅在 left_target 不为 None 时存在
                'left_joint_positions': np.array,
                'left_success': bool,
                'right_joint_names': [...],
                'right_joint_positions': np.array,
                'right_success': bool,
            }
        """
        if isaac_joint_names is not None and isaac_joint_positions is not None:
            self.sync_joint_positions(isaac_joint_names, isaac_joint_positions)

        result = {}

        if left_target_xyzrpy is not None:
            target = self.xyzrpy_to_se3(left_target_xyzrpy)
            arm_q, ok = self.solve_ik_single_arm(target, "left", **ik_kwargs)
            self._update_fail_count("left", ok)
            result['left_joint_names'] = list(self.LEFT_ARM_JOINTS)
            result['left_joint_positions'] = arm_q
            result['left_success'] = ok

        if right_target_xyzrpy is not None:
            target = self.xyzrpy_to_se3(right_target_xyzrpy)
            arm_q, ok = self.solve_ik_single_arm(target, "right", **ik_kwargs)
            self._update_fail_count("right", ok)
            result['right_joint_names'] = list(self.RIGHT_ARM_JOINTS)
            result['right_joint_positions'] = arm_q
            result['right_success'] = ok

        return result


    def reset_runtime_state(self):
            """重置IK运行时状态，用于机器人姿态回到初始位姿后重新开始求解。

            保留模型、关节映射和中性构型，仅清空warm-start参考与失败计数。

            Returns:
                None
            """
            self.q = pin.neutral(self.model)
            self.q_initial = None
            self._left_fail_count = 0
            self._right_fail_count = 0
            if hasattr(self, '_last_fail_info'):
                delattr(self, '_last_fail_info')