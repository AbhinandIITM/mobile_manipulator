"""
mobile_manipulator_controller.py
================================
Integrated Whole-Body Controller for Jackal J100 + UR5e + Robotiq 2F-85.

Features:
1. Arm DLS Inverse Kinematics with Nullspace Optimization (DOF slice 10:16).
2. Skid-steer drive mapping with slip compensation and slew rate limiting.
3. Multi-stage mission state machine:
   TRANSIT -> ALIGN_DOCK -> DEPLOY_ARM -> DESCEND -> GRASP -> LIFT -> TRANSPORT -> RELEASE
"""

import numpy as np
import mujoco


def orientation_error(R_curr: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    R_err = R_des @ R_curr.T
    return 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1]
    ])


class MobileManipulatorArmIK:
    """
    Damped Least Squares Inverse Kinematics for the UR5e arm mounted on the mobile base.
    """
    def __init__(
        self,
        model: mujoco.MjModel,
        site_name: str = "pinch",
        arm_dof_start: int = 10,
        num_arm_joints: int = 6,
        kp_pos: float = 12.0,
        kp_ori: float = 10.0,
        damping: float = 0.05,
    ):
        self.model = model
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Site '{site_name}' not found.")

        self.arm_dof_start = arm_dof_start
        self.num_arm_joints = num_arm_joints
        self.dof_slice = slice(arm_dof_start, arm_dof_start + num_arm_joints)

        self.kp_pos = kp_pos
        self.kp_ori = kp_ori
        self.damping = damping

        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

        self.stowed_q = np.array([0.0, -2.2, 2.4, -1.77, -1.57, 0.0])
        self.home_q   = np.array([-1.5708, -1.2, 1.4, -1.77, -1.57, 0.0])
        self.q_target = self.stowed_q.copy()

    def reset(self, q_init: np.ndarray):
        self.q_target = np.asarray(q_init, dtype=float).copy()

    def compute_ik(
        self,
        data: mujoco.MjData,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        dt: float = 0.002,
    ) -> np.ndarray:
        curr_pos = data.site_xpos[self.site_id].copy()
        curr_rot = data.site_xmat[self.site_id].reshape(3, 3).copy()

        pos_err = target_pos - curr_pos
        ori_err = orientation_error(curr_rot, target_rot)

        v_task = self.kp_pos * pos_err
        w_task = self.kp_ori * ori_err
        xdot_des = np.concatenate([v_task, w_task])

        mujoco.mj_jacSite(self.model, data, self.jacp, self.jacr, self.site_id)
        J_arm = np.vstack([self.jacp[:, self.dof_slice], self.jacr[:, self.dof_slice]])

        lambda_sq = (self.damping ** 2) * np.eye(6)
        J_dls = J_arm.T @ np.linalg.inv(J_arm @ J_arm.T + lambda_sq)

        # Nullspace posture pull
        q_null = 0.5 * (self.home_q - self.q_target)
        N = np.eye(self.num_arm_joints) - J_dls @ J_arm
        qdot = J_dls @ xdot_des + N @ q_null

        # Clamp joint velocities
        max_qdot = 2.5
        qdot = np.clip(qdot, -max_qdot, max_qdot)

        self.q_target += qdot * dt
        return self.q_target.copy()


class SkidSteerBaseDrive:
    """
    Slip-compensated 4-wheel skid-steer velocity generator with slew rate limiting.
    """
    def __init__(
        self,
        wheel_radius: float = 0.098,
        track_width: float = 0.368,
        accel_max: float = 1.0,
        w_dot_max: float = 2.5,
    ):
        self.r = wheel_radius
        self.L = track_width
        self.accel_max = accel_max
        self.w_dot_max = w_dot_max
        self.prev_v = 0.0
        self.prev_w = 0.0

    def compute_wheel_speeds(self, v_cmd: float, w_cmd: float, dt: float = 0.01) -> np.ndarray:
        # Slew rate limit
        dv = np.clip(v_cmd - self.prev_v, -self.accel_max * dt, self.accel_max * dt)
        dw = np.clip(w_cmd - self.prev_w, -self.w_dot_max * dt, self.w_dot_max * dt)
        v = self.prev_v + dv
        w = self.prev_w + dw
        self.prev_v = v
        self.prev_w = w

        slip = 0.30 if abs(v) > 0.05 else 0.34
        w_eff = w / slip

        v_L = v - w_eff * (self.L / 2.0)
        v_R = v + w_eff * (self.L / 2.0)

        # Actuators: FL(0), FR(1), RR(2), RL(3)
        # Note: FR & RR have gear="-1" in j100 XML
        w_FL = v_L / self.r
        w_FR = v_R / self.r
        w_RR = v_R / self.r
        w_RL = v_L / self.r

        return np.array([w_FL, w_FR, w_RR, w_RL])
