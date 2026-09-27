import numpy as np
import mujoco


def orientation_error(R_curr: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """
    Computes small-angle orientation error vector e_o such that:
      R_des ≈ exp([e_o]x) * R_curr
    Uses the skew-symmetric matrix extraction: e_o = 0.5 * (n x n_d + s x s_d + a x a_d)
    where R = [n s a].
    """
    R_err = R_des @ R_curr.T
    e_o = 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1]
    ])
    return e_o


class CartesianIKController:
    """
    Closed-Loop Inverse Kinematics (CLIK) controller using Damped Least Squares (DLS)
    with Nullspace Posture Projection and Integrated Reference State.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        site_name: str = "attachment_site",
        home_qpos: np.ndarray = None,
        kp_pos: float = 20.0,
        kp_ori: float = 15.0,
        k_null: float = 2.0,
        damping: float = 0.02,
        max_qdot: float = 3.14,  # rad/s
    ):
        self.model = model
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in MuJoCo model.")

        if home_qpos is None:
            self.home_qpos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
        else:
            self.home_qpos = np.asarray(home_qpos, dtype=float).copy()

        # Integrated reference configuration for position actuators
        self.q_target = self.home_qpos.copy()

        self.kp_pos = kp_pos
        self.kp_ori = kp_ori
        self.k_null = k_null
        self.damping = damping
        self.max_qdot = max_qdot

        # Preallocate Jacobian buffers (3xnv)
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def reset(self, init_qpos: np.ndarray):
        """Resets the internal reference configuration."""
        self.q_target = np.asarray(init_qpos, dtype=float).copy()

    def compute_control(
        self,
        data: mujoco.MjData,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_rot: np.ndarray,
        target_angvel: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, dict]:
        """
        Computes the target joint position command q_cmd for the position actuators.

        Returns
        -------
        q_cmd : np.ndarray
            Joint position commands for actuators (length nu = 6)
        diag_info : dict
            Diagnostic metrics (pos_err, ori_err, norm_pos_err, etc.)
        """
        # 1. Get current end-effector pose
        curr_pos = data.site_xpos[self.site_id].copy()
        curr_rot = data.site_xmat[self.site_id].reshape(3, 3).copy()

        # 2. Tracking errors
        pos_err = target_pos - curr_pos
        ori_err = orientation_error(curr_rot, target_rot)

        # 3. Desired task-space twist (feedforward + proportional feedback)
        v_task = target_vel + self.kp_pos * pos_err
        w_task = target_angvel + self.kp_ori * ori_err
        xdot_des = np.concatenate([v_task, w_task])  # 6D twist

        # 4. Compute Site Jacobian (6 x nv)
        mujoco.mj_jacSite(self.model, data, self.jacp, self.jacr, self.site_id)
        
        # Only use the arm joints for IK (first nv_arm joints)
        nv_arm = len(self.home_qpos)
        J = np.vstack([self.jacp[:, :nv_arm], self.jacr[:, :nv_arm]])  # (6, nv_arm)

        # 5. Damped Least Squares (DLS) Pseudo-inverse: J_dls = J^T (J J^T + lambda^2 I)^-1
        lambda_sq = (self.damping ** 2) * np.eye(6)
        J_dls = J.T @ np.linalg.inv(J @ J.T + lambda_sq)

        # 6. Nullspace Posture Control (pulls reference toward nominal home configuration)
        q_null = self.k_null * (self.home_qpos - self.q_target)
        N = np.eye(nv_arm) - J_dls @ J
        qdot_null = N @ q_null

        # 7. Total Joint Velocity Command
        qdot_cmd = J_dls @ xdot_des + qdot_null

        # Clamp joint velocities to safety limits
        qdot_cmd = np.clip(qdot_cmd, -self.max_qdot, self.max_qdot)

        # 8. Integrate reference target configuration
        # For position-controlled actuators in MuJoCo (gainprm * (ctrl - qpos) - biasprm * qvel),
        # ctrl must track the integrated desired reference angle q_target, NOT (current_qpos + qdot*dt).
        self.q_target += qdot_cmd * dt

        diag_info = {
            "curr_pos": curr_pos,
            "target_pos": target_pos,
            "pos_err": pos_err,
            "pos_err_norm": np.linalg.norm(pos_err),
            "ori_err_norm": np.linalg.norm(ori_err),
            "qdot_cmd": qdot_cmd,
            "q_cmd": self.q_target.copy(),
        }

        return self.q_target.copy(), diag_info
