"""
odometry.py - Dead-Reckoning Wheel Odometry and Extended Kalman Filter (EKF) State Estimator.

Fuses noisy wheel encoders and IMU gyroscope data to provide real-time state estimation
with gyro-bias correction.

State vector: x = [x, y, theta, b_g]^T
  - x, y   : 2D position in world frame (m)
  - theta  : Heading angle in world frame (rad)
  - b_g    : Gyroscope yaw-rate bias (rad/s)

Prediction step  : integrates encoder forward velocity + de-biased gyro yaw rate.
Update step      : fuses encoder-derived yaw rate against gyro to correct b_g drift.
"""

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wraps an angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class WheelOdometry:
    """
    Computes dead-reckoning odometry purely from wheel encoder velocities.

    Uses the effective skid-steer track width (1.15 m) rather than the
    physical track width (0.368 m) to account for lateral scrubbing.
    """
    def __init__(self, wheel_radius: float = 0.098, track_width: float = 1.15, initial_pose=None):
        self.r = wheel_radius
        self.L = track_width  # Effective skid-steer track width

        if initial_pose is None:
            self.x = 0.0
            self.y = 0.0
            self.theta = 0.0
        else:
            self.x, self.y, self.theta = initial_pose

        self.v = 0.0
        self.w = 0.0

    def update(self, encoder_vels: np.ndarray, dt: float):
        """
        Updates odometry using wheel velocities [FL, FR, RR, RL] in rad/s.

        Sign convention:
          FL, RL: positive when driving forward (joint axis +Z)
          FR, RR: read negative when forward (joint axis -Z), so negated here.
        """
        w_L = 0.5 * (encoder_vels[0] + encoder_vels[3])   # avg left-side rad/s
        w_R = -0.5 * (encoder_vels[1] + encoder_vels[2])  # avg right-side rad/s (sign corrected)

        v_L = w_L * self.r
        v_R = w_R * self.r

        v_lin = 0.5 * (v_R + v_L)
        w_ang = (v_R - v_L) / self.L

        self.v = v_lin
        self.w = w_ang

        # Midpoint Euler integration (reduces linearization error vs forward Euler)
        delta_theta = w_ang * dt
        mid_theta = self.theta + 0.5 * delta_theta

        self.x += v_lin * np.cos(mid_theta) * dt
        self.y += v_lin * np.sin(mid_theta) * dt
        self.theta = wrap_angle(self.theta + delta_theta)

        return np.array([self.x, self.y, self.theta])


class RobotStateEKF:
    """
    Extended Kalman Filter for 2D Mobile Robot Localization.

    Prediction step:
      - Forward velocity from wheel encoders (encoder_vels).
      - Yaw rate from IMU gyro de-biased by estimated b_g.

    Update step:
      - Measurement: encoder-derived yaw rate w_enc.
      - Observation model: h(x) = gyro_z - b_g  (predicted gyro reading minus bias).
      - Jacobian: H = [0, 0, 0, 1]  (measurement sensitive only to b_g).
      - This continuously corrects gyro bias drift and bounds P.

    Parameters
    ----------
    wheel_radius : float
        Wheel radius in metres.
    L_eff : float
        Effective skid-steer track width used for encoder yaw-rate estimation.
    q_pos, q_theta, q_bias : float
        Process noise standard deviations for position, heading, gyro bias.
    r_w_enc : float
        Measurement noise std for encoder-derived yaw rate (rad/s).
    """

    def __init__(
        self,
        wheel_radius: float = 0.098,
        L_eff: float = 1.15,               # Effective skid-steer track width (m)
        initial_pose=None,
        q_pos: float = 0.02,               # Position process noise std (m/sqrt(s))
        q_theta: float = 0.005,            # Heading process noise std (rad/sqrt(s))
        q_bias: float = 1e-5,              # Gyro bias process noise std (rad/s/sqrt(s))
        r_w_enc: float = 0.30,             # Encoder yaw-rate measurement noise std (rad/s)
                                           # 0.30 reflects skid-steer lateral slip uncertainty
    ):
        self.r = wheel_radius
        self.L_eff = L_eff

        if initial_pose is None:
            self.x = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            self.x = np.array([initial_pose[0], initial_pose[1], initial_pose[2], 0.0])

        self.P = np.diag([0.01, 0.01, 0.01, 0.0001])
        self.Q = np.diag([q_pos**2, q_pos**2, q_theta**2, q_bias**2])
        self.R_update = np.array([[r_w_enc**2]])  # 1×1 measurement noise covariance

        self.v_est = 0.0
        self.w_est = 0.0

    # ------------------------------------------------------------------
    # Private helper: compute left/right wheel speeds from encoder array
    # ------------------------------------------------------------------
    def _wheel_speeds(self, encoder_vels: np.ndarray):
        """Returns (v_L, v_R) in m/s from [FL, FR, RR, RL] rad/s array."""
        w_L = 0.5 * (encoder_vels[0] + encoder_vels[3])
        w_R = -0.5 * (encoder_vels[1] + encoder_vels[2])
        return w_L * self.r, w_R * self.r

    # ------------------------------------------------------------------
    # Prediction step
    # ------------------------------------------------------------------
    def predict(self, encoder_vels: np.ndarray, gyro_z: float, dt: float):
        """
        EKF Prediction: propagates state using encoder forward velocity
        and de-biased gyro yaw rate.

        Motion model (midpoint integration):
          x_new     = x + v·cos(θ + Δθ/2)·dt
          y_new     = y + v·sin(θ + Δθ/2)·dt
          θ_new     = θ + (gyro_z - b_g)·dt
          b_g_new   = b_g          (random walk modelled by Q)

        Jacobian F = ∂f/∂x evaluated at current state.
        """
        v_L, v_R = self._wheel_speeds(encoder_vels)
        v_wheel = 0.5 * (v_L + v_R)

        x, y, theta, b_g = self.x

        w_unbiased = gyro_z - b_g          # de-biased yaw rate
        self.v_est = v_wheel
        self.w_est = w_unbiased

        delta_theta = w_unbiased * dt
        mid_theta = theta + 0.5 * delta_theta

        x_new     = x + v_wheel * np.cos(mid_theta) * dt
        y_new     = y + v_wheel * np.sin(mid_theta) * dt
        theta_new = wrap_angle(theta + delta_theta)
        b_g_new   = b_g

        self.x = np.array([x_new, y_new, theta_new, b_g_new])

        # Jacobian F = ∂f/∂[x, y, θ, b_g]
        #   ∂x_new/∂θ   = -v·sin(mid_θ)·dt        (via mid_θ = θ + Δθ/2)
        #   ∂x_new/∂b_g = +0.5·v·sin(mid_θ)·dt²   (mid_θ depends on b_g)
        #   ∂y_new/∂θ   = +v·cos(mid_θ)·dt
        #   ∂y_new/∂b_g = -0.5·v·cos(mid_θ)·dt²
        #   ∂θ_new/∂b_g = -dt
        F = np.eye(4)
        F[0, 2] = -v_wheel * np.sin(mid_theta) * dt
        F[0, 3] =  0.5 * v_wheel * np.sin(mid_theta) * dt * dt
        F[1, 2] =  v_wheel * np.cos(mid_theta) * dt
        F[1, 3] = -0.5 * v_wheel * np.cos(mid_theta) * dt * dt
        F[2, 3] = -dt

        self.P = F @ self.P @ F.T + self.Q * dt

    # ------------------------------------------------------------------
    # Update step — encoder yaw rate vs gyro
    # ------------------------------------------------------------------
    def update(self, encoder_vels: np.ndarray, gyro_z: float):
        """
        EKF Update: corrects gyro bias using the discrepancy between the
        encoder-derived yaw rate and the gyro-predicted yaw rate.

        Measurement model:
          z    = w_enc             (encoder yaw rate — independent of gyro)
          h(x) = gyro_z - b_g     (state-predicted yaw rate)
          H    = [0, 0, 0, -1]    (∂h/∂x: ∂h/∂b_g = -1, all others = 0)
        """
        v_L, v_R = self._wheel_speeds(encoder_vels)
        w_enc = (v_R - v_L) / self.L_eff   # encoder-derived yaw rate

        # STRICT ZUPT (Zero Velocity Update):
        # Skid-steer kinematics have a massive systematic scale-factor error during turns.
        # A Kalman Filter cannot filter out systematic bias, because the consistent 
        # sign of the innovation will integrate into b_g over time and destroy the heading.
        # We MUST only update b_g when the robot is completely stationary.
        v_wheel = 0.5 * (v_L + v_R)
        if abs(v_wheel) > 0.01 or abs(w_enc) > 0.01:
            return  # Skip update when moving

        b_g = self.x[3]
        w_pred = gyro_z - b_g               # state-predicted yaw rate

        innovation = w_enc - w_pred         # scalar measurement residual

        # h(x) = gyro_z - b_g  →  ∂h/∂b_g = -1
        H = np.array([[0.0, 0.0, 0.0, -1.0]])

        S = H @ self.P @ H.T + self.R_update   # 1×1
        K = self.P @ H.T @ np.linalg.inv(S)    # 4×1

        # State correction
        self.x = self.x + K.flatten() * innovation
        self.x[2] = wrap_angle(self.x[2])

        # Covariance update
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_update @ K.T

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def pose(self) -> np.ndarray:
        """Returns estimated [x, y, theta]."""
        return self.x[0:3]

    @property
    def gyro_bias(self) -> float:
        """Returns the current gyro bias estimate b_g (rad/s)."""
        return self.x[3]
