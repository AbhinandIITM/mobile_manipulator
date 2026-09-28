"""
sensors.py - Simulated Sensor Extraction and Noise Modeling for J100 in MuJoCo.

Extracts simulated IMU (Gyroscope, Accelerometer) and Wheel Encoder readings
with realistic Gaussian white noise, quantization, and bias drift.
"""

import numpy as np
import mujoco


class SensorManager:
    def __init__(
        self,
        model: mujoco.MjModel,
        gyro_noise_std: float = 0.015,       # rad/s (IMU gyro noise)
        gyro_bias_drift_std: float = 1e-4,   # rad/s^2 (random walk bias)
        accel_noise_std: float = 0.05,       # m/s^2 (IMU accel noise)
        accel_bias_drift_std: float = 5e-4,  # m/s^3 (random walk bias)
        encoder_noise_std: float = 0.02,     # rad/s (encoder velocity noise)
        encoder_ticks_per_rev: int = 4096,   # Encoder resolution
    ):
        self.model = model
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base')

        # Sensor IDs
        self.gyro_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_gyro')
        self.accel_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'imu_accel')

        self.enc_pos_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f'encoder_{name}_pos')
            for name in ['FL', 'FR', 'RR', 'RL']
        ]
        self.enc_vel_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f'encoder_{name}_vel')
            for name in ['FL', 'FR', 'RR', 'RL']
        ]

        # Noise parameters
        self.gyro_noise_std = gyro_noise_std
        self.gyro_bias_drift_std = gyro_bias_drift_std
        self.accel_noise_std = accel_noise_std
        self.accel_bias_drift_std = accel_bias_drift_std
        self.encoder_noise_std = encoder_noise_std
        self.encoder_resolution = (2.0 * np.pi) / encoder_ticks_per_rev

        # Internal state for bias random walks
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)

    def read_sensors(self, data: mujoco.MjData, dt: float):
        """
        Reads raw sensors and injects realistic noise and bias drift.

        Returns a dictionary with noisy sensor values and ground truth.
        """
        # 1. Update biases via random walk
        self.gyro_bias += np.random.normal(0.0, self.gyro_bias_drift_std * np.sqrt(dt), size=3)
        self.accel_bias += np.random.normal(0.0, self.accel_bias_drift_std * np.sqrt(dt), size=3)

        # 2. Raw IMU Gyro
        gyro_adr = self.model.sensor_adr[self.gyro_sensor_id]
        true_gyro = np.array(data.sensordata[gyro_adr : gyro_adr + 3])
        noisy_gyro = true_gyro + self.gyro_bias + np.random.normal(0.0, self.gyro_noise_std, size=3)

        # 3. Raw IMU Accel
        accel_adr = self.model.sensor_adr[self.accel_sensor_id]
        true_accel = np.array(data.sensordata[accel_adr : accel_adr + 3])
        noisy_accel = true_accel + self.accel_bias + np.random.normal(0.0, self.accel_noise_std, size=3)

        # 4. Wheel Encoders — velocity with quantization-scaled noise
        true_enc_vel = np.zeros(4)
        noisy_enc_vel = np.zeros(4)

        for i in range(4):
            vel_adr = self.model.sensor_adr[self.enc_vel_ids[i]]
            true_enc_vel[i] = data.sensordata[vel_adr]
            noisy_enc_vel[i] = true_enc_vel[i] + np.random.normal(0.0, self.encoder_noise_std)

        # 5. Ground truth pose and velocity
        gt_x = data.qpos[0]
        gt_y = data.qpos[1]

        # Heading from rotation matrix — forward direction is -R[:,0] for this model
        R = data.xmat[self.body_id].reshape(3, 3)
        forward_vec = -R[:, 0]
        gt_theta = np.arctan2(forward_vec[1], forward_vec[0])

        gt_v = np.dot(data.qvel[0:3], forward_vec)
        gt_w = np.dot(data.qvel[3:6], R[:, 2])

        return {
            "time": data.time,
            # Noisy Sensor Readings
            "gyro":        noisy_gyro,      # [wx, wy, wz] in rad/s
            "accel":       noisy_accel,     # [ax, ay, az] in m/s^2
            "encoder_vel": noisy_enc_vel,   # [FL, FR, RR, RL] in rad/s
            # Ground Truth (for logging/plotting only)
            "gt_pose":     np.array([gt_x, gt_y, gt_theta]),
            "gt_twist":    np.array([gt_v, gt_w]),
            "gt_gyro":     true_gyro,
            "gt_accel":    true_accel,
            "gt_encoder_vel": true_enc_vel,
        }
