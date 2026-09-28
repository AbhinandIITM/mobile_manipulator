import cv2
import numpy as np


class VisualDetector:
    """
    Perception module extracting Line-of-Sight (LOS) azimuth bearing and range
    to the target beacon using pinhole camera geometry on RGB-D frames.
    """

    def __init__(self, width: int = 320, height: int = 240, fovy_deg: float = 60.0):
        self.width = width
        self.height = height
        self.fovy_rad = np.radians(fovy_deg)

        # Pinhole camera intrinsic parameters
        self.fy = (height / 2.0) / np.tan(self.fovy_rad / 2.0)
        self.fx = self.fy  # square pixels
        self.cx = width / 2.0
        self.cy = height / 2.0

    def detect(self, rgb: np.ndarray, depth: np.ndarray = None) -> dict:
        """
        Detects red beacon in the image frame and computes LOS angle & range.
        """
        r = rgb[:, :, 0].astype(float)
        g = rgb[:, :, 1].astype(float)
        b = rgb[:, :, 2].astype(float)

        # Red color segmentation mask (target beacon)
        red_mask = (r > 100) & ((r - g) > 30) & ((r - b) > 30)
        total_pixels = np.sum(red_mask)

        if total_pixels < 20:
            return {
                "detected": False,
                "lambda_vis": 0.0,
                "range_vis": 0.0,
                "u": self.cx,
                "v": self.cy,
                "bbox": None,
            }

        # Calculate pixel centroid
        v_indices, u_indices = np.where(red_mask)
        u_bar = float(np.mean(u_indices))
        v_bar = float(np.mean(v_indices))

        # Bounding box
        xmin, xmax = int(np.min(u_indices)), int(np.max(u_indices))
        ymin, ymax = int(np.min(v_indices)), int(np.max(v_indices))

        # Compute horizontal LOS bearing angle (positive = turn left, negative = turn right)
        lambda_vis = -np.arctan2(u_bar - self.cx, self.fx)

        # Distance estimation
        range_vis = 5.0  # default fallback
        if depth is not None:
            masked_depths = depth[red_mask]
            valid_depths = masked_depths[(masked_depths > 0.1) & (masked_depths < 100.0)]
            if len(valid_depths) > 0:
                range_vis = float(np.median(valid_depths))

        return {
            "detected": True,
            "lambda_vis": float(lambda_vis),
            "range_vis": float(range_vis),
            "u": u_bar,
            "v": v_bar,
            "bbox": (xmin, ymin, xmax, ymax),
        }

    def build_hud_frame(
        self,
        rgb_frame: np.ndarray,
        depth_frame: np.ndarray,
        detection: dict,
        telemetry: dict,
    ) -> np.ndarray:
        """
        Builds a real-time OpenCV HUD composite displaying RGB tracking overlay and Depth map.
        """
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cx, cy = int(self.cx), int(self.cy)

        # Center reticle
        cv2.line(bgr, (cx - 15, cy), (cx + 15, cy), (80, 80, 80), 1)
        cv2.line(bgr, (cx, cy - 15), (cx, cy + 15), (80, 80, 80), 1)

        if detection["detected"]:
            u, v = int(detection["u"]), int(detection["v"])
            bbox = detection["bbox"]
            if bbox is not None:
                xmin, ymin, xmax, ymax = bbox
                cv2.rectangle(bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            cv2.drawMarker(bgr, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
            cv2.line(bgr, (cx, cy), (u, v), (0, 255, 255), 1, cv2.LINE_AA)

            dist_txt = f"{detection['range_vis']:.2f}m"
            cv2.putText(bgr, f"BEACON: {dist_txt}", (u + 10, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # HUD Telemetry Overlay
        mode_str = telemetry.get("mode", "UNKNOWN")
        mode_color = (0, 255, 0) if "TRACKING" in mode_str else (0, 165, 255)
        cv2.putText(bgr, f"MODE: {mode_str}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, mode_color, 2)

        rng = telemetry.get("target_range", 0.0)
        cv2.putText(bgr, f"Distance: {rng:4.2f} m", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        los_deg = np.degrees(telemetry.get("los_angle", 0.0))
        cv2.putText(bgr, f"LOS Bearing: {los_deg:+5.1f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(bgr, f"v_cmd: {telemetry.get('v_cmd', 0.0):.2f} m/s", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(bgr, f"w_cmd: {telemetry.get('w_cmd', 0.0):+.2f} rad/s", (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Depth map visualization
        if depth_frame is not None:
            depth_clipped = np.clip(depth_frame, 0.2, 10.0)
            depth_norm = ((depth_clipped - 0.2) / 9.8 * 255.0).astype(np.uint8)
            depth_bgr = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            cv2.putText(depth_bgr, "DEPTH MAP (0.2m - 10m)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        else:
            depth_bgr = np.zeros_like(bgr)

        composite = np.hstack([bgr, depth_bgr])
        return composite


class LOSGuidanceSystem:
    """
    Multi-rate Line-of-Sight (LOS) Distance & Bearing Regulation System.
    - Closed-loop PI Distance Regulator drives range asymptotically to target_distance.
    - Closed-loop PI Steering Regulator drives LOS bearing angle asymptotically to 0.0°.
    - Multi-rate: 500 Hz high-rate IMU Gyro state propagation + 30 Hz RGB-D visual corrections.
    """

    def __init__(
        self,
        v_max: float = 1.8,
        w_max: float = 3.5,
        target_distance: float = 0.80,
        kp_range: float = 2.0,
        ki_range: float = 1.0,
        kp_los: float = 3.5,
        ki_los: float = 1.8,
        kd_gyro: float = 0.30,
        fov_tracking_limit_deg: float = 25.0,
        track_width: float = 0.368,
        slip_factor: float = 0.30,
    ):
        self.v_max = v_max
        self.w_max = w_max
        self.target_distance = target_distance

        # PI Range Distance Regulation Gains
        self.kp_range = kp_range
        self.ki_range = ki_range

        # PI Bearing Steering Gains
        self.kp_los = kp_los
        self.ki_los = ki_los
        self.kd_gyro = kd_gyro

        self.fov_limit_rad = np.radians(fov_tracking_limit_deg)
        self.L = track_width
        self.slip = slip_factor

        # Estimated Line of Sight state
        self.los_angle = 0.0          # Current estimated LOS bearing (rad), [-pi, +pi]
        self.los_integral = 0.0       # Steering integral error accumulator
        self.range_integral = 0.0     # Distance integral error accumulator (eliminates moving target lag)
        self.target_range = 5.0       # Current estimated range (m)
        self.target_visible = False
        self.last_seen_time = -1.0
        self.has_prior = False

    def set_target_bearing_prior(self, bearing_rad: float, approx_range: float = 5.0):
        """
        Sets a prior bearing angle (e.g. when switching waypoints or re-tasking).
        Ensures the robot turns along the shortest angular arc directly toward the new target.
        """
        self.los_angle = (bearing_rad + np.pi) % (2.0 * np.pi) - np.pi
        self.target_range = approx_range
        self.target_visible = False
        self.has_prior = True
        self.range_integral = 0.0
        self.los_integral = 0.0

    def update_vision(self, detection: dict, current_time: float):
        """Processes 30 Hz visual detection frame."""
        if detection["detected"]:
            self.los_angle = detection["lambda_vis"]
            self.target_range = detection["range_vis"]
            self.target_visible = True
            self.has_prior = False
            self.last_seen_time = current_time
        else:
            self.target_visible = False

    def update_imu_and_control(self, gyro_z: float, dt: float, current_time: float) -> tuple[float, float, dict]:
        """
        Runs at 500 Hz: propagates LOS state using gyro and regulates distance and bearing.

        Returns
        -------
        v_cmd : float
            Desired forward linear velocity (m/s)
        w_cmd : float
            Desired angular yaw rate (rad/s)
        telemetry : dict
            Diagnostic metrics
        """
        # High-frequency kinematic LOS angle propagation: d(lambda)/dt = -w_z
        self.los_angle -= gyro_z * dt

        # Wrap angle to [-pi, pi]
        self.los_angle = (self.los_angle + np.pi) % (2.0 * np.pi) - np.pi

        time_since_seen = current_time - self.last_seen_time if self.last_seen_time >= 0 else 999.0
        is_aligned = abs(self.los_angle) <= self.fov_limit_rad
        e_range = self.target_range - self.target_distance

        if self.target_visible and is_aligned:
            # 1. Closed-Loop Distance & Bearing Tracking (Full Visual Lock)
            self.has_prior = False
            self.range_integral = float(np.clip(self.range_integral + e_range * dt, -1.5, 1.5))
            v_raw = self.kp_range * e_range + self.ki_range * self.range_integral

            # Speed is scaled by heading alignment cos(lambda)
            v_cmd = float(np.clip(v_raw, -0.4, self.v_max)) * max(0.0, np.cos(self.los_angle))

            # Steering regulation (PI on lambda + gyro rate damping)
            self.los_integral = float(np.clip(self.los_integral + self.los_angle * dt, -0.4, 0.4))
            w_cmd = self.kp_los * self.los_angle + self.ki_los * self.los_integral - self.kd_gyro * gyro_z

            mode = "TRACKING"

        elif self.has_prior or time_since_seen < 1.0:
            # 2. Prior Bearing Active or Recent Target Track -> Shortest Arc Turn toward target
            self.los_integral *= 0.95
            self.range_integral *= 0.95
            turn_dir = np.sign(self.los_angle) if abs(self.los_angle) > 1e-3 else 1.0
            turn_rate = np.clip(self.kp_los * abs(self.los_angle), 0.6, self.w_max)

            w_cmd = turn_dir * turn_rate - self.kd_gyro * gyro_z
            v_cmd = 0.0
            mode = f"RE-ORIENTING ({'RIGHT' if turn_dir < 0 else 'LEFT'})"

        else:
            # 3. Target Lost & No Prior -> Sweep continuously in Positive Direction (+w, CCW / Left)
            self.los_integral *= 0.95
            self.range_integral *= 0.95
            v_cmd = 0.0
            w_cmd = 0.85 - self.kd_gyro * gyro_z
            mode = "EXPLORING (POSITIVE / LEFT)"

        # Clip commands to safety limits
        v_cmd = float(np.clip(v_cmd, -0.5, self.v_max))
        w_cmd = float(np.clip(w_cmd, -self.w_max, self.w_max))

        telemetry = {
            "mode": mode,
            "los_angle": self.los_angle,
            "target_range": self.target_range,
            "target_distance": self.target_distance,
            "range_error": e_range,
            "target_visible": self.target_visible,
            "v_cmd": v_cmd,
            "w_cmd": w_cmd,
        }

        return v_cmd, w_cmd, telemetry

    def compute_actuator_velocities(self, v_cmd: float, w_cmd: float, wheel_radius: float = 0.098) -> np.ndarray:
        """
        Maps (v, w) commands to 4-wheel skid-steer actuator velocities with slip compensation.
        Returns [FL, FR, RR, RL] actuator velocity commands in rad/s.
        """
        slip = self.slip if abs(v_cmd) > 0.05 else 0.34
        w_eff = w_cmd / slip

        v_L = v_cmd - w_eff * (self.L / 2.0)
        v_R = v_cmd + w_eff * (self.L / 2.0)

        w_wheel_L = v_L / wheel_radius
        w_wheel_R = v_R / wheel_radius

        # [FL, FR, RR, RL]
        return np.array([w_wheel_L, w_wheel_R, w_wheel_R, w_wheel_L])
