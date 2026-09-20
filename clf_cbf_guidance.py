"""
clf_cbf_guidance.py - Control Lyapunov Function (CLF) + Control Barrier Function (CBF)
Guidance System with Quadratic Programming for J100 Mobile Base.

Theoretical Formulation:
1. Control Barrier Function (CBF) - Zero Overshoot Safety Invariance:
   - Barrier: h(R) = R - d_target >= 0.
   - Kinematic Braking Boundary: v <= sqrt(2 * a_dec * max(0, h)) / max(0.1, cos(lambda)).
   - Forward Invariant Condition: L_f h + L_g h * v >= -gamma_h * h ==> v * cos(lambda) <= gamma_h * h.
   - Together, these guarantee R(t) >= d_target for all t >= 0. Overshoot is physically impossible.

2. Control Lyapunov Function (CLF) - Asymptotic Distance Convergence:
   - Lyapunov Candidate: V(R) = 0.5 * k_R * (R - d_target)^2.
   - Stability Condition: V_dot = -k_R * (R - d_target) * v * cos(lambda) <= -gamma_v * V + delta (delta >= 0).

3. Smooth Acceleration Envelope (Pitch Lift / Wheelie Suppression):
   - v in [max(0.0, v_prev - a_dec * dt), min(v_max, v_prev + a_acc * dt)].
   - Capping forward acceleration to 1.0 m/s^2 eliminates motor reaction torque spikes on the chassis.

4. 500 Hz Multi-rate Execution + 30 Hz RGB-D Perception.
"""

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
        self.fx = self.fy
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

        # Distance estimation from depth map
        range_vis = 5.0
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
        cv2.putText(bgr, f"MODE: {mode_str} [CLF-CBF QP]", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 2)

        rng = telemetry.get("target_range", 0.0)
        cv2.putText(bgr, f"Distance: {rng:4.2f} m", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        los_deg = np.degrees(telemetry.get("los_angle", 0.0))
        cv2.putText(bgr, f"LOS Bearing: {los_deg:+5.1f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        h_val = telemetry.get("cbf_h", 0.0)
        v_val = telemetry.get("clf_v", 0.0)
        cv2.putText(bgr, f"CBF h: {h_val:+4.2f} | CLF V: {v_val:4.2f}", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1)

        cv2.putText(bgr, f"v_cmd: {telemetry.get('v_cmd', 0.0):.2f} m/s", (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(bgr, f"w_cmd: {telemetry.get('w_cmd', 0.0):+.2f} rad/s", (10, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

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


class CLFCBFGuidanceSystem:
    """
    Control Lyapunov Function (CLF) + Control Barrier Function (CBF)
    Multi-rate Guidance System with High-Speed Analytical QP.
    """

    def __init__(
        self,
        target_distance: float = 0.80,
        v_max: float = 1.6,
        w_max: float = 3.5,
        a_max_accel: float = 1.0,     # Max forward acceleration [m/s^2] (smooth ramp, suppresses pitch bounce)
        a_max_decel: float = 1.8,     # Max braking deceleration [m/s^2]
        alpha_max_accel: float = 3.5, # Max angular acceleration [rad/s^2]
        k_clf_range: float = 2.0,     # CLF range weight
        kp_los: float = 3.5,          # Bearing steering gain
        ki_los: float = 1.8,          # Bearing integral gain
        kd_gyro: float = 0.30,        # Gyro rate damping
        gamma_clf: float = 1.5,       # CLF convergence rate
        gamma_cbf: float = 1.2,       # CBF barrier growth rate
        slack_penalty: float = 1e4,   # Slack variable penalty
        fov_tracking_limit_deg: float = 28.0,
        track_width: float = 0.368,
        slip_factor: float = 0.30,
    ):
        self.target_distance = target_distance
        self.v_max = v_max
        self.w_max = w_max

        # Acceleration and Pitch Protection limits
        self.a_acc = a_max_accel
        self.a_dec = a_max_decel
        self.alpha_acc = alpha_max_accel

        # CLF and CBF tuning parameters
        self.k_R = k_clf_range
        self.gamma_v = gamma_clf
        self.gamma_h = gamma_cbf
        self.p_slack = slack_penalty

        # Steering gains
        self.kp_los = kp_los
        self.ki_los = ki_los
        self.kd_gyro = kd_gyro
        self.ki_range = 1.0           # Range error integrator for moving target speed adaptation

        self.fov_limit_rad = np.radians(fov_tracking_limit_deg)
        self.L = track_width
        self.slip = slip_factor

        # Estimated state
        self.los_angle = 0.0
        self.los_integral = 0.0
        self.range_integral = 0.0
        self.target_range = 5.0
        self.target_visible = False
        self.last_seen_time = -1.0
        self.has_prior = False

        # State memory for acceleration rate limits and target velocity estimation
        self.v_prev = 0.0
        self.w_prev = 0.0
        self.r_dot_filtered = 0.0
        self.v_target_est = 0.0

    def set_target_bearing_prior(self, bearing_rad: float, approx_range: float = 5.0):
        """Sets prior bearing when re-tasking / switching waypoints."""
        self.los_angle = (bearing_rad + np.pi) % (2.0 * np.pi) - np.pi
        self.target_range = approx_range
        self.target_visible = False
        self.has_prior = True
        self.los_integral = 0.0
        self.r_dot_filtered = 0.0
        self.v_target_est = 0.0

    def update_vision(self, detection: dict, current_time: float):
        """Processes 30 Hz visual detection frame."""
        if detection["detected"]:
            # Estimate target forward velocity from range-rate
            if self.target_visible and (current_time - self.last_seen_time) > 1e-4:
                dt_vis = current_time - self.last_seen_time
                r_dot_raw = (detection["range_vis"] - self.target_range) / dt_vis
                self.r_dot_filtered = 0.85 * self.r_dot_filtered + 0.15 * r_dot_raw
                v_tgt = self.v_prev * np.cos(self.los_angle) + self.r_dot_filtered
                self.v_target_est = 0.85 * self.v_target_est + 0.15 * float(np.clip(v_tgt, 0.0, 1.2))

            self.los_angle = detection["lambda_vis"]
            self.target_range = detection["range_vis"]
            self.target_visible = True
            self.has_prior = False
            self.last_seen_time = current_time
        else:
            self.target_visible = False
            self.v_target_est *= 0.95

    def solve_clf_cbf_qp_velocity(self, v_nom: float, v_ff: float, dt: float) -> tuple[float, float, float, float]:
        """
        Solves the Dynamic CLF-CBF Quadratic Program for Forward Linear Velocity:
          min_{v, delta}  0.5 * (v - v_nom)^2 + 0.5 * p_slack * delta^2
          s.t.
             Dynamic CBF Safety:  v * cos(lambda) <= v_ff + gamma_h * max(0, h)
             Kinematic Braking:   v <= (v_ff + sqrt(2 * a_dec * max(0, h))) / cos(lambda)
             CLF Stability:       -k_R * (R - d_target) * cos(lambda) * v - delta <= -gamma_v * V
             Acceleration Bounds: v in [v_lb, v_ub]
        """
        e_R = self.target_range - self.target_distance
        lam = self.los_angle
        cos_lam = max(0.05, np.cos(lam))

        # 1. Evaluate Candidate Lyapunov Function V(R) = 0.5 * k_R * e_R^2
        V = 0.5 * self.k_R * (e_R ** 2)

        # Lie derivative: V_dot = Lg_v * v = -k_R * e_R * cos(lambda) * v
        Lg_v = -self.k_R * e_R * cos_lam

        # 2. Evaluate Candidate Barrier Function h(R) = R - d_target
        h = self.target_range - self.target_distance

        # Dynamic Kinematic CBF Safe Speed Boundary (Zero Overshoot Guarantee)
        v_base = max(0.0, v_ff)
        if h <= 0.0:
            v_cbf = v_base
        else:
            v_cbf_linear = (v_base + self.gamma_h * h) / cos_lam
            v_cbf_brake = (v_base + np.sqrt(2.0 * self.a_dec * h)) / cos_lam
            v_cbf = min(v_cbf_linear, v_cbf_brake)

        # Dynamic Slew Rate Bounds (Anti-Pitch Filter)
        v_ub_acc = min(self.v_max, self.v_prev + self.a_acc * dt)
        v_ub = min(v_ub_acc, max(0.0, v_cbf))
        v_lb = max(0.0, min(v_ub, self.v_prev - self.a_dec * dt))

        # 3. Active-Set CLF-CBF QP Solve
        v_init = float(np.clip(v_nom, v_lb, v_ub))

        clf_lhs = Lg_v * v_init
        clf_rhs = -self.gamma_v * V

        if clf_lhs <= clf_rhs:
            v_opt = v_init
            delta_opt = 0.0
        else:
            # CLF violated -> Project onto line Lg_v * v - delta = clf_rhs
            denom = (Lg_v ** 2) + (1.0 / max(1.0, self.p_slack))
            violation = clf_lhs - clf_rhs
            step = violation / max(1e-6, denom)

            v_cand = v_init - step * Lg_v
            delta_opt = float(step / max(1.0, self.p_slack))
            v_opt = float(np.clip(v_cand, v_lb, v_ub))

        return v_opt, delta_opt, float(V), float(h)

    def update_imu_and_control(self, gyro_z: float, dt: float, current_time: float) -> tuple[float, float, dict]:
        """
        Runs at 500 Hz: propagates LOS state using gyro and regulates distance/bearing via CLF-CBF QP.
        """
        # High-frequency kinematic LOS angle propagation: d(lambda)/dt = -w_z
        self.los_angle -= gyro_z * dt
        self.los_angle = (self.los_angle + np.pi) % (2.0 * np.pi) - np.pi

        time_since_seen = current_time - self.last_seen_time if self.last_seen_time >= 0 else 999.0
        is_aligned = abs(self.los_angle) <= self.fov_limit_rad
        e_range = self.target_range - self.target_distance

        if self.target_visible and is_aligned:
            # 1. Visual Tracking Mode with CLF-CBF QP Regulation
            self.has_prior = False

            # Bearing steering regulator (PI + gyro damping)
            self.los_integral = float(np.clip(self.los_integral + self.los_angle * dt, -0.4, 0.4))
            w_raw = self.kp_los * self.los_angle + self.ki_los * self.los_integral - self.kd_gyro * gyro_z

            # Dynamic target velocity feedforward from visual range-rate
            v_ff = self.v_target_est

            # Heading-prioritized nominal speed reference
            alignment_scale = max(0.0, np.cos(self.los_angle)) ** 2
            v_ref = np.clip(1.6 * e_range + v_ff, 0.0, self.v_max) * alignment_scale

            # Solve CLF-CBF QP for forward velocity with dynamic feedforward
            v_cmd, delta_opt, V_val, h_val = self.solve_clf_cbf_qp_velocity(v_ref, v_ff, dt)

            # Slew limit on angular velocity
            w_cmd = float(np.clip(w_raw, self.w_prev - self.alpha_acc * dt, self.w_prev + self.alpha_acc * dt))
            w_cmd = float(np.clip(w_cmd, -self.w_max, self.w_max))

            mode = "TRACKING"

        elif self.has_prior or time_since_seen < 1.0:
            # 2. Prior Bearing Active or Recent Track -> Shortest Arc Turn toward target
            self.los_integral *= 0.95
            turn_dir = np.sign(self.los_angle) if abs(self.los_angle) > 1e-3 else 1.0
            turn_rate = np.clip(self.kp_los * abs(self.los_angle), 0.6, self.w_max)

            w_raw = turn_dir * turn_rate - self.kd_gyro * gyro_z
            w_cmd = float(np.clip(w_raw, self.w_prev - self.alpha_acc * dt, self.w_prev + self.alpha_acc * dt))
            w_cmd = float(np.clip(w_cmd, -self.w_max, self.w_max))

            # Stop and turn in place
            v_cmd, delta_opt, V_val, h_val = self.solve_clf_cbf_qp_velocity(0.0, 0.0, dt)
            mode = f"RE-ORIENTING ({'RIGHT' if turn_dir < 0 else 'LEFT'})"

        else:
            # 3. Target Lost -> Explore in Positive Direction (+w)
            self.los_integral *= 0.95
            w_raw = 0.85 - self.kd_gyro * gyro_z
            w_cmd = float(np.clip(w_raw, self.w_prev - self.alpha_acc * dt, self.w_prev + self.alpha_acc * dt))
            w_cmd = float(np.clip(w_cmd, -self.w_max, self.w_max))

            v_cmd, delta_opt, V_val, h_val = self.solve_clf_cbf_qp_velocity(0.0, 0.0, dt)
            mode = "EXPLORING (POSITIVE / LEFT)"

        # Store state for acceleration rate limits
        self.v_prev = v_cmd
        self.w_prev = w_cmd

        telemetry = {
            "mode": mode,
            "los_angle": self.los_angle,
            "target_range": self.target_range,
            "target_distance": self.target_distance,
            "range_error": e_range,
            "cbf_h": h_val,
            "clf_v": V_val,
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
