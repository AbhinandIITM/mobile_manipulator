"""
controllers.py - Path and Trajectory Following Controllers for J100 Mobile Robot.

Implements:
1. PurePursuitController: Geometric lookahead path tracking with curvature-adaptive velocity.
2. WaypointPath: Dense 2D waypoint path generator and spatial search.
3. ReferenceTrajectory: Analytical trajectories (Figure-8, Circle, Square).
"""

import numpy as np
from odometry import wrap_angle


class ReferenceTrajectory:
    """
    Generates reference paths and mathematical trajectories.
    """
    @staticmethod
    def figure_eight(scale: float = 3.0, num_points: int = 600):
        """
        Generates a dense, closed 2D waypoint path for Lemniscate of Bernoulli (Figure-8).
        """
        t_vals = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
        x = scale * np.sin(t_vals)
        y = (scale / 2.0) * np.sin(2.0 * t_vals)
        return np.column_stack([x, y])

    @staticmethod
    def circle(radius: float = 2.5, num_points: int = 400):
        """
        Generates a dense, closed circular waypoint path.
        """
        t_vals = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
        x = radius * np.sin(t_vals)
        y = radius * (1.0 - np.cos(t_vals))
        return np.column_stack([x, y])

    @staticmethod
    def square(size: float = 4.0, num_points_per_side: int = 100):
        """
        Generates a closed square waypoint path with rounded corners.
        """
        s = size / 2.0
        p1 = np.column_stack([np.linspace(-s, s, num_points_per_side), np.full(num_points_per_side, -s)])
        p2 = np.column_stack([np.full(num_points_per_side, s), np.linspace(-s, s, num_points_per_side)])
        p3 = np.column_stack([np.linspace(s, -s, num_points_per_side), np.full(num_points_per_side, s)])
        p4 = np.column_stack([np.full(num_points_per_side, -s), np.linspace(s, -s, num_points_per_side)])
        return np.vstack([p1, p2, p3, p4])


class WaypointPath:
    """
    Encapsulates a 2D waypoint path and spatial lookahead search for Pure Pursuit.
    """
    def __init__(self, waypoints: np.ndarray, is_closed: bool = True):
        self.waypoints = np.asarray(waypoints, dtype=np.float64)
        self.is_closed = is_closed
        self.num_points = len(self.waypoints)

    def find_lookahead_point(self, current_pos: np.ndarray, lookahead_dist: float = 0.6):
        """
        Finds the closest point on the path and projects forward by lookahead_dist.
        
        :param current_pos: Current robot [x, y] in world frame
        :param lookahead_dist: Desired lookahead distance L_d (meters)
        :return: (target_point, closest_idx, cross_track_distance)
        """
        dx = self.waypoints[:, 0] - current_pos[0]
        dy = self.waypoints[:, 1] - current_pos[1]
        dists_sq = dx**2 + dy**2
        min_idx = int(np.argmin(dists_sq))
        cross_track_dist = np.sqrt(dists_sq[min_idx])

        # Search forward along path for the lookahead point
        lookahead_sq = lookahead_dist**2
        for offset in range(self.num_points):
            idx = (min_idx + offset) % self.num_points if self.is_closed else min(min_idx + offset, self.num_points - 1)
            if dists_sq[idx] >= lookahead_sq:
                return self.waypoints[idx], min_idx, cross_track_dist

        # Fallback if no point exceeds lookahead
        fallback_idx = (min_idx + 10) % self.num_points if self.is_closed else self.num_points - 1
        return self.waypoints[fallback_idx], min_idx, cross_track_dist


class PurePursuitController:
    """
    Pure Pursuit Geometric Path Tracker with Curvature-Adaptive Speed Control.

    Classical formula: κ = 2·sin(α) / L_d  where L_d is the fixed lookahead
    distance parameter — NOT the instantaneous distance to the target point.
    """
    def __init__(
        self,
        lookahead_distance: float = 0.6,     # Lookahead distance L_d (m)
        v_target: float = 0.65,               # Desired cruising speed (m/s)
        v_max: float = 1.2,
        w_max: float = 2.5,
        accel_max: float = 3.0,
        w_dot_max: float = 5.0,              # Max yaw-rate change rate (rad/s²)
    ):
        self.L_d = lookahead_distance
        self.v_target = v_target
        self.v_max = v_max
        self.w_max = w_max
        self.accel_max = accel_max
        self.w_dot_max = w_dot_max

        self.prev_v = 0.0
        self.prev_w = 0.0

    def compute_control(
        self,
        current_pose: np.ndarray,
        target_point: np.ndarray,
        dt: float,
        target_speed: float = None,
    ):
        """
        Computes (v, w) command to track the target lookahead point.

        Pure Pursuit curvature: κ = 2·sin(α) / L_d
          α  — heading error to the lookahead point (robot frame)
          L_d — fixed lookahead distance parameter (not target distance)
        """
        if target_speed is None:
            target_speed = self.v_target

        x, y, theta = current_pose
        tx, ty = target_point[0:2]

        dx = tx - x
        dy = ty - y
        dist = np.hypot(dx, dy)

        if dist < 1e-4:
            return 0.0, 0.0, {"dist": 0.0, "alpha": 0.0, "kappa": 0.0}

        # Heading error from robot heading to target point (robot body frame)
        angle_to_target = np.arctan2(dy, dx)
        alpha = wrap_angle(angle_to_target - theta)

        # Classical Pure Pursuit curvature — denominator is the fixed L_d parameter,
        # not dist. This gives consistent steering gain regardless of how far the
        # robot drifts off the path.
        kappa = (2.0 * np.sin(alpha)) / self.L_d

        # Curvature-adaptive speed scaling: smoothly slows on sharp corners
        v_cmd = target_speed / (1.0 + 1.2 * abs(kappa))
        w_cmd = v_cmd * kappa

        # Acceleration and yaw-rate-change limiting
        dv = np.clip(v_cmd - self.prev_v, -self.accel_max * dt, self.accel_max * dt)
        dw = np.clip(w_cmd - self.prev_w, -self.w_dot_max * dt, self.w_dot_max * dt)

        v_out = np.clip(self.prev_v + dv, -self.v_max, self.v_max)
        w_out = np.clip(self.prev_w + dw, -self.w_max, self.w_max)

        self.prev_v = v_out
        self.prev_w = w_out

        errors = {"dist": dist, "alpha": alpha, "kappa": kappa}
        return v_out, w_out, errors
