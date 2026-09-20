import numpy as np


class TrajectoryGenerator:
    """
    Generates smooth 3D Cartesian trajectories with minimum-jerk lead-in
    for robot manipulator end-effector tracking.
    """

    def __init__(
        self,
        traj_type: str = "figure8",
        center: np.ndarray = np.array([-0.10, 0.45, 0.45]),
        scale: np.ndarray = np.array([0.16, 0.12, 0.08]),
        period: float = 6.0,
        lead_in_time: float = 2.0,
    ):
        """
        Parameters
        ----------
        traj_type : str
            'figure8', 'circle', 'spiral', or 'waypoints'
        center : np.ndarray
            Center [x, y, z] of the trajectory in workspace (meters)
        scale : np.ndarray
            [rx, ry, rz] amplitudes of the trajectory
        period : float
            Period (seconds) for one full cycle
        lead_in_time : float
            Time in seconds to smoothly blend from initial pose to trajectory start
        """
        self.traj_type = traj_type
        self.center = np.asarray(center, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.period = period
        self.omega = 2.0 * np.pi / period
        self.lead_in_time = lead_in_time

        # Fixed nominal downward orientation (rotation matrix)
        # End-effector pointing downward: Z points -Z_world, X points X_world, Y points -Y_world
        self.target_rot = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0]
        ])

        # Predefined 3D waypoints for 'waypoints' mode
        self.waypoint_list = np.array([
            self.center + np.array([ 0.12, -0.10,  0.05]),
            self.center + np.array([ 0.12,  0.10, -0.05]),
            self.center + np.array([-0.12,  0.10,  0.05]),
            self.center + np.array([-0.12, -0.10, -0.05]),
        ])

    def _nominal_traj(self, t: float):
        """Returns nominal trajectory (pos, vel) at parameter t."""
        if self.traj_type == "figure8":
            # 3D Lemniscate (Figure-8 in XY with vertical Z oscillation)
            wt = self.omega * t
            pos = self.center + np.array([
                self.scale[0] * np.sin(wt),
                self.scale[1] * np.sin(2.0 * wt) * 0.5,
                self.scale[2] * np.cos(wt),
            ])
            vel = np.array([
                self.scale[0] * self.omega * np.cos(wt),
                self.scale[1] * self.omega * np.cos(2.0 * wt),
                -self.scale[2] * self.omega * np.sin(wt),
            ])

        elif self.traj_type == "circle":
            # 3D inclined circle
            wt = self.omega * t
            pos = self.center + np.array([
                self.scale[0] * np.cos(wt),
                self.scale[1] * np.sin(wt),
                self.scale[2] * np.sin(wt),
            ])
            vel = np.array([
                -self.scale[0] * self.omega * np.sin(wt),
                self.scale[1] * self.omega * np.cos(wt),
                self.scale[2] * self.omega * np.cos(wt),
            ])

        elif self.traj_type == "spiral":
            # 3D Helix / Spiral (circular orbit + sinusoidal elevation)
            wt = self.omega * t
            pos = self.center + np.array([
                self.scale[0] * np.cos(wt),
                self.scale[1] * np.sin(wt),
                self.scale[2] * np.sin(0.5 * wt),
            ])
            vel = np.array([
                -self.scale[0] * self.omega * np.sin(wt),
                self.scale[1] * self.omega * np.cos(wt),
                self.scale[2] * 0.5 * self.omega * np.cos(0.5 * wt),
            ])

        elif self.traj_type == "waypoints":
            # Smooth piecewise minimum-jerk spline through 3D waypoints
            num_wp = len(self.waypoint_list)
            seg_duration = self.period / num_wp
            seg_idx = int((t % self.period) / seg_duration)
            next_idx = (seg_idx + 1) % num_wp

            p0 = self.waypoint_list[seg_idx]
            p1 = self.waypoint_list[next_idx]

            tau = (t % seg_duration) / seg_duration
            s = 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)
            s_dot = (30.0 * (tau**2) - 60.0 * (tau**3) + 30.0 * (tau**4)) / seg_duration

            pos = (1.0 - s) * p0 + s * p1
            vel = s_dot * (p1 - p0)

        else:
            raise ValueError(f"Unknown traj_type: '{self.traj_type}'. Choose from 'figure8', 'circle', 'spiral', 'waypoints'.")

        return pos, vel

    def get_target(self, t: float, init_pos: np.ndarray):
        """
        Computes the target position, velocity, orientation, and angular velocity at time t.
        Uses 5th-order minimum jerk polynomial during lead-in phase.
        """
        init_pos = np.asarray(init_pos, dtype=float)
        p_start_traj, _ = self._nominal_traj(0.0)

        if t < self.lead_in_time:
            # 5th-order minimum jerk blend from initial end-effector position
            tau = t / self.lead_in_time
            s = 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)
            s_dot = (30.0 * (tau**2) - 60.0 * (tau**3) + 30.0 * (tau**4)) / self.lead_in_time

            pos = (1.0 - s) * init_pos + s * p_start_traj
            vel = s_dot * (p_start_traj - init_pos)
        else:
            # Nominal steady-state periodic trajectory
            t_traj = t - self.lead_in_time
            pos, vel = self._nominal_traj(t_traj)

        rot = self.target_rot
        angular_vel = np.zeros(3)

        return pos, vel, rot, angular_vel

    def sample_path(self, num_points: int = 150) -> np.ndarray:
        """
        Samples the 3D trajectory path for visualization in MuJoCo viewer.
        """
        t_vals = np.linspace(0, self.period, num_points, endpoint=False)
        pts = np.zeros((num_points, 3))
        for i, t in enumerate(t_vals):
            pts[i], _ = self._nominal_traj(t)
        return pts
