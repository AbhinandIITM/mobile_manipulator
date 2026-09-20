"""
simulate_j100.py - Live Simulation with Pure Pursuit Path Tracking and EKF Localization.

Features:
- Pure Pursuit Geometric Path Tracking (Smooth, robust, lookahead-based).
- Waypoint Path Generation (Figure-8, Circle, Square).
- Simulated IMU & Wheel Encoders with noise & drift.
- Real-time EKF Pose Estimation.
- Automatic Analysis Plotting on Viewer Close.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

from sensors import SensorManager
from odometry import WheelOdometry, RobotStateEKF, wrap_angle
from controllers import ReferenceTrajectory, WaypointPath, PurePursuitController


def set_actuator_vels(data, v: float, w: float, wheel_radius: float = 0.098, track_width: float = 0.368):
    """
    Applies wheel velocity commands with skid-steer slip compensation.
    """
    slip = 0.30 if abs(v) > 0.05 else 0.34
    w_eff = w / slip

    v_L = v - w_eff * (track_width / 2.0)
    v_R = v + w_eff * (track_width / 2.0)

    # FR & RR actuators have gear="-1" in XML
    data.ctrl[0] = v_L / wheel_radius  # FL
    data.ctrl[1] = v_R / wheel_radius  # FR
    data.ctrl[2] = v_R / wheel_radius  # RR
    data.ctrl[3] = v_L / wheel_radius  # RL


def plot_pure_pursuit_analysis(
    t_arr, path_waypoints, gt_arr, ekf_arr, odom_arr,
    cross_track_errs, alphas, v_cmds, w_cmds, bias_arr
):
    """
    Generates a 5-panel analysis figure for the Pure Pursuit controller.
    """
    if len(t_arr) == 0:
        return

    fig = plt.figure(figsize=(18, 11))

    # --- Panel 1: 2D Path Tracking ---
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(path_waypoints[:, 0], path_waypoints[:, 1], 'r--', label='Reference Path (Figure-8)', linewidth=2.5)
    ax1.plot(gt_arr[:, 0], gt_arr[:, 1], 'b-', label='Ground Truth', linewidth=2.0)
    ax1.plot(ekf_arr[:, 0], ekf_arr[:, 1], 'g:', label='EKF Estimate', linewidth=2.2)
    ax1.plot(odom_arr[:, 0], odom_arr[:, 1], 'm-.', label='Raw Odometry (Drift)', alpha=0.5)
    ax1.set_title('2D Pure Pursuit Path Tracking', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X Position [m]')
    ax1.set_ylabel('Y Position [m]')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.axis('equal')

    # --- Panel 2: Cross-Track Error ---
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(t_arr, cross_track_errs * 100.0, 'navy', label='Cross-Track Error [cm]', linewidth=2.0)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('Cross-Track Distance Error', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Cross-Track Error [cm]')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- Panel 3: Controller Velocity Commands (v, w) ---
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(t_arr, v_cmds, 'b-', label='Linear Velocity $v_{cmd}$ [m/s]', linewidth=2.0)
    ax3.plot(t_arr, w_cmds, 'r--', label=r'Angular Velocity $\omega_{cmd}$ [rad/s]', linewidth=2.0)
    ax3.set_title(r'Pure Pursuit Velocity Commands ($v, \omega$)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Command Value')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, linestyle='--', alpha=0.6)

    # --- Panel 4: Heading Orientation (GT vs EKF) ---
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(t_arr, np.rad2deg(gt_arr[:, 2]), 'b-', label=r'Ground Truth Heading $\theta_{gt}$', linewidth=2.0)
    ax4.plot(t_arr, np.rad2deg(ekf_arr[:, 2]), 'g:', label=r'EKF Heading $\hat{\theta}$', linewidth=2.0)
    ax4.plot(t_arr, np.rad2deg(alphas), color='orange', label=r'Lookahead Angle $\alpha$ [deg]', linewidth=1.5, alpha=0.8)
    ax4.set_title('Heading & Lookahead Angle Error', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Angle [deg]')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, linestyle='--', alpha=0.6)

    # --- Panel 5: EKF Gyro Bias Convergence ---
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(t_arr, bias_arr * 1000.0, color='darkgreen', linewidth=2.0, label=r'EKF Gyro Bias $\hat{b}_g$ [mrad/s]')
    ax5.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_title('EKF Gyro Bias Estimation (Update Step)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Gyro Bias [mrad/s]')
    ax5.legend(loc='best', fontsize=9)
    ax5.grid(True, linestyle='--', alpha=0.6)

    # --- Panel 6: EKF vs GT Position Error over time ---
    ax6 = plt.subplot(2, 3, 6)
    ekf_pos_err = np.hypot(ekf_arr[:, 0] - gt_arr[:, 0], ekf_arr[:, 1] - gt_arr[:, 1])
    ax6.plot(t_arr, ekf_pos_err * 100.0, color='crimson', linewidth=2.0, label='EKF Position Error [cm]')
    ax6.axhline(np.mean(ekf_pos_err) * 100.0, color='gray', linestyle='--', alpha=0.7,
                label=f'Mean = {np.mean(ekf_pos_err)*100.0:.1f} cm')
    ax6.set_title('EKF Position Error vs Ground Truth', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Position Error [cm]')
    ax6.legend(loc='best', fontsize=9)
    ax6.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_filename = 'pure_pursuit_tracking_analysis.png'
    plt.savefig(plot_filename, dpi=200)
    print(f"\nAnalysis plot saved to {plot_filename}")
    plt.show()


def main():
    print("Loading the J100 MuJoCo model...")
    model = mujoco.MjModel.from_xml_path('j100_asm/j100_asm.xml')
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base')

    # 1. Generate 2D Reference Waypoint Path (Figure-8)
    waypoints = ReferenceTrajectory.figure_eight(scale=3.0, num_points=600)
    path = WaypointPath(waypoints, is_closed=True)

    # Initial robot pose at start of path
    p0 = waypoints[0]
    p1 = waypoints[1]
    th0 = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
    psi0 = th0 + np.pi

    data.qpos[0] = p0[0]
    data.qpos[1] = p0[1]
    data.qpos[2] = 0.065
    data.qpos[3] = np.cos(psi0 / 2.0)
    data.qpos[6] = np.sin(psi0 / 2.0)
    mujoco.mj_forward(model, data)

    init_pose = np.array([p0[0], p0[1], th0])

    # 2. Instantiate Sensor Suite, Estimator, and Pure Pursuit Controller
    sensors = SensorManager(model)
    odom = WheelOdometry(initial_pose=init_pose, track_width=1.227)
    ekf = RobotStateEKF(initial_pose=init_pose)
    controller = PurePursuitController(
        lookahead_distance=1.2,   # Increased for higher speed stability
        v_target=1.5,             # 1.5 m/s cruising speed (was 0.65)
        v_max=2.0,
        w_max=4.0,
        accel_max=2.5,            # Faster acceleration
    )

    # Logging structures
    t_list, gt_list, ekf_list, odom_list = [], [], [], []
    cross_track_list, alpha_list = [], []
    v_cmd_list, w_cmd_list = [], []

    dt = model.opt.timestep

    print("=" * 70)
    print(" J100 Pure Pursuit Path Tracking Stack Initialized")
    print(f" - Algorithm: Geometric Pure Pursuit (L_d = {controller.L_d}m, v_target = {controller.v_target}m/s)")
    print(" - State Estimation: EKF Fusing Noisy Wheel Encoders & IMU Gyro")
    print(" - EKF Update: Gyro-bias correction via encoder yaw-rate measurement")
    print(" - Reference Path: Figure-8 Waypoint Loop (600 waypoints)")
    print("=" * 70)
    print("Starting simulation... (Close the viewer window to view analysis plots)")

    # Extra log for EKF bias tracking
    bias_list = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            t = data.time

            # A. Read Noisy Sensors
            meas = sensors.read_sensors(data, dt)

            # B. Real-Time State Estimation (EKF — predict then update)
            odom_pose = odom.update(meas['encoder_vel'], dt)
            ekf.predict(meas['encoder_vel'], meas['gyro'][2], dt)
            ekf.update(meas['encoder_vel'], meas['gyro'][2])   # corrects b_g

            # C. Pure Pursuit Geometric Lookahead Point Search
            target_pt, closest_idx, cross_track_d = path.find_lookahead_point(ekf.pose[0:2], lookahead_dist=controller.L_d)

            # D. Compute Pure Pursuit Control Command
            v_cmd, w_cmd, err_info = controller.compute_control(ekf.pose, target_pt, dt)

            # E. Send Velocity Commands to Actuators
            set_actuator_vels(data, v_cmd, w_cmd)

            # F. Step Physics
            mujoco.mj_step(model, data)

            # G. Log Data (~100Hz)
            if int(t / dt) % 5 == 0:
                if int(t / dt) % 500 == 0:
                    print(f"t={data.time:.1f}s | "
                          f"Pos: ({ekf.pose[0]:.2f}, {ekf.pose[1]:.2f}) "
                          f"v: {v_cmd:.2f} w: {w_cmd:.2f} "
                          f"b_g: {ekf.gyro_bias*1000:.1f} mrad/s", flush=True)
                t_list.append(t)
                gt_list.append(meas['gt_pose'])
                ekf_list.append(ekf.pose.copy())
                odom_list.append(odom_pose.copy())
                cross_track_list.append(cross_track_d)
                alpha_list.append(err_info['alpha'])
                v_cmd_list.append(v_cmd)
                w_cmd_list.append(w_cmd)
                bias_list.append(ekf.gyro_bias)

            # H. Camera Tracking
            viewer.cam.lookat[:] = data.xpos[body_id]
            viewer.sync()

            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Process and print performance metrics
    if len(t_list) > 0:
        t_arr          = np.array(t_list)
        gt_arr         = np.array(gt_list)
        ekf_arr        = np.array(ekf_list)
        odom_arr       = np.array(odom_list)
        cross_track_arr = np.array(cross_track_list)
        alpha_arr      = np.array(alpha_list)
        v_cmds         = np.array(v_cmd_list)
        w_cmds         = np.array(w_cmd_list)
        bias_arr       = np.array(bias_list)

        ekf_pos_err = np.hypot(ekf_arr[:, 0] - gt_arr[:, 0], ekf_arr[:, 1] - gt_arr[:, 1])
        ekf_th_err  = np.abs(np.rad2deg(wrap_angle(ekf_arr[:, 2] - gt_arr[:, 2])))

        print("\n" + "=" * 70)
        print(" PURE PURSUIT PERFORMANCE REPORT")
        print("=" * 70)
        print(f" Total Simulation Time:      {t_arr[-1]:.2f} s")
        print(f" Mean Cross-Track Error:     {np.mean(cross_track_arr)*100.0:.2f} cm (Max: {np.max(cross_track_arr)*100.0:.2f} cm)")
        print(f" Cross-Track Error RMSE:     {np.sqrt(np.mean(cross_track_arr**2))*100.0:.2f} cm")
        print(f" EKF Position Error:         Mean = {np.mean(ekf_pos_err)*100.0:.2f} cm,  Max = {np.max(ekf_pos_err)*100.0:.2f} cm")
        print(f" EKF Heading Error:          Mean = {np.mean(ekf_th_err):.2f} deg")
        print(f" EKF Gyro Bias Estimate:     Final = {bias_arr[-1]*1000.0:.3f} mrad/s  (|Mean| = {np.abs(bias_arr).mean()*1000.0:.3f} mrad/s)")
        print("=" * 70)

        plot_pure_pursuit_analysis(
            t_arr, waypoints, gt_arr, ekf_arr, odom_arr,
            cross_track_arr, alpha_arr, v_cmds, w_cmds, bias_arr
        )


if __name__ == '__main__':
    main()
