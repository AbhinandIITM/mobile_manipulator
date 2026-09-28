"""
simulate_j100_clf_cbf.py - Simulation and Validation of CLF-CBF Guidance Stack for J100 Mobile Robot in MuJoCo.

Validates:
1. Zero Distance Overshoot on Static Target Docking (enforced by Control Barrier Function h(R) >= 0).
2. Elimination of Chassis Pitch/Wheelie Bounce (enforced by Acceleration Slew Limits).
3. Exponential Bearing Convergence and Smooth Velocity Profile.
4. Generates 6-Panel Performance Analysis Plot: j100_clf_cbf_analysis.png.
"""

import argparse
import os
import time
import cv2
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from clf_cbf_guidance import CLFCBFGuidanceSystem, VisualDetector
from sensors import SensorManager


def plot_clf_cbf_analysis(
    t_list,
    robot_pos_list,
    target_pos_list,
    pitch_list,
    range_list,
    v_cmd_list,
    w_cmd_list,
    cbf_h_list,
    clf_v_list,
    target_distance=0.80,
    save_path="j100_clf_cbf_analysis.png",
):
    """
    Generates a 6-panel rigorous Control Theoretical & Dynamic Analysis Plot:
    1. 2D Trajectory Interception Path.
    2. Range vs Time & Zero-Overshoot Barrier.
    3. Chassis Pitch Angle (Anti-Lift / Wheelie Verification).
    4. Velocity & Angular Rate Commands.
    5. Control Barrier Function (CBF) h(t) >= 0 invariant safety check.
    6. Control Lyapunov Function (CLF) V(t) exponential convergence.
    """
    t = np.array(t_list)
    robot_pos = np.array(robot_pos_list)
    target_pos = np.array(target_pos_list)
    pitches_deg = np.degrees(np.array(pitch_list))
    ranges = np.array(range_list)
    v_cmds = np.array(v_cmd_list)
    w_cmds = np.array(w_cmd_list)
    hs = np.array(cbf_h_list)
    vs = np.array(clf_v_list)

    fig = plt.figure(figsize=(18, 10))
    plt.suptitle("J100 Mobile Base — CLF-CBF Quadratic Programming Guidance Analysis", fontsize=16, fontweight="bold")

    # 1. 2D Top-down Arena Trajectory
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(robot_pos[:, 0], robot_pos[:, 1], "b-", linewidth=2.0, label="Robot Path")
    ax1.plot(target_pos[:, 0], target_pos[:, 1], "r--", linewidth=1.5, label="Target Beacon")
    ax1.scatter([robot_pos[0, 0]], [robot_pos[0, 1]], color="g", s=80, marker="o", label="Start", zorder=5)
    ax1.scatter([target_pos[-1, 0]], [target_pos[-1, 1]], color="r", s=100, marker="*", label="Target", zorder=5)
    ax1.set_title("2D Interception Trajectory", fontweight="bold")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.axis("equal")

    # 2. Range to Target vs Distance Barrier
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(t, ranges, "b-", linewidth=1.8, label="Estimated Range R(t)")
    ax2.axhline(target_distance, color="r", linestyle="--", linewidth=1.8, label=f"Target Barrier ({target_distance:.2f}m)")
    ax2.set_title("Range to Target R(t) [Zero Overshoot]", fontweight="bold")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Distance [m]")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Chassis Pitch Angle (Pitch Bounce / Lift Suppression)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t, pitches_deg, "m-", linewidth=1.5, label="Chassis Pitch θ_pitch [deg]")
    ax3.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax3.set_title("Chassis Pitch Angle (Lift/Wheelie Suppression)", fontweight="bold")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Pitch Angle [deg]")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Velocity Commands
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(t, v_cmds, "g-", linewidth=1.6, label="v_cmd [m/s]")
    ax4.plot(t, w_cmds, "orange", linestyle="--", linewidth=1.2, label="ω_cmd [rad/s]")
    ax4.set_title("CLF-CBF QP Control Inputs (v, ω)", fontweight="bold")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Magnitude")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, linestyle="--", alpha=0.5)

    # 5. Control Barrier Function h(t)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t, hs, "r-", linewidth=1.8, label="Barrier h(t) = R - d_target")
    ax5.axhline(0.0, color="k", linestyle="--", linewidth=1.5, label="Safety Boundary (h=0)")
    ax5.fill_between(t, hs, 0, where=(hs >= 0), color="green", alpha=0.15, label="Safe Invariant Set")
    ax5.fill_between(t, hs, 0, where=(hs < 0), color="red", alpha=0.3, label="Unsafe Invariant Set")
    ax5.set_title("Control Barrier Function h(t) >= 0", fontweight="bold")
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("h(t)")
    ax5.legend(loc="upper right", fontsize=8)
    ax5.grid(True, linestyle="--", alpha=0.5)

    # 6. Control Lyapunov Function V(t)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(t, vs, "purple", linewidth=1.8, label="Lyapunov Energy V(t)")
    ax6.set_title("Control Lyapunov Function V(t) Convergence", fontweight="bold")
    ax6.set_xlabel("Time [s]")
    ax6.set_ylabel("V(t)")
    ax6.legend(loc="upper right", fontsize=8)
    ax6.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Plot Saved] CLF-CBF Analysis plot successfully saved to: {os.path.abspath(save_path)}")


def main():
    parser = argparse.ArgumentParser(description="J100 CLF-CBF QP Guidance Simulation")
    parser.add_argument(
        "--scenario",
        type=str,
        default="static",
        choices=["static", "moving", "waypoints"],
        help="Target scenario: 'static', 'moving', or 'waypoints'",
    )
    parser.add_argument("--v_max", type=float, default=1.6, help="Max forward velocity (m/s)")
    parser.add_argument("--d_stop", type=float, default=0.80, help="Target standoff distance (m)")
    parser.add_argument("--duration", type=float, default=15.0, help="Simulation duration (s)")
    parser.add_argument("--headless", action="store_true", help="Run in fast headless mode")
    parser.add_argument("--no_cv", action="store_true", help="Disable live OpenCV HUD")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "j100_los_scene.xml")

    print(f"Loading J100 Model from: {model_path} ...")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    beacon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_beacon")
    mocap_id = model.body_mocapid[beacon_id] if beacon_id >= 0 else -1

    # Initial robot pose
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = 0.08
    data.qpos[3] = 0.0
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = 1.0

    mujoco.mj_forward(model, data)

    sensors = SensorManager(model)
    detector = VisualDetector(width=320, height=240, fovy_deg=60.0)

    guidance = CLFCBFGuidanceSystem(
        target_distance=args.d_stop,
        v_max=args.v_max,
        w_max=3.5,
        a_max_accel=1.0,      # Smooth 1.0 m/s^2 forward acceleration (suppresses pitch lift)
        a_max_decel=2.0,      # Smooth braking deceleration
        alpha_max_accel=3.5,
        k_clf_range=2.0,
        kp_los=3.5,
        ki_los=1.8,
        kd_gyro=0.30,
        gamma_clf=1.5,
        gamma_cbf=1.2,
        slack_penalty=1e4,
    )

    renderer = mujoco.Renderer(model, height=240, width=320)

    waypoint_list = [
        np.array([5.0, 2.5, 0.3]),
        np.array([8.0, -3.0, 0.3]),
        np.array([2.0, -4.5, 0.3]),
        np.array([0.0, 0.0, 0.3]),
    ]
    current_wp_idx = 0

    print("=" * 70)
    print(f" J100 CLF-CBF Guidance Stack Initialized [{args.scenario.upper()}]")
    print(" - Safety Barrier: Control Barrier Function h(R) = R - d_target >= 0 (Zero Overshoot)")
    print(" - Stability: Control Lyapunov Function V(R, lambda) Exponential Convergence")
    print(" - Pitch Protection: Dynamic Acceleration Slew Envelope (Smooth 1.0 m/s^2 ramp)")
    print(" - Perception: 30 Hz RGB-D Vision + 500 Hz IMU Kinematic State Propagation")
    print("=" * 70)

    t_list, robot_pos_list, target_pos_list, pitch_list = [], [], [], []
    range_list, v_cmd_list, w_cmd_list, cbf_h_list, clf_v_list = [], [], [], [], []

    dt = model.opt.timestep
    vision_dt = 1.0 / 30.0
    last_vision_time = -1.0

    latest_detection = {"detected": False, "lambda_vis": 0.0, "range_vis": 0.0, "u": 160.0, "v": 120.0, "bbox": None}
    latest_telem = {"mode": "INIT", "los_angle": 0.0, "target_range": 5.0, "cbf_h": 0.0, "clf_v": 0.0, "v_cmd": 0.0, "w_cmd": 0.0}

    class NullViewer:
        def __init__(self):
            self.cam = type("Cam", (), {"distance": 0, "elevation": 0, "azimuth": 0, "lookat": [0, 0, 0]})()
        def is_running(self):
            return True
        def sync(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    max_duration = args.duration if args.duration is not None else float("inf")
    viewer_ctx = NullViewer() if args.headless else mujoco.viewer.launch_passive(model, data)

    with viewer_ctx as viewer:
        if not args.headless:
            viewer.cam.distance = 9.0
            viewer.cam.elevation = -40.0
            viewer.cam.azimuth = 135.0

        while viewer.is_running() and data.time < max_duration:
            step_start = time.time()
            t = data.time

            # A. Update Target Position
            if mocap_id >= 0:
                if args.scenario == "static":
                    beacon_pos = np.array([6.0, 3.0, 0.3])
                elif args.scenario == "moving":
                    beacon_pos = np.array([4.0 + 3.0 * np.cos(0.3 * t), 3.0 * np.sin(0.3 * t), 0.3])
                elif args.scenario == "waypoints":
                    target_wp = waypoint_list[current_wp_idx]
                    beacon_pos = target_wp
                    dist_to_wp = np.linalg.norm(data.xpos[body_id][:2] - target_wp[:2])
                    if dist_to_wp < 1.5:
                        current_wp_idx = (current_wp_idx + 1) % len(waypoint_list)
                        next_wp = waypoint_list[current_wp_idx]
                        dx = next_wp[0] - data.xpos[body_id][0]
                        dy = next_wp[1] - data.xpos[body_id][1]
                        target_azimuth = np.arctan2(dy, dx)
                        meas_tmp = sensors.read_sensors(data, dt)
                        robot_heading = meas_tmp["gt_pose"][2]
                        bearing_prior = (target_azimuth - robot_heading + np.pi) % (2.0 * np.pi) - np.pi
                        guidance.set_target_bearing_prior(bearing_prior, approx_range=np.hypot(dx, dy))
                data.mocap_pos[mocap_id] = beacon_pos

            # B. 30 Hz Vision Pipeline
            if (t - last_vision_time) >= vision_dt:
                last_vision_time = t

                renderer.disable_depth_rendering()
                renderer.update_scene(data, camera="front_cam")
                rgb_frame = renderer.render()

                renderer.enable_depth_rendering()
                renderer.update_scene(data, camera="front_cam")
                depth_frame = renderer.render()

                latest_detection = detector.detect(rgb_frame, depth_frame)
                guidance.update_vision(latest_detection, t)

                if not args.no_cv and not args.headless:
                    hud_frame = detector.build_hud_frame(rgb_frame, depth_frame, latest_detection, latest_telem)
                    cv2.imshow("J100 CLF-CBF Guidance HUD", hud_frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            # C. 500 Hz IMU & CLF-CBF Guidance QP Step
            sensor_data = sensors.read_sensors(data, dt)
            gyro_z = sensor_data["gyro"][2]

            v_cmd, w_cmd, latest_telem = guidance.update_imu_and_control(gyro_z, dt, t)

            # D. Actuation
            actuator_vels = guidance.compute_actuator_velocities(v_cmd, w_cmd)
            data.ctrl[0] = actuator_vels[0]  # FL
            data.ctrl[1] = actuator_vels[1]  # FR
            data.ctrl[2] = actuator_vels[2]  # RR
            data.ctrl[3] = actuator_vels[3]  # RL

            # E. Step Physics
            mujoco.mj_step(model, data)

            if not args.headless:
                viewer.cam.lookat[:] = data.xpos[body_id]
                viewer.sync()

            # F. Pitch angle extraction from base orientation matrix
            R_mat = data.xmat[body_id].reshape(3, 3)
            # Body forward is -R[:, 0], Level vertical is R[:, 2]
            pitch_angle = np.arcsin(np.clip(-R_mat[2, 0], -1.0, 1.0))

            # G. Log Telemetry (~100 Hz)
            if int(t / dt) % 5 == 0:
                t_list.append(t)
                robot_pos_list.append(data.xpos[body_id][:2].copy())
                target_pos_list.append(data.xpos[beacon_id][:2].copy())
                pitch_list.append(pitch_angle)
                range_list.append(latest_telem["target_range"])
                v_cmd_list.append(v_cmd)
                w_cmd_list.append(w_cmd)
                cbf_h_list.append(latest_telem.get("cbf_h", 0.0))
                clf_v_list.append(latest_telem.get("clf_v", 0.0))

                if int(t / dt) % 500 == 0:
                    print(
                        f"t={t:5.1f}s | Mode: {latest_telem['mode']:10s} | "
                        f"Distance: {latest_telem['target_range']:4.2f}m | "
                        f"CBF h: {latest_telem.get('cbf_h', 0.0):+4.2f} | "
                        f"Pitch: {np.degrees(pitch_angle):+4.1f} deg | "
                        f"v_cmd: {v_cmd:4.2f}m/s | w_cmd: {w_cmd:4.2f}rad/s",
                        flush=True,
                    )

            if not args.headless:
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    if not args.no_cv and not args.headless:
        cv2.destroyAllWindows()

    # 5. Performance Report & Analysis Plots
    if len(t_list) > 10:
        robot_pos = np.array(robot_pos_list)
        target_pos = np.array(target_pos_list)
        dists = np.linalg.norm(robot_pos - target_pos, axis=1)
        pitches = np.degrees(np.array(pitch_list))
        ranges_arr = np.array(range_list)

        min_range = np.min(ranges_arr[t > 1.0]) if np.any(t > 1.0) else np.min(ranges_arr)
        max_pitch_mag = np.max(np.abs(pitches))

        print("\n" + "=" * 70)
        print(f" J100 CLF-CBF Guidance Performance Summary [{args.scenario.upper()}]")
        print("=" * 70)
        print(f"Total Simulation Time:         {t_list[-1]:.2f} s")
        print(f"Target Standoff Distance (CBF): {args.d_stop:.2f} m")
        print(f"Final Estimated Range R:       {ranges_arr[-1]:.2f} m")
        print(f"Minimum Estimated Range R:     {min_range:.2f} m (Barrier Invariance: {'PASSED [R >= d_target]' if min_range >= args.d_stop - 0.02 else 'VIOLATED'})")
        print(f"Max Pitch Oscillation:         {max_pitch_mag:.2f} deg (Anti-Wheelie: {'PASSED (< 2.5 deg)' if max_pitch_mag < 2.5 else 'HIGH PITCH'})")
        print(f"Final Interception Distance:   {dists[-1]:.2f} m")
        print("=" * 70)

        plot_path = os.path.join(script_dir, "j100_clf_cbf_analysis.png")
        plot_clf_cbf_analysis(
            t_list=t_list,
            robot_pos_list=robot_pos_list,
            target_pos_list=target_pos_list,
            pitch_list=pitch_list,
            range_list=range_list,
            v_cmd_list=v_cmd_list,
            w_cmd_list=w_cmd_list,
            cbf_h_list=cbf_h_list,
            clf_v_list=clf_v_list,
            target_distance=args.d_stop,
            save_path=plot_path,
        )


if __name__ == "__main__":
    main()
