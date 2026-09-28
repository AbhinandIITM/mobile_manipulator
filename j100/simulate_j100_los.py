import argparse
import os
import time
import cv2
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from los_guidance import LOSGuidanceSystem, VisualDetector
from sensors import SensorManager


def plot_los_analysis(
    t_list,
    robot_pos_list,
    target_pos_list,
    los_angle_list,
    range_list,
    v_cmd_list,
    w_cmd_list,
    u_list,
    v_list,
    save_path="j100_los_guidance_analysis.png",
):
    """
    Generates and saves a 6-panel Line of Sight (LOS) Guidance Analysis Plot.
    """
    t = np.array(t_list)
    robot_pos = np.array(robot_pos_list)
    target_pos = np.array(target_pos_list)
    los_deg = np.degrees(np.array(los_angle_list))
    ranges = np.array(range_list)
    v_cmds = np.array(v_cmd_list)
    w_cmds = np.array(w_cmd_list)
    us = np.array(u_list)
    vs = np.array(v_list)

    fig = plt.figure(figsize=(18, 10))
    plt.suptitle("J100 Mobile Robot — Vision + IMU Line-of-Sight (LOS) Guidance Analysis", fontsize=16, fontweight="bold")

    # 1. 2D Top-down Arena Trajectory
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(robot_pos[:, 0], robot_pos[:, 1], "b-", linewidth=2.0, label="Robot Trajectory")
    ax1.plot(target_pos[:, 0], target_pos[:, 1], "r--", linewidth=1.5, label="Target Beacon Path")
    ax1.scatter([robot_pos[0, 0]], [robot_pos[0, 1]], color="g", s=80, marker="o", label="Robot Start", zorder=5)
    ax1.scatter([target_pos[-1, 0]], [target_pos[-1, 1]], color="r", s=100, marker="*", label="Target Final", zorder=5)
    ax1.set_title("2D Top-Down Arena Interception Path", fontweight="bold")
    ax1.set_xlabel("X Position [m]")
    ax1.set_ylabel("Y Position [m]")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.axis("equal")

    # 2. Line of Sight Bearing Angle (deg)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(t, los_deg, "c-", linewidth=1.5, label="LOS Azimuth Bearing λ [deg]")
    ax2.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax2.set_title("Line of Sight (LOS) Bearing Angle (Robot Frame)", fontweight="bold")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("LOS Angle λ [deg]")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Range / Distance to Target Beacon
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t, ranges, "m-", linewidth=1.5, label="Distance to Target R [m]")
    ax3.axhline(0.8, color="r", linestyle="--", alpha=0.7, label="Stop Threshold (0.8m)")
    ax3.set_title("Range to Target Beacon vs Time", fontweight="bold")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Range [m]")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Guidance Commands (v_cmd, w_cmd)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(t, v_cmds, "b-", linewidth=1.5, label="Forward Velocity v_cmd [m/s]")
    ax4.plot(t, w_cmds, "r--", linewidth=1.2, label="Steering Yaw Rate ω_cmd [rad/s]")
    ax4.set_title("LOS Guidance Velocity Commands (v, ω)", fontweight="bold")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Command Magnitude")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, linestyle="--", alpha=0.5)

    # 5. Camera Image Plane Pixel Coordinates (u, v)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t, us, "g-", label="Centroid u (Horizontal)", linewidth=1.5)
    ax5.axhline(160, color="k", linestyle="--", alpha=0.5, label="Image Center (cx=160)")
    ax5.set_title("Visual Target Centroid in Camera Frame", fontweight="bold")
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Pixel X Coordinate [px]")
    ax5.legend(loc="upper right", fontsize=8)
    ax5.grid(True, linestyle="--", alpha=0.5)

    # 6. Interception Summary Metric
    ax6 = fig.add_subplot(2, 3, 6)
    euclidean_dist = np.linalg.norm(robot_pos - target_pos, axis=1)
    ax6.plot(t, euclidean_dist, "purple", linewidth=1.8, label="Ground Truth Distance [m]")
    ax6.axhline(np.min(euclidean_dist), color="k", linestyle="--", label=f"Min Distance: {np.min(euclidean_dist):.2f}m")
    ax6.set_title("True Interception Distance", fontweight="bold")
    ax6.set_xlabel("Time [s]")
    ax6.set_ylabel("Distance [m]")
    ax6.legend(loc="upper right", fontsize=8)
    ax6.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Plot Saved] Analysis plot successfully saved to: {os.path.abspath(save_path)}")


def main():
    parser = argparse.ArgumentParser(description="J100 Line-of-Sight (LOS) Vision + IMU Guidance")
    parser.add_argument(
        "--scenario",
        type=str,
        default="waypoints",
        choices=["static", "moving", "waypoints"],
        help="Target scenario: 'static', 'moving', or 'waypoints'",
    )
    parser.add_argument("--v_max", type=float, default=2.0, help="Max cruising forward velocity (m/s)")
    parser.add_argument("--d_stop", type=float, default=None, help="Target standoff distance in meters (default: 0.80m)")
    parser.add_argument("--duration", type=float, default=None, help="Max simulation duration in seconds (optional)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI viewers")
    parser.add_argument("--no_cv", action="store_true", help="Disable live OpenCV HUD window")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "j100_los_scene.xml")

    print(f"Loading J100 LOS Model from: {model_path} ...")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    beacon_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_beacon")
    mocap_id = model.body_mocapid[beacon_id] if beacon_id >= 0 else -1

    # Initial heading facing +X (Euler 0 0 pi since robot -X is forward)
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

    # Target convergence distance
    if args.d_stop is not None:
        target_dist = args.d_stop
    elif args.scenario == "moving":
        target_dist = 0.80   # Default 0.80m standoff for moving targets
    elif args.scenario == "waypoints":
        target_dist = 0.50   # 0.50m waypoint standoff
    else:
        target_dist = 0.80   # 0.80m static target docking

    guidance = LOSGuidanceSystem(
        v_max=args.v_max,
        w_max=4.0,
        target_distance=target_dist,
        kp_range=2.0,
        ki_range=1.0,
        kp_los=3.5,
        ki_los=1.8,
        kd_gyro=0.30,
        track_width=0.368,
        slip_factor=0.30,
    )

    # Offscreen Camera Renderer for 30 Hz Vision Pipeline
    renderer = mujoco.Renderer(model, height=240, width=320)

    # Target Waypoints for 'waypoints' scenario
    waypoint_list = [
        np.array([5.0, 2.5, 0.3]),
        np.array([8.0, -3.0, 0.3]),
        np.array([2.0, -4.5, 0.3]),
        np.array([0.0, 0.0, 0.3]),
    ]
    current_wp_idx = 0

    print("=" * 70)
    print(f" J100 Line-of-Sight (LOS) Guidance Initialized [{args.scenario.upper()}]")
    print(" - Perception: 30 Hz RGB-D Centroid & Range Detection (front_cam)")
    print(" - State Estimation: 500 Hz IMU Gyro High-Rate LOS Angle Kinematic Propagation")
    print(" - OpenCV HUD: Live RGB + Colorized Depth Map feed with Tracking Overlays")
    print(" - Guidance Law: Proportional LOS Navigation + Rate Damping + Range-Adaptive Speed")
    print(" - Close the MuJoCo viewer window to view performance metrics.")
    print("=" * 70)

    # Telemetry logging
    t_list, robot_pos_list, target_pos_list = [], [], []
    los_angle_list, range_list, v_cmd_list, w_cmd_list = [], [], [], []
    u_list, v_list = [], []

    dt = model.opt.timestep
    vision_dt = 1.0 / 30.0  # 30 Hz vision update
    last_vision_time = -1.0

    current_u = 160.0
    current_v = 120.0
    latest_detection = {"detected": False, "lambda_vis": 0.0, "range_vis": 0.0, "u": 160.0, "v": 120.0, "bbox": None}
    latest_telem = {"mode": "INIT", "los_angle": 0.0, "target_range": 5.0, "v_cmd": 0.0, "w_cmd": 0.0}

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

            # A. Update Dynamic Target Beacon Position
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
                        
                        # Compute shortest-arc bearing in robot frame
                        dx = next_wp[0] - data.xpos[body_id][0]
                        dy = next_wp[1] - data.xpos[body_id][1]
                        target_azimuth = np.arctan2(dy, dx)
                        meas_tmp = sensors.read_sensors(data, dt)
                        robot_heading = meas_tmp["gt_pose"][2]
                        bearing_prior = (target_azimuth - robot_heading + np.pi) % (2.0 * np.pi) - np.pi
                        
                        turn_name = "RIGHT" if bearing_prior < 0 else "LEFT"
                        print(f"\nt={t:.1f}s | >>> REACHED WAYPOINT #{current_wp_idx}! Next Beacon: {next_wp[:2]} | Shortest Turn: {turn_name} ({np.degrees(bearing_prior):+.1f}°)\n", flush=True)
                        guidance.set_target_bearing_prior(bearing_prior, approx_range=np.hypot(dx, dy))
                data.mocap_pos[mocap_id] = beacon_pos

            # B. 30 Hz Vision Pipeline (Offscreen RGB-D Render + OpenCV HUD)
            if (t - last_vision_time) >= vision_dt:
                last_vision_time = t

                # 1. Render RGB
                renderer.disable_depth_rendering()
                renderer.update_scene(data, camera="front_cam")
                rgb_frame = renderer.render()

                # 2. Render Depth
                renderer.enable_depth_rendering()
                renderer.update_scene(data, camera="front_cam")
                depth_frame = renderer.render()

                # 3. Detect visual target
                latest_detection = detector.detect(rgb_frame, depth_frame)
                guidance.update_vision(latest_detection, t)

                if latest_detection["detected"]:
                    current_u = latest_detection["u"]
                    current_v = latest_detection["v"]

                # 4. Display live OpenCV HUD window
                if not args.no_cv:
                    hud_frame = detector.build_hud_frame(rgb_frame, depth_frame, latest_detection, latest_telem)
                    cv2.imshow("J100 Onboard Vision Feed (RGB + Depth)", hud_frame)
                    cv2.waitKey(1)

            # C. 500 Hz High-Rate Sensor & IMU Guidance Loop
            meas = sensors.read_sensors(data, dt)
            gyro_z = float(meas["gyro"][2])

            v_cmd, w_cmd, latest_telem = guidance.update_imu_and_control(gyro_z, dt, t)

            # D. Send Commands to Skid-Steer Actuators
            actuator_vels = guidance.compute_actuator_velocities(v_cmd, w_cmd)
            data.ctrl[0] = actuator_vels[0]  # FL
            data.ctrl[1] = actuator_vels[1]  # FR
            data.ctrl[2] = actuator_vels[2]  # RR
            data.ctrl[3] = actuator_vels[3]  # RL

            # E. Step Physics
            mujoco.mj_step(model, data)

            # F. Camera Tracking in Viewer
            viewer.cam.lookat[:] = data.xpos[body_id]
            viewer.sync()

            # G. Log Telemetry (~100 Hz)
            if int(t / dt) % 5 == 0:
                t_list.append(t)
                robot_pos_list.append(data.xpos[body_id][:2].copy())
                target_pos_list.append(data.xpos[beacon_id][:2].copy())
                los_angle_list.append(latest_telem["los_angle"])
                range_list.append(latest_telem["target_range"])
                v_cmd_list.append(v_cmd)
                w_cmd_list.append(w_cmd)
                u_list.append(current_u)
                v_list.append(current_v)

                if int(t / dt) % 500 == 0:
                    print(
                        f"t={t:5.1f}s | Mode: {latest_telem['mode']:9s} | "
                        f"LOS Angle: {np.degrees(latest_telem['los_angle']):5.1f}° | "
                        f"Distance: {latest_telem['target_range']:4.2f}m | "
                        f"v_cmd: {v_cmd:4.2f}m/s | w_cmd: {w_cmd:4.2f}rad/s",
                        flush=True,
                    )

            # Keep real-time pace when viewing
            if not args.headless:
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    # Clean up OpenCV windows
    if not args.no_cv:
        cv2.destroyAllWindows()

    # 5. Performance Report & Analysis Plots
    if len(t_list) > 10:
        robot_pos = np.array(robot_pos_list)
        target_pos = np.array(target_pos_list)
        dists = np.linalg.norm(robot_pos - target_pos, axis=1)

        print("\n" + "=" * 70)
        print(f" J100 LOS Vision + IMU Guidance Performance Summary [{args.scenario.upper()}]")
        print("=" * 70)
        print(f"Total Simulation Time:        {t_list[-1]:.2f} s")
        print(f"Initial Distance to Target:   {dists[0]:.2f} m")
        print(f"Final Interception Distance:  {dists[-1]:.2f} m")
        print(f"Minimum Distance Achieved:    {np.min(dists):.2f} m")
        print(f"Mean LOS Bearing Alignment:   {np.mean(np.abs(np.degrees(los_angle_list))):.2f}°")
        print("=" * 70)

        plot_path = os.path.join(script_dir, "j100_los_guidance_analysis.png")
        plot_los_analysis(
            t_list=t_list,
            robot_pos_list=robot_pos_list,
            target_pos_list=target_pos_list,
            los_angle_list=los_angle_list,
            range_list=range_list,
            v_cmd_list=v_cmd_list,
            w_cmd_list=w_cmd_list,
            u_list=u_list,
            v_list=v_list,
            save_path=plot_path,
        )


if __name__ == "__main__":
    main()
