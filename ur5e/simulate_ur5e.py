import argparse
import os
import time
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from controllers import CartesianIKController
from trajectories import TrajectoryGenerator


def plot_ur5e_analysis(
    t_list,
    target_pos_list,
    eef_pos_list,
    pos_err_list,
    ori_err_list,
    q_list,
    qdot_list,
    traj_name="figure8",
    save_path="ur5e_trajectory_tracking_analysis.png",
):
    """
    Generates and saves a publication-quality 6-panel trajectory tracking analysis plot.
    """
    t = np.array(t_list)
    target_pos = np.array(target_pos_list)
    eef_pos = np.array(eef_pos_list)
    pos_err_norm = np.array(pos_err_list) * 100.0  # convert to cm
    ori_err_norm = np.degrees(np.array(ori_err_list))
    q = np.degrees(np.array(q_list))
    qdot = np.degrees(np.array(qdot_list))

    fig = plt.figure(figsize=(18, 10))
    plt.suptitle(f"UR5e 6-DOF Cartesian Target Trajectory Tracking [{traj_name.upper()}]", fontsize=16, fontweight="bold")

    # 1. 3D Workspace Path
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    ax1.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2], "r--", label="Target Trajectory", linewidth=2.0)
    ax1.plot(eef_pos[:, 0], eef_pos[:, 1], eef_pos[:, 2], "b-", label="Actual End-Effector", linewidth=1.5, alpha=0.85)
    ax1.scatter([target_pos[0, 0]], [target_pos[0, 1]], [target_pos[0, 2]], color="g", s=60, label="Start")
    ax1.set_title("3D End-Effector Workspace Path", fontweight="bold")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.set_zlabel("Z [m]")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. XYZ Coordinates vs Time
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(t, target_pos[:, 0], "r--", label="Target X", linewidth=1.5)
    ax2.plot(t, eef_pos[:, 0], "r-", label="Actual X", alpha=0.8)
    ax2.plot(t, target_pos[:, 1], "g--", label="Target Y", linewidth=1.5)
    ax2.plot(t, eef_pos[:, 1], "g-", label="Actual Y", alpha=0.8)
    ax2.plot(t, target_pos[:, 2], "b--", label="Target Z", linewidth=1.5)
    ax2.plot(t, eef_pos[:, 2], "b-", label="Actual Z", alpha=0.8)
    ax2.set_title("Cartesian Coordinates (X, Y, Z) vs Time", fontweight="bold")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Position [m]")
    ax2.legend(loc="upper right", ncol=3, fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Position Error (cm)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t, pos_err_norm, "m-", linewidth=1.5, label="Position Error ||p_des - p||")
    mean_err = np.mean(pos_err_norm)
    max_err = np.max(pos_err_norm)
    ax3.axhline(mean_err, color="k", linestyle="--", alpha=0.7, label=f"Mean: {mean_err:.2f} cm")
    ax3.set_title("Cartesian Tracking Error (Euclidean)", fontweight="bold")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Error [cm]")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Orientation Tracking Error (deg)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(t, ori_err_norm, "c-", linewidth=1.5, label="Orientation Error")
    mean_ori = np.mean(ori_err_norm)
    ax4.axhline(mean_ori, color="k", linestyle="--", alpha=0.7, label=f"Mean: {mean_ori:.2f}°")
    ax4.set_title("End-Effector Orientation Error", fontweight="bold")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Error [deg]")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, linestyle="--", alpha=0.5)

    # 5. Joint Angles
    ax5 = fig.add_subplot(2, 3, 5)
    joint_names = ["Shoulder Pan", "Shoulder Lift", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3"]
    for i in range(6):
        ax5.plot(t, q[:, i], label=joint_names[i], linewidth=1.2)
    ax5.set_title("Joint Angles vs Time", fontweight="bold")
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Angle [deg]")
    ax5.legend(loc="upper right", ncol=2, fontsize=7)
    ax5.grid(True, linestyle="--", alpha=0.5)

    # 6. Joint Velocities
    ax6 = fig.add_subplot(2, 3, 6)
    for i in range(6):
        ax6.plot(t, qdot[:, i], label=joint_names[i], linewidth=1.2)
    ax6.set_title("Joint Velocities vs Time", fontweight="bold")
    ax6.set_xlabel("Time [s]")
    ax6.set_ylabel("Velocity [deg/s]")
    ax6.legend(loc="upper right", ncol=2, fontsize=7)
    ax6.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Plot Saved] Analysis plot successfully saved to: {os.path.abspath(save_path)}")


def render_trajectory_preview(viewer, path_points: np.ndarray, eef_trail: list = None):
    """
    Renders the target trajectory ribbon and live end-effector trail directly in MuJoCo viewer.
    """
    num_path_pts = len(path_points)
    trail_pts = eef_trail if eef_trail is not None else []
    total_geoms = num_path_pts + len(trail_pts)

    # Clamp to max capacity
    total_geoms = min(total_geoms, viewer.user_scn.maxgeom)
    viewer.user_scn.ngeom = total_geoms

    geom_idx = 0

    # 1. Render green reference path dots
    for i in range(min(num_path_pts, total_geoms)):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_idx],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.008, 0.008, 0.008]),
            path_points[i],
            np.eye(3).flatten(),
            np.array([0.1, 0.9, 0.2, 0.5]),  # Translucent green
        )
        geom_idx += 1

    # 2. Render blue actual end-effector trail dots
    remaining = total_geoms - geom_idx
    for i in range(min(len(trail_pts), remaining)):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_idx],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.006, 0.006, 0.006]),
            trail_pts[i],
            np.eye(3).flatten(),
            np.array([0.2, 0.4, 1.0, 0.7]),  # Blue trail
        )
        geom_idx += 1


def main():
    parser = argparse.ArgumentParser(description="UR5e Cartesian Target Trajectory Tracking")
    parser.add_argument(
        "--traj",
        type=str,
        default="figure8",
        choices=["figure8", "circle", "spiral", "waypoints"],
        help="Trajectory profile: 'figure8', 'circle', 'spiral', 'waypoints'",
    )
    parser.add_argument("--period", type=float, default=6.0, help="Period in seconds for one full trajectory loop")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "ur5e_scene.xml")

    print(f"Loading UR5e MuJoCo model from: {model_path} ...")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # 1. Reset to standard home keyframe
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        home_qpos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
        data.qpos[:6] = home_qpos
        data.ctrl[:6] = home_qpos

    mujoco.mj_forward(model, data)

    # 2. Get Site and Target Mocap IDs
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
    init_eef_pos = data.site_xpos[site_id].copy()

    # Target Mocap body for real-time visualization
    mocap_id = -1
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
        if body_id >= 0:
            mocap_id = model.body_mocapid[body_id]
    except Exception:
        pass

    # 3. Instantiate 3D Trajectory Generator and Closed-Loop IK Controller
    traj_gen = TrajectoryGenerator(
        traj_type=args.traj,
        center=np.array([-0.10, 0.45, 0.45]),
        scale=np.array([0.16, 0.12, 0.08]),
        period=args.period,
        lead_in_time=2.0,
    )

    controller = CartesianIKController(
        model=model,
        site_name="pinch",
        home_qpos=data.qpos[:6].copy(),
        kp_pos=20.0,       # Fast position convergence
        kp_ori=15.0,       # Fast orientation convergence
        k_null=2.0,        # Nullspace posture regulation
        damping=0.02,      # Singularity robustness
        max_qdot=3.14,     # Max joint speed limit (rad/s)
    )
    controller.reset(data.qpos[:6].copy())

    # Pre-sample path points for 3D visual preview in viewer
    preview_path_points = traj_gen.sample_path(num_points=240 if args.traj == "spiral" else 120)

    print("=" * 70)
    print(f" UR5e 6-DOF Cartesian Target Trajectory Tracking [{args.traj.upper()}]")
    print(" - Controller: Closed-Loop Inverse Kinematics (DLS + Nullspace)")
    print(" - Reference Trajectory: Green 3D Waypoint Ribbon in Viewer")
    print(" - Dynamic Target: Red Mocap Sphere in Viewer")
    print(" - Actual EEF Path: Blue Trail in Viewer")
    print(" - Close the MuJoCo viewer window to exit and generate plots.")
    print("=" * 70)

    # Telemetry logging lists
    t_list = []
    target_pos_list = []
    eef_pos_list = []
    pos_err_list = []
    ori_err_list = []
    q_list = []
    qdot_list = []
    eef_trail = []

    dt = model.opt.timestep

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Configure viewer camera
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -25.0
        viewer.cam.distance = 2.0
        viewer.cam.lookat[:] = [0.0, 0.3, 0.4]

        start_real_time = time.time()

        while viewer.is_running():
            step_start = time.time()
            t = data.time

            # A. Compute Desired Target Pose & Twist along Trajectory
            target_pos, target_vel, target_rot, target_angvel = traj_gen.get_target(t, init_eef_pos)

            # B. Update Target Mocap Marker in Viewer
            if mocap_id >= 0:
                data.mocap_pos[mocap_id] = target_pos

            # C. Compute Joint Control via DLS Differential IK
            q_cmd, diag_info = controller.compute_control(
                data=data,
                target_pos=target_pos,
                target_vel=target_vel,
                target_rot=target_rot,
                target_angvel=target_angvel,
                dt=dt,
            )

            # D. Send Commands to Actuators
            data.ctrl[:6] = q_cmd

            # E. Advance Simulation Step
            mujoco.mj_step(model, data)

            # F. Log Telemetry and Update Visual Trajectory Preview (~100 Hz)
            if int(t / dt) % 5 == 0:
                curr_pos = data.site_xpos[site_id].copy()
                curr_q = data.qpos[:6].copy()
                curr_qdot = data.qvel[:6].copy()

                t_list.append(t)
                target_pos_list.append(target_pos.copy())
                eef_pos_list.append(curr_pos.copy())
                pos_err_list.append(diag_info["pos_err_norm"])
                ori_err_list.append(diag_info["ori_err_norm"])
                q_list.append(curr_q)
                qdot_list.append(curr_qdot)

                # Keep a rolling trail of actual end-effector positions (last 400 points)
                if int(t / dt) % 20 == 0:
                    eef_trail.append(curr_pos.copy())
                    if len(eef_trail) > 400:
                        eef_trail.pop(0)

                # Render trajectory preview and live trail in viewer
                render_trajectory_preview(viewer, preview_path_points, eef_trail)

                if int(t / dt) % 500 == 0:
                    print(
                        f"t={t:5.1f}s | "
                        f"Target XYZ: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}) | "
                        f"EEF Error: {diag_info['pos_err_norm']*100:5.2f} cm | "
                        f"Ori Error: {np.degrees(diag_info['ori_err_norm']):4.2f}°",
                        flush=True,
                    )

            viewer.sync()

            # Keep real-time pace
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # 4. Generate Performance Report & Analysis Plots
    if len(t_list) > 10:
        pos_err_cm = np.array(pos_err_list) * 100.0
        ori_err_deg = np.degrees(np.array(ori_err_list))

        print("\n" + "=" * 70)
        print(f" UR5e Trajectory Tracking Performance Summary [{args.traj.upper()}]")
        print("=" * 70)
        print(f"Total Simulation Time:        {t_list[-1]:.2f} s")
        print(f"Mean Cartesian Pos Error:     {np.mean(pos_err_cm):.3f} cm ({np.mean(pos_err_cm)*10:.2f} mm)")
        print(f"Max Cartesian Pos Error:      {np.max(pos_err_cm):.3f} cm ({np.max(pos_err_cm)*10:.2f} mm)")
        print(f"Steady-State Mean Error:      {np.mean(pos_err_cm[int(len(pos_err_cm)*0.3):]):.3f} cm")
        print(f"Mean Orientation Error:       {np.mean(ori_err_deg):.3f}°")
        print(f"Max Orientation Error:        {np.max(ori_err_deg):.3f}°")
        print("=" * 70)

        plot_path = os.path.join(script_dir, "ur5e_trajectory_tracking_analysis.png")
        plot_ur5e_analysis(
            t_list=t_list,
            target_pos_list=target_pos_list,
            eef_pos_list=eef_pos_list,
            pos_err_list=pos_err_list,
            ori_err_list=ori_err_list,
            q_list=q_list,
            qdot_list=qdot_list,
            traj_name=args.traj,
            save_path=plot_path,
        )


if __name__ == "__main__":
    main()
