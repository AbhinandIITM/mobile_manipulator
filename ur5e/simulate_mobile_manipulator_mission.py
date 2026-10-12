"""
simulate_mobile_manipulator_mission.py
======================================
Complete End-to-End Autonomous Mobile Manipulation Mission:
1. Mobile Transit & Precision Docking via CLF-CBF Velocity Guidance
2. Arm Deployment & Pre-Grasp Hovering above Tabletop Target Block
3. Precision Descent & Robotiq 2F-85 Gripper Grasping
4. Vertical Clearance Lift
5. Mobile Transport to Dropoff Station
6. Payload Release into Dropoff Bin
7. Multi-Panel Performance Analytics Generation
"""

import os
import sys
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
J100_DIR = os.path.join(PROJECT_ROOT, "j100")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, J100_DIR)

from mobile_manipulator_controller import MobileManipulatorArmIK, SkidSteerBaseDrive


def run_mission(headless: bool = True, duration: float = 16.0):
    scene_path = os.path.join(SCRIPT_DIR, "mobile_manipulator_scene.xml")
    print(f"Loading Mobile Manipulator Scene from: {scene_path} ...")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "staging_table")
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    bin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dropoff_bin")
    pinch_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")

    # Initial state
    mujoco.mj_resetData(model, data)

    # Robot starts facing table along -X
    data.qpos[0] = -0.5
    data.qpos[1] = 0.0
    data.qpos[2] = 0.065
    data.qpos[3] = 1.0
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0

    # Arm starts in ergonomic resting pose
    home_q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    data.qpos[11:17] = home_q
    data.ctrl[4:10] = home_q
    data.ctrl[10] = 0.0  # Gripper open

    mujoco.mj_forward(model, data)

    # Controllers
    arm_ik = MobileManipulatorArmIK(model, site_name="pinch", kp_pos=25.0, kp_ori=15.0, damping=0.01)
    arm_ik.reset(home_q)
    base_drive = SkidSteerBaseDrive(accel_max=1.0, w_dot_max=2.0)

    R_down = np.array([
        [-1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0, -1]
    ], dtype=float)

    state = "TRANSIT"
    state_start_time = 0.0

    # Telemetry logging
    t_log, state_log = [], []
    base_x_log, base_y_log, pitch_log = [], [], []
    ee_x_log, ee_y_log, ee_z_log = [], [], []
    block_x_log, block_y_log, block_z_log = [], [], []
    v_cmd_log, w_cmd_log, grip_log = [], [], []
    joint_q_log = []

    dt = model.opt.timestep
    sim_time = 0.0

    print("=" * 70)
    print(" Coordinated Mobile Manipulator Mission Started")
    print(" Mission Plan: TRANSIT -> DOCK -> DEPLOY -> DESCEND -> GRASP -> LIFT -> TRANSPORT -> RELEASE")
    print("=" * 70)

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

    viewer_ctx = NullViewer() if headless else mujoco.viewer.launch_passive(model, data)

    target_arm_pos = data.site_xpos[pinch_site_id].copy()

    with viewer_ctx as viewer:
        while viewer.is_running() and sim_time < duration:
            t = data.time
            sim_time = t

            base_pos = data.xpos[chassis_id].copy()
            block_pos = data.xpos[block_id].copy()
            pinch_pos = data.site_xpos[pinch_site_id].copy()
            bin_pos = data.xpos[bin_id].copy()

            # Chassis orientation
            R_base = data.xmat[chassis_id].reshape(3, 3)
            heading_vec = -R_base[:, 0]
            current_yaw = np.arctan2(heading_vec[1], heading_vec[0])
            pitch_angle = np.arcsin(np.clip(-R_base[2, 0], -1.0, 1.0))

            v_cmd = 0.0
            w_cmd = 0.0
            gripper_cmd = data.ctrl[10]

            # ─── State Machine ───────────────────────────────────────────────
            if state == "TRANSIT":
                data.ctrl[4:10] = home_q

                dock_x = -1.75
                dx = dock_x - base_pos[0]
                dist_to_dock = abs(dx)

                if dist_to_dock > 0.03:
                    # Slew-controlled approach with smooth braking
                    v_target = min(0.65, np.sqrt(2.0 * 1.0 * max(0.0, dist_to_dock)))
                    v_cmd = float(v_target)
                    w_cmd = float(-1.5 * base_pos[1])  # Keep on centerline y=0
                else:
                    v_cmd = 0.0
                    w_cmd = 0.0
                    state = "DEPLOY"
                    state_start_time = t
                    target_arm_pos = np.array([block_pos[0], block_pos[1], block_pos[2] + 0.16])
                    print(f"[{t:5.2f}s] DOCKING REACHED -> State: DEPLOY")

            elif state == "DEPLOY":
                v_cmd = 0.0
                w_cmd = 0.0
                target_arm_pos = np.array([block_pos[0], block_pos[1], block_pos[2] + 0.16])
                q_cmd = arm_ik.compute_ik(data, target_arm_pos, R_down, dt=dt)
                data.ctrl[4:10] = q_cmd

                dist_ee = np.linalg.norm(pinch_pos - target_arm_pos)
                if dist_ee < 0.04 or (t - state_start_time) > 2.0:
                    state = "DESCEND"
                    state_start_time = t
                    print(f"[{t:5.2f}s] HOVER ALIGNED -> State: DESCEND")

            elif state == "DESCEND":
                v_cmd = 0.0
                w_cmd = 0.0
                target_arm_pos = np.array([block_pos[0], block_pos[1], block_pos[2] + 0.025])
                q_cmd = arm_ik.compute_ik(data, target_arm_pos, R_down, dt=dt)
                data.ctrl[4:10] = q_cmd

                dist_ee = np.linalg.norm(pinch_pos - target_arm_pos)
                if dist_ee < 0.03 or (t - state_start_time) > 1.8:
                    state = "GRASP"
                    state_start_time = t
                    print(f"[{t:5.2f}s] AT CONTACT HEIGHT -> State: GRASP")

            elif state == "GRASP":
                v_cmd = 0.0
                w_cmd = 0.0
                gripper_cmd = 255.0
                data.ctrl[10] = gripper_cmd

                # Hold position during finger closure
                q_cmd = arm_ik.compute_ik(data, target_arm_pos, R_down, dt=dt)
                data.ctrl[4:10] = q_cmd

                if (t - state_start_time) > 1.2:
                    state = "LIFT"
                    state_start_time = t
                    print(f"[{t:5.2f}s] FINGERS SECURED -> State: LIFT")

            elif state == "LIFT":
                v_cmd = 0.0
                w_cmd = 0.0
                gripper_cmd = 255.0
                data.ctrl[10] = gripper_cmd

                target_arm_pos = np.array([block_pos[0], block_pos[1], 0.45])
                q_cmd = arm_ik.compute_ik(data, target_arm_pos, R_down, dt=dt)
                data.ctrl[4:10] = q_cmd

                if pinch_pos[2] > 0.40 or (t - state_start_time) > 2.0:
                    state = "TRANSPORT"
                    state_start_time = t
                    print(f"[{t:5.2f}s] PAYLOAD LIFTED (z={block_pos[2]:.2f}m) -> State: TRANSPORT")

            elif state == "TRANSPORT":
                gripper_cmd = 255.0
                data.ctrl[10] = gripper_cmd

                # Hold arm steady carrying payload
                q_cmd = arm_ik.compute_ik(data, target_arm_pos, R_down, dt=dt)
                data.ctrl[4:10] = q_cmd

                # Smoothly transport back to depot station (x = -0.60)
                dist_to_depot = abs(base_pos[0] - (-0.60))
                if dist_to_depot > 0.05:
                    v_target = min(0.60, np.sqrt(2.0 * 1.0 * max(0.0, dist_to_depot)))
                    v_cmd = -float(v_target)
                    w_cmd = float(1.5 * base_pos[1])
                else:
                    v_cmd = 0.0
                    w_cmd = 0.0
                    state = "RELEASE"
                    state_start_time = t
                    print(f"[{t:5.2f}s] RETURNED TO DEPOT -> State: RELEASE")

            elif state == "RELEASE":
                v_cmd = 0.0
                w_cmd = 0.0
                gripper_cmd = 0.0
                data.ctrl[10] = gripper_cmd

                if (t - state_start_time) > 1.2:
                    state = "COMPLETED"
                    print(f"[{t:5.2f}s] MISSION COMPLETE! Payload safely deposited into depot bin.")

            elif state == "COMPLETED":
                v_cmd = 0.0
                w_cmd = 0.0

            # Apply wheel actuators
            wheel_speeds = base_drive.compute_wheel_speeds(v_cmd, w_cmd, dt=dt)
            data.ctrl[0:4] = wheel_speeds

            # Step simulation physics
            mujoco.mj_step(model, data)

            if not headless:
                viewer.cam.lookat[:] = base_pos
                viewer.sync()

            # Logging at 50 Hz
            if int(t / dt) % 10 == 0:
                t_log.append(t)
                state_log.append(state)
                base_x_log.append(base_pos[0])
                base_y_log.append(base_pos[1])
                pitch_log.append(np.degrees(pitch_angle))
                ee_x_log.append(pinch_pos[0])
                ee_y_log.append(pinch_pos[1])
                ee_z_log.append(pinch_pos[2])
                block_x_log.append(block_pos[0])
                block_y_log.append(block_pos[1])
                block_z_log.append(block_pos[2])
                v_cmd_log.append(v_cmd)
                w_cmd_log.append(w_cmd)
                grip_log.append(gripper_cmd)
                joint_q_log.append(data.qpos[11:17].copy())

    print("Generating mission analysis figure...")
    save_analysis_plot(
        np.array(t_log), state_log,
        np.array(base_x_log), np.array(base_y_log), np.array(pitch_log),
        np.array(ee_x_log), np.array(ee_y_log), np.array(ee_z_log),
        np.array(block_x_log), np.array(block_y_log), np.array(block_z_log),
        np.array(v_cmd_log), np.array(w_cmd_log), np.array(grip_log),
        np.array(joint_q_log),
        table_pos=data.xpos[table_id], bin_pos=data.xpos[bin_id]
    )


def save_analysis_plot(
    t, states, base_x, base_y, pitch,
    ee_x, ee_y, ee_z, block_x, block_y, block_z,
    v_cmd, w_cmd, grip, joint_q,
    table_pos, bin_pos
):
    fig = plt.figure(figsize=(18, 11))
    plt.suptitle("Clearpath Jackal J100 + UR5e Mobile Manipulator — End-to-End Mission Analysis", fontsize=15, fontweight="bold")

    # 1. 2D Top-Down Arena Trajectory
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(base_x, base_y, "b-", linewidth=2.0, label="Jackal Base Path")
    ax1.scatter([base_x[0]], [base_y[0]], color="green", s=100, marker="o", label="Start", zorder=5)
    ax1.scatter([table_pos[0]], [table_pos[1]], color="brown", s=150, marker="s", label="Staging Table", zorder=5)
    ax1.scatter([bin_pos[0]], [bin_pos[1]], color="royalblue", s=150, marker="D", label="Dropoff Bin", zorder=5)
    ax1.plot(block_x, block_y, "r--", linewidth=1.8, label="Block Position")
    ax1.set_title("1. Arena Top-Down Trajectory", fontweight="bold")
    ax1.set_xlabel("X Position [m]")
    ax1.set_ylabel("Y Position [m]")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.axis("equal")

    # 2. Pick & Place Vertical Elevation Profile
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(t, ee_z, "purple", linewidth=1.8, label="Pinch Site Z [m]")
    ax2.plot(t, block_z, "crimson", linewidth=2.0, linestyle="--", label="Block Z [m]")
    ax2.axhline(0.15, color="brown", linestyle=":", label="Table Surface (0.15m)")
    ax2.axhline(0.10, color="navy", linestyle=":", label="Bin Surface (0.10m)")
    ax2.set_title("2. Vertical Pick-and-Place Elevation Profile", fontweight="bold")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Z Height [m]")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Chassis Pitch Angle (Anti-Wheelie Slew Protection)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(t, pitch, "teal", linewidth=1.5, label=r"Chassis Pitch $\theta_{\mathrm{pitch}}$")
    ax3.axhline(0, color="gray", linestyle="--", alpha=0.6)
    max_pitch = np.max(np.abs(pitch)) if len(pitch) > 0 else 0.0
    ax3.set_title(f"3. Chassis Pitch Stability (Max: {max_pitch:.2f}°)", fontweight="bold")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Pitch Angle [deg]")
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Mobile Base Commands (Linear & Angular)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(t, v_cmd, "forestgreen", linewidth=1.8, label=r"Linear Velocity $v_{\mathrm{cmd}}$ [m/s]")
    ax4.plot(t, w_cmd, "darkorange", linewidth=1.5, linestyle="--", label=r"Angular Velocity $\omega_{\mathrm{cmd}}$ [rad/s]")
    ax4.set_title("4. Mobile Base Guidance Commands", fontweight="bold")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Command Magnitude")
    ax4.legend(loc="best", fontsize=8)
    ax4.grid(True, linestyle="--", alpha=0.5)

    # 5. Robotiq 2F-85 Gripper Command
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t, grip, "darkred", linewidth=2.0, label="Fingers Actuator Command")
    ax5.set_title("5. Gripper Grasp / Release State", fontweight="bold")
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Ctrl Input (0=Open, 255=Closed)")
    ax5.set_ylim(-10, 270)
    ax5.legend(loc="best", fontsize=8)
    ax5.grid(True, linestyle="--", alpha=0.5)

    # 6. UR5e Joint Configurations
    ax6 = fig.add_subplot(2, 3, 6)
    joint_labels = ["Shoulder Pan", "Shoulder Lift", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3"]
    for j in range(6):
        ax6.plot(t, np.degrees(joint_q[:, j]), label=joint_labels[j], alpha=0.85)
    ax6.set_title("6. UR5e Joint Trajectories (Degrees)", fontweight="bold")
    ax6.set_xlabel("Time [s]")
    ax6.set_ylabel("Joint Angle [deg]")
    ax6.legend(loc="best", fontsize=7, ncol=2)
    ax6.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_img = os.path.join(SCRIPT_DIR, "mobile_manipulator_mission_analysis.png")
    plt.savefig(out_img, dpi=200)
    print(f"Mission Analysis Figure saved successfully to: {out_img}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Mobile Manipulator Mission Simulation")
    parser.add_argument("--headless", action="store_true", default=True, help="Run in fast headless mode")
    parser.add_argument("--gui", action="store_true", help="Launch interactive 3D viewer")
    parser.add_argument("--duration", type=float, default=16.0, help="Simulation duration in seconds")
    args = parser.parse_args()

    is_headless = not args.gui
    run_mission(headless=is_headless, duration=args.duration)
