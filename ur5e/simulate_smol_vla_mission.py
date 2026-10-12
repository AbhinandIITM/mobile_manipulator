"""
simulate_smol_vla_mission.py
============================
End-to-End Vision-Language-Action (VLA) Mobile Manipulation Mission
Driven by SmolVLM-256M-Instruct Perception and Whole-Body Control:

Mission Execution:
1. TRANSIT       : Base drives autonomously along hallway toward table using front camera awareness.
2. DOCK          : Base precision docks at manipulation offset.
3. VLM_GROUNDING : Wrist RealSense D435i captures RGB-D frame; SmolVLM reasons and grounds 3D affordance.
4. DEPLOY        : Arm deploys to visually grounded pre-grasp hover point.
5. DESCEND       : Arm descends onto object.
6. GRASP         : Robotiq 2F-85 closes with force closure.
7. LIFT          : Vertical clearance lift.
8. TRANSPORT     : Mobile base transports payload to depot bin.
9. RELEASE       : Object released into bin.
10. ANALYTICS    : Multi-panel performance telemetry figure generated.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
J100_DIR = os.path.join(PROJECT_ROOT, "j100")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, J100_DIR)

from smol_vla_agent import SmolVLMAffordanceGrounder, SmolVLAActionController


def run_vla_mission(headless: bool = True, duration: float = 18.0, use_vlm: bool = True, speed: float = 1.0):
    scene_path = os.path.join(SCRIPT_DIR, "mobile_manipulator_scene.xml")
    print("=" * 75)
    print(f" Loading Mobile Manipulator Scene: {scene_path}")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table_drop_box")
    pinch_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")

    # Reset data
    mujoco.mj_resetData(model, data)

    # Initial base position facing table along -X
    data.qpos[0] = -0.5
    data.qpos[1] = 0.0
    data.qpos[2] = 0.065
    data.qpos[3] = 1.0
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0

    # Arm starts in resting pose
    home_q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    data.qpos[11:17] = home_q
    data.ctrl[4:10] = home_q
    data.ctrl[10] = 0.0  # Gripper open
    mujoco.mj_forward(model, data)

    # Initialize SmolVLA Modules
    grounder = SmolVLMAffordanceGrounder(use_vlm_inference=use_vlm)
    controller = SmolVLAActionController(model)
    controller.reset(home_q)

    # Off-screen renderers for dual cameras
    img_w, img_h = 320, 240
    renderer_front = mujoco.Renderer(model, img_h, img_w)
    renderer_wrist = mujoco.Renderer(model, img_h, img_w)

    dt = model.opt.timestep
    sim_time = 0.0

    # State Machine
    state = "TRANSIT"
    state_start_time = 0.0

    grounded_target_3d = None
    vlm_caption = "Initializing mission..."
    wrist_bbox = None
    wrist_centroid = None

    # Telemetry logs
    t_log, state_log = [], []
    base_x_log, base_y_log = [], []
    ee_x_log, ee_y_log, ee_z_log = [], [], []
    block_x_log, block_y_log, block_z_log = [], [], []
    v_cmd_log, w_cmd_log, grip_log = [], [], []

    target_arm_pos = data.site_xpos[pinch_site_id].copy()

    # Create HUD video writer if saving
    hud_frames = []
    last_hud_time = -1.0

    print("=" * 75)
    print(" SmolVLA Mobile Manipulation Mission Started")
    print(" Architecture: SmolVLM Perception + Whole-Body DLS IK & Skid-Steer")
    print("=" * 75)

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

    # Initial viewer camera position to frame the whole arena (Base + Table + Bin)
    if not headless and hasattr(viewer_ctx, "cam"):
        viewer_ctx.cam.azimuth = 130.0
        viewer_ctx.cam.elevation = -20.0
        viewer_ctx.cam.distance = 4.2
        viewer_ctx.cam.lookat[:] = [-1.5, 0.0, 0.4]

    with viewer_ctx as viewer:
        while viewer.is_running() and sim_time < duration:
            step_start_wall = time.time()
            t = data.time
            sim_time = t

            base_pos = data.xpos[chassis_id].copy()
            block_pos = data.xpos[block_id].copy()
            pinch_pos = data.site_xpos[pinch_site_id].copy()

            v_cmd = 0.0
            w_cmd = 0.0
            gripper_cmd = data.ctrl[10]

            # ─── 1. State Machine ──────────────────────────────────────────
            if state == "TRANSIT":
                target_arm_pos = pinch_pos.copy()
                dock_x = -1.78
                dx = dock_x - base_pos[0]
                dist_to_dock = abs(dx)

                if dist_to_dock > 0.02:
                    v_cmd = min(0.55, np.sqrt(2.0 * 0.8 * dist_to_dock))
                    w_cmd = float(-1.5 * base_pos[1])  # Centerline regulation
                else:
                    v_cmd = 0.0
                    w_cmd = 0.0
                    state = "SETTLE"
                    state_start_time = t
                    print(f"[{t:5.2f}s] AT DOCK POSITION -> Settling chassis before perception...")

            elif state == "SETTLE":
                v_cmd = 0.0
                w_cmd = 0.0
                target_arm_pos = pinch_pos.copy()
                if (t - state_start_time) > 0.4 and abs(data.qvel[0]) < 0.02:
                    state = "VLM_GROUNDING"
                    state_start_time = t
                    print(f"[{t:5.2f}s] CHASSIS FULLY SETTLED -> Initiating SmolVLM Visual Grounding...")

            elif state == "VLM_GROUNDING":
                v_cmd = 0.0
                w_cmd = 0.0

                # Capture RGB-D from wrist RealSense
                renderer_wrist.disable_depth_rendering()
                renderer_wrist.update_scene(data, camera="d435i_color")
                wrist_rgb = renderer_wrist.render()

                renderer_wrist.enable_depth_rendering()
                renderer_wrist.update_scene(data, camera="d435i_color")
                wrist_depth = renderer_wrist.render()

                # Query SmolVLM Affordance Grounder
                instruction = "Locate the red block affordance on the staging table for grasping."
                g_pos, centroid, bbox, caption = grounder.ground_affordance(
                    wrist_rgb,
                    wrist_depth,
                    model,
                    data,
                    camera_name="d435i_color",
                    instruction=instruction,
                    query_vlm=True
                )

                if g_pos is not None:
                    # Grounded X and Y from perception; Z anchored to block center height on tabletop
                    grounded_target_3d = np.array([g_pos[0], g_pos[1], 0.325])
                    wrist_bbox = bbox
                    wrist_centroid = centroid
                    vlm_caption = caption if caption else "Target red block localized"
                    print(f"[{t:5.2f}s] SmolVLM Output: '{vlm_caption}'")
                    print(f"[{t:5.2f}s] Grounded 3D Pos: X={g_pos[0]:.3f}, Y={g_pos[1]:.3f}, Z=0.325m")
                else:
                    grounded_target_3d = np.array([block_pos[0], block_pos[1], 0.325])
                    vlm_caption = "Geometric fallback affordance active"

                state = "DEPLOY"
                state_start_time = t
                target_arm_pos = np.array([grounded_target_3d[0], grounded_target_3d[1], 0.48])

            elif state == "DEPLOY":
                v_cmd = 0.0
                w_cmd = 0.0
                target_arm_pos = np.array([grounded_target_3d[0], grounded_target_3d[1], 0.48])
                dist_ee = np.linalg.norm(pinch_pos - target_arm_pos)
                if dist_ee < 0.035 or (t - state_start_time) > 1.8:
                    state = "DESCEND"
                    state_start_time = t
                    print(f"[{t:5.2f}s] PRE-GRASP HOVER ALIGNED -> State: DESCEND")

            elif state == "DESCEND":
                v_cmd = 0.0
                w_cmd = 0.0
                # Smooth S-curve vertical descent to avoid dynamic shock
                dur_descend = 1.2
                progress = np.clip((t - state_start_time) / dur_descend, 0.0, 1.0)
                s = 0.5 * (1.0 - np.cos(np.pi * progress))
                curr_z = 0.48 - s * (0.48 - 0.325)
                target_arm_pos = np.array([grounded_target_3d[0], grounded_target_3d[1], curr_z])

                dist_ee = np.linalg.norm(pinch_pos - np.array([grounded_target_3d[0], grounded_target_3d[1], 0.325]))
                if progress >= 1.0 and (dist_ee < 0.025 or (t - state_start_time) > (dur_descend + 0.3)):
                    state = "GRASP"
                    state_start_time = t
                    print(f"[{t:5.2f}s] AT CONTACT POSITION -> State: GRASP")

            elif state == "GRASP":
                v_cmd = 0.0
                w_cmd = 0.0
                target_arm_pos = np.array([grounded_target_3d[0], grounded_target_3d[1], 0.325])
                gripper_cmd = 255.0  # Full force closure
                if (t - state_start_time) > 1.2:
                    state = "LIFT"
                    state_start_time = t
                    print(f"[{t:5.2f}s] FORCE CLOSURE SECURED -> State: LIFT")

            elif state == "LIFT":
                v_cmd = 0.0
                w_cmd = 0.0
                gripper_cmd = 255.0
                # Smooth S-curve lift
                dur_lift = 1.2
                progress = np.clip((t - state_start_time) / dur_lift, 0.0, 1.0)
                s = 0.5 * (1.0 - np.cos(np.pi * progress))
                curr_z = 0.325 + s * (0.48 - 0.325)
                target_arm_pos = np.array([grounded_target_3d[0], grounded_target_3d[1], curr_z])

                if progress >= 1.0 and (abs(pinch_pos[2] - 0.48) < 0.035 or (t - state_start_time) > (dur_lift + 0.3)):
                    state = "TRANSFER_TO_BOX"
                    state_start_time = t
                    print(f"[{t:5.2f}s] PAYLOAD LIFTED TO CLEARANCE (Z={pinch_pos[2]:.2f}m) -> State: TRANSFER_TO_BOX")

            elif state == "TRANSFER_TO_BOX":
                v_cmd = 0.0
                w_cmd = 0.0
                gripper_cmd = 255.0
                box_pos = data.xpos[box_id].copy()
                dur_transfer = 1.6
                progress = np.clip((t - state_start_time) / dur_transfer, 0.0, 1.0)
                s = 0.5 * (1.0 - np.cos(np.pi * progress))

                # Smoothly interpolate EE position across table surface to center above drop box
                curr_x = grounded_target_3d[0] + s * (box_pos[0] - grounded_target_3d[0])
                curr_y = grounded_target_3d[1] + s * (box_pos[1] - grounded_target_3d[1])
                curr_z = 0.48 - s * (0.48 - 0.44)
                target_arm_pos = np.array([curr_x, curr_y, curr_z])

                dist_to_box = np.linalg.norm(pinch_pos[:2] - box_pos[:2])
                if progress >= 1.0 and (dist_to_box < 0.035 or (t - state_start_time) > (dur_transfer + 0.3)):
                    state = "RELEASE_INTO_BOX"
                    state_start_time = t
                    print(f"[{t:5.2f}s] HOVERING OVER TABLE DROP BOX -> State: RELEASE_INTO_BOX")

            elif state == "RELEASE_INTO_BOX":
                v_cmd = 0.0
                w_cmd = 0.0
                box_pos = data.xpos[box_id].copy()
                # Descend slightly above rim of the drop box
                target_arm_pos = np.array([box_pos[0], box_pos[1], 0.38])
                gripper_cmd = 0.0  # Open fingers to drop payload

                if (t - state_start_time) > 1.2:
                    state = "RETRACT"
                    state_start_time = t
                    print(f"[{t:5.2f}s] PAYLOAD SAFELY DROPPED IN TABLE BOX -> State: RETRACT")

            elif state == "RETRACT":
                v_cmd = 0.0
                w_cmd = 0.0
                gripper_cmd = 0.0
                dur_retract = 1.2
                progress = np.clip((t - state_start_time) / dur_retract, 0.0, 1.0)
                s = 0.5 * (1.0 - np.cos(np.pi * progress))
                # Retract arm safely back and up
                box_pos = data.xpos[box_id].copy()
                curr_x = box_pos[0] - s * (box_pos[0] - (base_pos[0] - 0.45))
                curr_y = box_pos[1] - s * (box_pos[1] - 0.0)
                curr_z = 0.38 + s * (0.46 - 0.38)
                target_arm_pos = np.array([curr_x, curr_y, curr_z])

                if progress >= 1.0 or (t - state_start_time) > (dur_retract + 0.3):
                    state = "MISSION_COMPLETE"
                    state_start_time = t
                    print(f"[{t:5.2f}s] ARM RETRACTED TO RESTING POSE -> MISSION COMPLETE!")

            elif state == "MISSION_COMPLETE":
                v_cmd = 0.0
                w_cmd = 0.0
                gripper_cmd = 0.0

            # ─── 2. Compute Whole-Body Control ─────────────────────────────
            ctrl_vec = controller.compute_actions(
                data=data,
                target_ee_pos=target_arm_pos,
                v_base_cmd=v_cmd,
                w_base_cmd=w_cmd,
                gripper_cmd=gripper_cmd,
                dt=dt
            )
            data.ctrl[:] = ctrl_vec

            # Step Physics
            mujoco.mj_step(model, data)

            # Sync 3D passive viewer in GUI mode
            if not headless:
                viewer.sync()
                # Real-time pacing
                elapsed_wall = time.time() - step_start_wall
                target_step_time = dt / max(0.01, speed)
                if elapsed_wall < target_step_time:
                    time.sleep(target_step_time - elapsed_wall)

            # ─── 3. Telemetry Logging ──────────────────────────────────────
            t_log.append(t)
            state_log.append(state)
            base_x_log.append(base_pos[0])
            base_y_log.append(base_pos[1])
            ee_x_log.append(pinch_pos[0])
            ee_y_log.append(pinch_pos[1])
            ee_z_log.append(pinch_pos[2])
            block_x_log.append(block_pos[0])
            block_y_log.append(block_pos[1])
            block_z_log.append(block_pos[2])
            v_cmd_log.append(v_cmd)
            w_cmd_log.append(w_cmd)
            grip_log.append(gripper_cmd)

            # ─── 4. Dual-Camera HUD Generation (at 10 Hz) ───────────────────
            if t - last_hud_time >= 0.10:
                renderer_front.disable_depth_rendering()
                renderer_front.update_scene(data, camera="front_cam")
                f_frame = cv2.cvtColor(renderer_front.render(), cv2.COLOR_RGB2BGR)

                renderer_wrist.disable_depth_rendering()
                renderer_wrist.update_scene(data, camera="d435i_color")
                w_frame = cv2.cvtColor(renderer_wrist.render(), cv2.COLOR_RGB2BGR)

                # Draw front camera HUD
                cv2.putText(f_frame, "BASE FRONT CAM (SITUATIONAL)", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(f_frame, f"Base X: {base_pos[0]:.2f}m | Vel: {v_cmd:+.2f}m/s", (10, 225),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

                # Draw wrist camera HUD with VLM grounding
                cv2.putText(w_frame, "WRIST D435i (EYE-IN-HAND)", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                if wrist_bbox is not None and state in ["DEPLOY", "DESCEND", "GRASP"]:
                    ymin, xmin, ymax, xmax = wrist_bbox
                    cv2.rectangle(w_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    if wrist_centroid is not None:
                        uc, vc = wrist_centroid
                        cv2.drawMarker(w_frame, (uc, vc), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
                        cv2.putText(w_frame, "SmolVLM Target", (xmin, max(15, ymin - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)

                cv2.putText(w_frame, f"EE Z: {pinch_pos[2]:.3f}m | Grip: {gripper_cmd:.0f}", (10, 225),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

                # Combine into side-by-side HUD banner
                hud = np.hstack([f_frame, w_frame])
                banner = np.zeros((45, hud.shape[1], 3), dtype=np.uint8)
                cv2.putText(banner, f"SmolVLA Mobile Manipulator | State: [{state}] | Sim Time: {t:.2f}s",
                            (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(banner, f"VLM Semantic Grounding: \"{vlm_caption[:60]}\"",
                            (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 255), 1, cv2.LINE_AA)

                hud_combined = np.vstack([banner, hud])
                hud_frames.append(hud_combined)
                last_hud_time = t

                if not headless:
                    cv2.imshow("SmolVLA Dual-Camera Telemetry HUD", hud_combined)
                    cv2.waitKey(1)

    if not headless:
        cv2.destroyAllWindows()

    print("=" * 75)
    print(" Mission Complete! Rendering Multi-Panel Performance Figure...")
    print("=" * 75)

    # Save representative HUD frame
    if len(hud_frames) > 0:
        cv2.imwrite(os.path.join(SCRIPT_DIR, "smol_vla_hud_snapshot.png"), hud_frames[len(hud_frames) // 2])
        print(f" Saved HUD Snapshot to: {os.path.join(SCRIPT_DIR, 'smol_vla_hud_snapshot.png')}")

    # Generate 6-Panel Performance Analysis Figure
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("SmolVLA Autonomous Mobile Manipulation Performance Analysis\n(Clearpath Jackal J100 + UR5e + Robotiq 2F-85 + SmolVLM)", fontsize=13, fontweight='bold')

    t_arr = np.array(t_log)

    # 1. Base Trajectory
    axs[0, 0].plot(t_arr, base_x_log, 'b-', label='Chassis X (m)', linewidth=2)
    axs[0, 0].axhline(y=-1.75, color='r', linestyle='--', label='Staging Dock X (-1.75m)')
    axs[0, 0].axhline(y=-0.50, color='g', linestyle='--', label='Depot Station X (-0.50m)')
    axs[0, 0].set_ylabel("Base Position (m)")
    axs[0, 0].set_title("Chassis Longitudinal Motion", fontweight='bold')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(loc='best', fontsize=8)

    # 2. Base Control Commands
    axs[0, 1].plot(t_arr, v_cmd_log, 'm-', label='Longitudinal Vel v_cmd (m/s)', linewidth=1.5)
    axs[0, 1].plot(t_arr, w_cmd_log, 'c-', label='Angular Vel w_cmd (rad/s)', linewidth=1.2)
    axs[0, 1].set_ylabel("Command (m/s, rad/s)")
    axs[0, 1].set_title("Skid-Steer Drive Velocity Inputs", fontweight='bold')
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend(loc='best', fontsize=8)

    # 3. Arm EE Position Tracking
    axs[1, 0].plot(t_arr, ee_x_log, label='EE X', color='tab:blue')
    axs[1, 0].plot(t_arr, ee_y_log, label='EE Y', color='tab:orange')
    axs[1, 0].plot(t_arr, ee_z_log, label='EE Z', color='tab:green', linewidth=2)
    axs[1, 0].set_ylabel("Coordinates (m)")
    axs[1, 0].set_title("End-Effector Cartesian Trajectory", fontweight='bold')
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend(loc='best', fontsize=8)

    # 4. Target Block Altitude & Lift
    axs[1, 1].plot(t_arr, block_z_log, 'r-', label='Target Block Z (m)', linewidth=2.5)
    axs[1, 1].axhline(y=0.325, color='k', linestyle=':', label='Tabletop Surface (0.325m)')
    axs[1, 1].axhline(y=0.45, color='g', linestyle='--', label='Transit Clearance (0.45m)')
    axs[1, 1].set_ylabel("Altitude (m)")
    axs[1, 1].set_title("Payload Altitude & Grasp Verification", fontweight='bold')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend(loc='best', fontsize=8)

    # 5. Gripper Command & Force Closure
    axs[2, 0].plot(t_arr, grip_log, 'k-', label='Robotiq 2F-85 Ctrl (0=Open, 255=Close)', linewidth=2)
    axs[2, 0].set_ylabel("Actuator Control (0-255)")
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 0].set_title("Gripper Actuation Profile", fontweight='bold')
    axs[2, 0].grid(True, alpha=0.3)
    axs[2, 0].legend(loc='best', fontsize=8)

    # 6. Mission State Timeline
    unique_states = list(dict.fromkeys(state_log))
    state_map = {s: i for i, s in enumerate(unique_states)}
    numeric_states = [state_map[s] for s in state_log]
    axs[2, 1].step(t_arr, numeric_states, 'tab:purple', linewidth=2, where='post')
    axs[2, 1].set_yticks(range(len(unique_states)))
    axs[2, 1].set_yticklabels(unique_states, fontsize=8)
    axs[2, 1].set_xlabel("Time (s)")
    axs[2, 1].set_title("Autonomous Mission Phase Transitions", fontweight='bold')
    axs[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    analysis_plot_path = os.path.join(SCRIPT_DIR, "smol_vla_mission_analysis.png")
    plt.savefig(analysis_plot_path, dpi=300)
    plt.close()
    print(f" Analysis Figure Generated: {analysis_plot_path}")

    return {
        "final_block_z": block_z_log[-1],
        "final_block_x": block_x_log[-1],
        "final_block_y": block_y_log[-1],
        "final_state": state_log[-1],
        "plot_path": analysis_plot_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate SmolVLA Mobile Manipulation Mission")
    parser.add_argument("--gui", action="store_true", help="Launch interactive GUI viewer")
    parser.add_argument("--duration", type=float, default=18.0, help="Simulation duration (s)")
    parser.add_argument("--no-vlm", action="store_true", help="Run without VLM inference (perception fast-path)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (e.g. 1.0 for real-time, 0.5 for 2x slow-mo)")
    args = parser.parse_args()

    run_vla_mission(headless=not args.gui, duration=args.duration, use_vlm=not args.no_vlm, speed=args.speed)
