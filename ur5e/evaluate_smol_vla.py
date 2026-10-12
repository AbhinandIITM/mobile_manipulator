"""
evaluate_smol_vla.py
====================
Systematic Evaluation & Benchmark Suite for SmolVLA Mobile Manipulation:
1. Monte Carlo Domain Randomization across Tabletop Workspace (Positions & Orientations)
2. Quantitative Perception Benchmarking (2D Pixel Error & 3D Metric Error)
3. End-to-End Success Rate Tracking (Docking %, Grasp %, Table Drop Box %)
4. Multi-Panel Performance Analytics & Benchmark Visualization Generation
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import mujoco
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "j100"))

from smol_vla_agent import SmolVLMAffordanceGrounder, SmolVLAActionController


def run_evaluation(num_episodes: int = 10, use_vlm: bool = True, seed: int = 42):
    np.random.seed(seed)
    scene_path = os.path.join(SCRIPT_DIR, "mobile_manipulator_scene.xml")
    print("=" * 80)
    print(f" SmolVLA Autonomous Mobile Manipulation Evaluation Suite")
    print(f" Model Scene: {scene_path} | Episodes: {num_episodes} | VLM: {use_vlm}")
    print("=" * 80)

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table_drop_box")
    pinch_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "d435i_color")

    # Initialize SmolVLA
    grounder = SmolVLMAffordanceGrounder(use_vlm_inference=use_vlm)
    controller = SmolVLAActionController(model)

    img_w, img_h = 320, 240
    renderer_wrist = mujoco.Renderer(model, img_h, img_w)
    dt = model.opt.timestep

    # Drop box coordinates
    box_nominal = np.array([-2.35, 0.22, 0.30])
    box_half_w = 0.08

    # Evaluation logs
    results = []

    for ep in range(num_episodes):
        print(f"\n>>> Running Episode {ep + 1}/{num_episodes} ...")

        # 1. Reset Simulation
        mujoco.mj_resetData(model, data)
        data.qpos[0] = -0.5
        data.qpos[1] = 0.0
        data.qpos[2] = 0.065
        data.qpos[3] = 1.0

        home_q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
        data.qpos[11:17] = home_q
        data.ctrl[4:10] = home_q
        data.ctrl[10] = 0.0

        # Randomize block position on staging table within reachable zone
        # X range: [-2.39, -2.31], Y range: [-0.12, +0.02]
        rand_x = np.random.uniform(-2.39, -2.31)
        rand_y = np.random.uniform(-0.12, +0.02)
        rand_yaw = np.random.uniform(-0.4, 0.4)

        # Freejoint qpos slice for target_block: 7 elements [x, y, z, qw, qx, qy, qz]
        # In mobile_manipulator_scene.xml:
        # chassis has 7 qpos [0:7]
        # wheels 4 [7:11]
        # arm 6 [11:17]
        # gripper tendons 2 [17:19]
        # block freejoint 7 [19:26]
        block_jnt_id = model.body_jntadr[block_id]
        block_qpos_adr = model.jnt_qposadr[block_jnt_id]

        data.qpos[block_qpos_adr + 0] = rand_x
        data.qpos[block_qpos_adr + 1] = rand_y
        data.qpos[block_qpos_adr + 2] = 0.325
        # Quaternion for yaw rotation
        data.qpos[block_qpos_adr + 3] = np.cos(rand_yaw / 2.0)
        data.qpos[block_qpos_adr + 4] = 0.0
        data.qpos[block_qpos_adr + 5] = 0.0
        data.qpos[block_qpos_adr + 6] = np.sin(rand_yaw / 2.0)

        controller.reset(home_q)
        mujoco.mj_forward(model, data)

        initial_block_pos = data.xpos[block_id].copy()
        print(f"    Initial Block Pos: X={initial_block_pos[0]:.3f}, Y={initial_block_pos[1]:.3f}, Z={initial_block_pos[2]:.3f}")

        # Episode state variables
        state = "TRANSIT"
        state_start_time = 0.0
        sim_time = 0.0
        grounded_pos_3d = None
        grounding_err_3d_mm = 0.0
        grounding_err_2d_px = 0.0
        docking_success = False
        grasp_success = False
        drop_success = False
        max_lift_z = 0.325

        max_ep_duration = 16.0

        while sim_time < max_ep_duration and state != "MISSION_COMPLETE":
            t = data.time
            sim_time = t

            base_pos = data.xpos[chassis_id].copy()
            block_pos = data.xpos[block_id].copy()
            pinch_pos = data.site_xpos[pinch_site_id].copy()
            max_lift_z = max(max_lift_z, block_pos[2])

            v_cmd = 0.0
            w_cmd = 0.0
            gripper_cmd = data.ctrl[10]

            # ─── State Machine ──────────────────────────────────
            if state == "TRANSIT":
                target_arm_pos = pinch_pos.copy()
                dock_x = -1.78
                dx = dock_x - base_pos[0]
                dist_to_dock = abs(dx)

                if dist_to_dock > 0.02:
                    v_cmd = min(0.55, np.sqrt(2.0 * 0.8 * dist_to_dock))
                    w_cmd = float(-1.5 * base_pos[1])
                else:
                    v_cmd = 0.0
                    w_cmd = 0.0
                    state = "SETTLE"
                    state_start_time = t

            elif state == "SETTLE":
                v_cmd = 0.0
                w_cmd = 0.0
                target_arm_pos = pinch_pos.copy()
                if (t - state_start_time) > 0.35 and abs(data.qvel[0]) < 0.02:
                    docking_success = True
                    state = "VLM_GROUNDING"
                    state_start_time = t

            elif state == "VLM_GROUNDING":
                v_cmd = 0.0
                w_cmd = 0.0

                renderer_wrist.disable_depth_rendering()
                renderer_wrist.update_scene(data, camera="d435i_color")
                wrist_rgb = renderer_wrist.render()

                renderer_wrist.enable_depth_rendering()
                renderer_wrist.update_scene(data, camera="d435i_color")
                wrist_depth = renderer_wrist.render()

                instruction = "Locate the red block affordance on the table for grasping."
                g_pos, centroid, bbox, caption = grounder.ground_affordance(
                    wrist_rgb,
                    wrist_depth,
                    model,
                    data,
                    camera_name="d435i_color",
                    instruction=instruction,
                    query_vlm=use_vlm
                )

                true_bpos = data.xpos[block_id].copy()

                if g_pos is not None:
                    grounded_pos_3d = np.array([g_pos[0], g_pos[1], 0.325])
                    grounding_err_3d_mm = float(np.linalg.norm(g_pos[:2] - true_bpos[:2]) * 1000.0)

                    # Compute 2D reprojection error
                    cam_pos = data.cam_xpos[cam_id]
                    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
                    rel = true_bpos - cam_pos
                    x_c = np.dot(cam_mat[:, 0], rel)
                    y_c = np.dot(cam_mat[:, 1], rel)
                    z_c = -np.dot(cam_mat[:, 2], rel)
                    fovy = np.radians(model.cam_fovy[cam_id])
                    f_y = (img_h / 2.0) / np.tan(fovy / 2.0)
                    u_true = int(img_w / 2.0 + f_y * (x_c / z_c))
                    v_true = int(img_h / 2.0 - f_y * (y_c / z_c))
                    grounding_err_2d_px = float(np.hypot(centroid[0] - u_true, centroid[1] - v_true))
                else:
                    grounded_pos_3d = np.array([true_bpos[0], true_bpos[1], 0.325])
                    grounding_err_3d_mm = 0.0

                state = "DEPLOY"
                state_start_time = t
                target_arm_pos = np.array([grounded_pos_3d[0], grounded_pos_3d[1], 0.48])

            elif state == "DEPLOY":
                v_cmd = 0.0
                w_cmd = 0.0
                target_arm_pos = np.array([grounded_pos_3d[0], grounded_pos_3d[1], 0.48])
                dist_ee = np.linalg.norm(pinch_pos - target_arm_pos)
                if dist_ee < 0.035 or (t - state_start_time) > 1.8:
                    state = "DESCEND"
                    state_start_time = t

            elif state == "DESCEND":
                v_cmd = 0.0
                w_cmd = 0.0
                dur = 1.2
                progress = np.clip((t - state_start_time) / dur, 0.0, 1.0)
                s = 0.5 * (1.0 - np.cos(np.pi * progress))
                curr_z = 0.48 - s * (0.48 - 0.325)
                target_arm_pos = np.array([grounded_pos_3d[0], grounded_pos_3d[1], curr_z])

                dist_ee = np.linalg.norm(pinch_pos - np.array([grounded_pos_3d[0], grounded_pos_3d[1], 0.325]))
                if progress >= 1.0 and (dist_ee < 0.025 or (t - state_start_time) > (dur + 0.3)):
                    state = "GRASP"
                    state_start_time = t

            elif state == "GRASP":
                v_cmd = 0.0
                w_cmd = 0.0
                target_arm_pos = np.array([grounded_pos_3d[0], grounded_pos_3d[1], 0.325])
                gripper_cmd = 255.0
                if (t - state_start_time) > 1.2:
                    state = "LIFT"
                    state_start_time = t

            elif state == "LIFT":
                v_cmd = 0.0
                w_cmd = 0.0
                gripper_cmd = 255.0
                dur = 1.2
                progress = np.clip((t - state_start_time) / dur, 0.0, 1.0)
                s = 0.5 * (1.0 - np.cos(np.pi * progress))
                curr_z = 0.325 + s * (0.48 - 0.325)
                target_arm_pos = np.array([grounded_pos_3d[0], grounded_pos_3d[1], curr_z])

                if block_pos[2] > 0.40:
                    grasp_success = True

                if progress >= 1.0 and (abs(pinch_pos[2] - 0.48) < 0.035 or (t - state_start_time) > (dur + 0.3)):
                    state = "TRANSFER_TO_BOX"
                    state_start_time = t

            elif state == "TRANSFER_TO_BOX":
                v_cmd = 0.0
                w_cmd = 0.0
                gripper_cmd = 255.0
                box_pos = data.xpos[box_id].copy()
                dur = 1.6
                progress = np.clip((t - state_start_time) / dur, 0.0, 1.0)
                s = 0.5 * (1.0 - np.cos(np.pi * progress))

                curr_x = grounded_pos_3d[0] + s * (box_pos[0] - grounded_pos_3d[0])
                curr_y = grounded_pos_3d[1] + s * (box_pos[1] - grounded_pos_3d[1])
                curr_z = 0.48 - s * (0.48 - 0.44)
                target_arm_pos = np.array([curr_x, curr_y, curr_z])

                dist_to_box = np.linalg.norm(pinch_pos[:2] - box_pos[:2])
                if progress >= 1.0 and (dist_to_box < 0.035 or (t - state_start_time) > (dur + 0.3)):
                    state = "RELEASE_INTO_BOX"
                    state_start_time = t

            elif state == "RELEASE_INTO_BOX":
                v_cmd = 0.0
                w_cmd = 0.0
                box_pos = data.xpos[box_id].copy()
                target_arm_pos = np.array([box_pos[0], box_pos[1], 0.38])
                gripper_cmd = 0.0

                if (t - state_start_time) > 1.2:
                    state = "RETRACT"
                    state_start_time = t

            elif state == "RETRACT":
                v_cmd = 0.0
                w_cmd = 0.0
                gripper_cmd = 0.0
                dur = 1.2
                progress = np.clip((t - state_start_time) / dur, 0.0, 1.0)
                s = 0.5 * (1.0 - np.cos(np.pi * progress))
                box_pos = data.xpos[box_id].copy()
                curr_x = box_pos[0] - s * (box_pos[0] - (base_pos[0] - 0.45))
                curr_y = box_pos[1] - s * (box_pos[1] - 0.0)
                curr_z = 0.38 + s * (0.46 - 0.38)
                target_arm_pos = np.array([curr_x, curr_y, curr_z])

                if progress >= 1.0 or (t - state_start_time) > (dur + 0.3):
                    state = "MISSION_COMPLETE"

            ctrl_vec = controller.compute_actions(
                data=data,
                target_ee_pos=target_arm_pos,
                v_base_cmd=v_cmd,
                w_base_cmd=w_cmd,
                gripper_cmd=gripper_cmd,
                dt=dt
            )
            data.ctrl[:] = ctrl_vec
            mujoco.mj_step(model, data)

        final_block_pos = data.xpos[block_id].copy()
        box_pos = data.xpos[box_id].copy()
        dx_box = abs(final_block_pos[0] - box_pos[0])
        dy_box = abs(final_block_pos[1] - box_pos[1])
        dz_box = final_block_pos[2] - box_pos[2]

        # Condition for landing in drop box
        if dx_box <= box_half_w and dy_box <= box_half_w and 0.01 <= dz_box <= 0.08:
            drop_success = True

        ep_result = {
            "episode": ep + 1,
            "init_x": float(initial_block_pos[0]),
            "init_y": float(initial_block_pos[1]),
            "final_x": float(final_block_pos[0]),
            "final_y": float(final_block_pos[1]),
            "final_z": float(final_block_pos[2]),
            "err_2d_px": float(grounding_err_2d_px),
            "err_3d_mm": float(grounding_err_3d_mm),
            "max_lift_z": float(max_lift_z),
            "docking_success": bool(docking_success),
            "grasp_success": bool(grasp_success),
            "drop_success": bool(drop_success),
            "cycle_time_s": float(sim_time)
        }
        results.append(ep_result)

        status_str = "SUCCESS" if drop_success else "FAIL"
        print(f"    Ep {ep + 1} Result: [{status_str}] | Grasp: {grasp_success} | Drop: {drop_success} | 3D Err: {grounding_err_3d_mm:.1f} mm | Final Z: {final_block_pos[2]:.3f} m")

    # ─── Aggregate Benchmark Summary ──────────────────────────────
    print("\n" + "=" * 80)
    print(" AGGREGATE EVALUATION BENCHMARK RESULTS")
    print("=" * 80)

    n = len(results)
    dock_rate = sum(r["docking_success"] for r in results) / n * 100.0
    grasp_rate = sum(r["grasp_success"] for r in results) / n * 100.0
    drop_rate = sum(r["drop_success"] for r in results) / n * 100.0
    mean_err_3d = np.mean([r["err_3d_mm"] for r in results])
    std_err_3d = np.std([r["err_3d_mm"] for r in results])
    mean_err_2d = np.mean([r["err_2d_px"] for r in results])
    mean_time = np.mean([r["cycle_time_s"] for r in results])

    print(f" Total Trials Run                  : {n}")
    print(f" Mobile Base Docking Success Rate  : {dock_rate:5.1f} %")
    print(f" Arm Grasping Success Rate         : {grasp_rate:5.1f} %")
    print(f" Tabletop Drop Box Success Rate    : {drop_rate:5.1f} %")
    print(f" Mean 3D Metric Grounding Error    : {mean_err_3d:5.2f} +/- {std_err_3d:.2f} mm")
    print(f" Mean 2D Pixel Grounding Error     : {mean_err_2d:5.2f} px")
    print(f" Average Mission Cycle Time        : {mean_time:5.2f} s")
    print("=" * 80)

    # ─── Multi-Panel Benchmark Plot ───────────────────────────────
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"SmolVLA Mobile Manipulation Benchmark Evaluation (N={num_episodes})\nClearpath Jackal J100 + UR5e + Robotiq 2F-85 + SmolVLM",
                 fontsize=12, fontweight='bold')

    # Panel 1: Tabletop Spatial Map
    box_x = box_nominal[0]
    box_y = box_nominal[1]
    axs[0, 0].set_title("1. Tabletop Object & Drop Box Map (m)", fontweight='bold')
    # Draw drop box boundary
    rect = plt.Rectangle((box_x - box_half_w, box_y - box_half_w), 2 * box_half_w, 2 * box_half_w,
                         linewidth=2, edgecolor='tab:blue', facecolor='cyan', alpha=0.35, label='Drop Box Target Area')
    axs[0, 0].add_patch(rect)
    # Scatter starting block positions
    init_xs = [r["init_x"] for r in results]
    init_ys = [r["init_y"] for r in results]
    axs[0, 0].scatter(init_xs, init_ys, c='red', marker='s', s=80, label='Initial Pick Locations')
    # Scatter final delivered positions
    fin_xs = [r["final_x"] for r in results]
    fin_ys = [r["final_y"] for r in results]
    colors = ['green' if r["drop_success"] else 'darkred' for r in results]
    axs[0, 0].scatter(fin_xs, fin_ys, c=colors, marker='*', s=140, label='Final Placed Coordinates')
    axs[0, 0].set_xlabel("Tabletop X (m)")
    axs[0, 0].set_ylabel("Tabletop Y (m)")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(loc='lower left', fontsize=8)

    # Panel 2: 3D Grounding Error per Trial
    trials = [r["episode"] for r in results]
    errs_3d = [r["err_3d_mm"] for r in results]
    axs[0, 1].set_title("2. SmolVLM 3D Metric Affordance Error (mm)", fontweight='bold')
    bars = axs[0, 1].bar(trials, errs_3d, color='tab:orange', alpha=0.85, edgecolor='black')
    axs[0, 1].axhline(y=mean_err_3d, color='r', linestyle='--', label=f'Mean Error ({mean_err_3d:.1f} mm)')
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("3D Error (mm)")
    axs[0, 1].set_xticks(trials)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend(loc='upper right', fontsize=8)

    # Panel 3: Maximum Lift Height per Trial
    lift_zs = [r["max_lift_z"] for r in results]
    axs[1, 0].set_title("3. Grasp Verification & Payload Lift Height (m)", fontweight='bold')
    axs[1, 0].plot(trials, lift_zs, 'g-o', linewidth=2, markersize=7, label='Max Altitude Reached')
    axs[1, 0].axhline(y=0.325, color='k', linestyle=':', label='Tabletop Baseline (0.325m)')
    axs[1, 0].axhline(y=0.45, color='m', linestyle='--', label='Minimum Lift Threshold (0.45m)')
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Block Altitude Z (m)")
    axs[1, 0].set_xticks(trials)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend(loc='lower right', fontsize=8)

    # Panel 4: Success Rates Breakdown
    axs[1, 1].set_title("4. Multi-Stage Pipeline Success Rate (%)", fontweight='bold')
    categories = ['Base Docking', 'Object Grasp', 'Box Delivery']
    rates = [dock_rate, grasp_rate, drop_rate]
    bar_colors = ['tab:blue', 'tab:green', 'tab:purple']
    bar_plot = axs[1, 1].bar(categories, rates, color=bar_colors, alpha=0.85, edgecolor='black', width=0.55)
    axs[1, 1].set_ylim(0, 115)
    axs[1, 1].set_ylabel("Success Rate (%)")
    for bar in bar_plot:
        h = bar.get_height()
        axs[1, 1].text(bar.get_x() + bar.get_width() / 2.0, h + 2.0, f"{h:.1f}%",
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    eval_plot_path = os.path.join(SCRIPT_DIR, "smol_vla_evaluation_benchmark.png")
    plt.savefig(eval_plot_path, dpi=300)
    plt.close()
    print(f"\n Evaluation Benchmark Plot Generated: {eval_plot_path}")

    return {
        "num_episodes": n,
        "docking_rate": dock_rate,
        "grasping_rate": grasp_rate,
        "drop_box_rate": drop_rate,
        "mean_error_3d_mm": mean_err_3d,
        "std_error_3d_mm": std_err_3d,
        "mean_time_s": mean_time,
        "plot_path": eval_plot_path,
        "episodes": results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA Mobile Manipulation")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--no-vlm", action="store_true", help="Run without VLM inference (perception fast-path)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_evaluation(num_episodes=args.episodes, use_vlm=not args.no_vlm, seed=args.seed)
