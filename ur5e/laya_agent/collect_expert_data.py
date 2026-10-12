"""
Expert Data Collector for Laya Fine-Tuning (Behavioral Cloning)
================================================================
Uses the same IK controller from simulate_vision_grasp.py but runs in
FAST headless mode by simulating multiple sub-steps per control step
(not real-time, but correct physics).

Key fix: runs N_SUBSTEPS physics steps per control update, matching
the step-rate that the viewer-based expert uses.

Output: laya_expert_dataset.jsonl
        laya_expert_stats.png

Usage:
    python collect_expert_data.py --episodes 200
"""

import os, sys, json, argparse, time
import numpy as np
import mujoco

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UR5E_DIR   = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(UR5E_DIR, "ur5e_grasp_scene.xml")
sys.path.insert(0, UR5E_DIR)

from controllers import CartesianIKController

# Action-space bounds
DX_MIN, DX_MAX = -4, 5
DY_MIN, DY_MAX = -4, 5
DZ_MIN, DZ_MAX = -4, 5

TARGET_ROT_DOWN = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=float)

# Physics sub-steps per control update (matches ~500Hz physics / 50Hz control = 10)
N_SUBSTEPS = 10


def build_state(model, data):
    box_pos   = data.geom_xpos[model.geom("block_geom").id]
    pinch_pos = data.site_xpos[model.site("pinch").id]
    dist_xy   = float(np.linalg.norm(box_pos[:2] - pinch_pos[:2]))
    dist_z    = float(pinch_pos[2] - box_pos[2])
    gripper_open = bool(data.ctrl[6] < 120)
    dx_cm = int(np.clip(round((box_pos[0] - pinch_pos[0]) * 100), DX_MIN, DX_MAX))
    dy_cm = int(np.clip(round((box_pos[1] - pinch_pos[1]) * 100), DY_MIN, DY_MAX))
    dz_cm = int(np.clip(round((box_pos[2] - pinch_pos[2]) * 100), DZ_MIN, DZ_MAX))
    return {
        "dist_xy_meters":            round(dist_xy, 3),
        "height_above_block_meters": round(dist_z, 3),
        "gripper_is_open":           gripper_open,
        "target_dx_cm":              dx_cm,
        "target_dy_cm":              dy_cm,
        "target_dz_cm":              dz_cm,
    }


def build_expert_action(model, data, target_pos, gripper_cmd):
    pinch_pos = data.site_xpos[model.site("pinch").id]
    dx_cm = int(np.clip(round((target_pos[0] - pinch_pos[0]) * 100), DX_MIN, DX_MAX))
    dy_cm = int(np.clip(round((target_pos[1] - pinch_pos[1]) * 100), DY_MIN, DY_MAX))
    dz_cm = int(np.clip(round((target_pos[2] - pinch_pos[2]) * 100), DZ_MIN, DZ_MAX))
    return {
        "dx_cm":         dx_cm,
        "dy_cm":         dy_cm,
        "dz_cm":         dz_cm,
        "close_gripper": bool(gripper_cmd > 128),
    }


def reset_episode(model, data, home_qpos):
    mujoco.mj_resetData(model, data)
    data.qpos[:6] = home_qpos.copy()
    data.ctrl[:6] = home_qpos.copy()
    data.ctrl[6]  = 0.0

    jnt_id = model.body_jntadr[model.body("target_block").id]
    qadr   = model.jnt_qposadr[jnt_id]
    data.qpos[qadr]   = -0.25 + np.random.uniform(0.0, 0.30)  # X in [-0.25, 0.05]
    data.qpos[qadr+1] =  0.40 + np.random.uniform(0.0, 0.20)  # Y in [0.40, 0.60]
    data.qpos[qadr+2] =  0.025
    # Freejoint: qpos[qadr+3:qadr+7] = identity quaternion (already set by resetData)
    mujoco.mj_forward(model, data)


def run_episode(model, data, controller, max_ctrl_steps=2000):
    """
    Runs one expert grasp episode.
    Each 'step' = N_SUBSTEPS physics steps at 500Hz = ~20ms control loop.
    Returns: (list of {state, action} dicts, success bool)
    """
    dt_ctrl = model.opt.timestep * N_SUBSTEPS   # ~0.02s per control update

    samples = []
    phase            = 0
    phase_start_step = 0
    target_pos       = np.array([-0.1, 0.5, 0.5])
    gripper_cmd      = 0.0
    success          = False

    # Lock on block position at episode start (updated each step)
    def get_block_xy():
        return data.xpos[model.body("target_block").id][:2].copy()

    t_ctrl = 0  # control step counter

    for ctrl_step in range(max_ctrl_steps):
        t = ctrl_step * dt_ctrl
        bxy     = get_block_xy()
        pinch   = data.site_xpos[model.site("pinch").id]

        # Expert state machine
        if phase == 0:
            # Sinusoidal scan
            target_pos[0] = -0.15 + 0.25 * np.sin(t * 1.5)
            target_pos[1] =  0.50 + 0.15 * np.cos(t * 1.0)
            target_pos[2] =  0.50
            if np.linalg.norm(target_pos[:2] - bxy) < 0.08:
                target_pos[0] = float(bxy[0])
                target_pos[1] = float(bxy[1])
                phase = 1

        elif phase == 1:
            # Hover above block at Z=0.3
            target_pos[0] = float(bxy[0])
            target_pos[1] = float(bxy[1])
            target_pos[2] = 0.3
            dist_xy = np.linalg.norm(pinch[:2] - target_pos[:2])
            if dist_xy < 0.015:
                phase = 2

        elif phase == 2:
            # Descend: move target down gradually, wait for actual arm to follow
            target_pos[2] = max(target_pos[2] - 0.002, 0.025)
            # Only transition when arm has ACTUALLY come down close to block
            if pinch[2] <= 0.06 and target_pos[2] <= 0.025:
                phase = 3
                phase_start_step = ctrl_step

        elif phase == 3:
            # Hold position, close gripper
            gripper_cmd = 255.0
            if ctrl_step - phase_start_step > 75:   # ~1.5s at 50Hz
                phase = 4
                phase_start_step = ctrl_step

        elif phase == 4:
            # Lift
            target_pos[2] = 0.35
            gripper_cmd   = 255.0
            block_z = data.xpos[model.body("target_block").id][2]
            if block_z > 0.06:
                success = True
                # Collect a few more lift samples
                for _ in range(20):
                    state  = build_state(model, data)
                    action = build_expert_action(model, data, target_pos, gripper_cmd)
                    samples.append({"state": state, "action": action})
                    q_cmd, _ = controller.compute_control(
                        data=data, target_pos=target_pos, target_vel=np.zeros(3),
                        target_rot=TARGET_ROT_DOWN, target_angvel=np.zeros(3), dt=model.opt.timestep)
                    data.ctrl[:6] = q_cmd
                    data.ctrl[6]  = gripper_cmd
                    for _ in range(N_SUBSTEPS):
                        mujoco.mj_step(model, data)
                break

        # IK command
        q_cmd, _ = controller.compute_control(
            data=data,
            target_pos=target_pos,
            target_vel=np.zeros(3),
            target_rot=TARGET_ROT_DOWN,
            target_angvel=np.zeros(3),
            dt=model.opt.timestep,
        )
        data.ctrl[:6] = q_cmd
        data.ctrl[6]  = gripper_cmd

        # Record state/action every control step
        state  = build_state(model, data)
        action = build_expert_action(model, data, target_pos, gripper_cmd)
        samples.append({"state": state, "action": action})

        # Advance physics N_SUBSTEPS times
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(model, data)

    return samples, success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",  type=int,   default=200)
    parser.add_argument("--max-steps", type=int,   default=2000,
                        help="Max control steps per episode (2000 @ 50Hz = 40s)")
    parser.add_argument("--out", type=str,
                        default=os.path.join(SCRIPT_DIR, "laya_expert_dataset.jsonl"))
    args = parser.parse_args()

    print(f"Loading scene: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)

    home_qpos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    reset_episode(model, data, home_qpos)

    controller = CartesianIKController(
        model=model, site_name="pinch", home_qpos=home_qpos,
        kp_pos=10.0, kp_ori=10.0, k_null=1.0, damping=0.05, max_qdot=1.5,
    )

    total_samples  = 0
    success_count  = 0
    samples_per_ep = []
    success_flags  = []

    print(f"\nCollecting {args.episodes} expert episodes --> {args.out}")
    print("-" * 60)

    with open(args.out, "w") as fout:
        for ep in range(args.episodes):
            reset_episode(model, data, home_qpos)
            controller.reset(home_qpos)

            t0 = time.time()
            samples, success = run_episode(model, data, controller, max_ctrl_steps=args.max_steps)
            elapsed = time.time() - t0

            for s in samples:
                fout.write(json.dumps(s) + "\n")

            ep_samples     = len(samples)
            total_samples += ep_samples
            success_count += int(success)
            samples_per_ep.append(ep_samples)
            success_flags.append(success)

            status = "[OK]" if success else "[X] "
            print(f"  Ep {ep+1:>4}/{args.episodes}  {status}  "
                  f"{ep_samples:>4} samples  ({elapsed:.1f}s)  "
                  f"[Total: {total_samples:,}]")

    success_rate = success_count / args.episodes * 100
    print(f"\n{'='*60}")
    print(f"  Episodes:      {args.episodes}")
    print(f"  Successes:     {success_count}  ({success_rate:.1f}%)")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Dataset saved: {args.out}")
    print(f"{'='*60}")

    # Stats plot
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle("Expert Data Collection Statistics", fontsize=14, fontweight="bold")

    ep_range = range(1, args.episodes + 1)
    colors   = ["green" if s else "red" for s in success_flags]
    axs[0].bar(ep_range, samples_per_ep, color=colors, alpha=0.7, width=0.8)
    axs[0].set_xlabel("Episode"); axs[0].set_ylabel("Samples")
    axs[0].set_title("Samples Per Episode  (green=success, red=failure)")
    axs[0].grid(axis="y", alpha=0.4)

    cumulative = np.cumsum(samples_per_ep)
    rolling_sr = [
        sum(success_flags[max(0,i-19):i+1]) / min(i+1,20) * 100
        for i in range(len(success_flags))
    ]
    ax2 = axs[1].twinx()
    axs[1].plot(ep_range, cumulative, "b-",  label="Cumulative Samples")
    ax2.plot(ep_range,    rolling_sr, "orange", linestyle="--", label="Rolling Success % (20ep)")
    axs[1].set_xlabel("Episode"); axs[1].set_ylabel("Cumulative Samples", color="blue")
    ax2.set_ylabel("Success Rate (%)", color="orange")
    axs[1].set_title("Cumulative Dataset Size & Rolling Success Rate")
    axs[1].grid(alpha=0.4)
    lines1, labels1 = axs[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axs[1].legend(lines1+lines2, labels1+labels2, loc="upper left")

    plt.tight_layout()
    stats_path = os.path.join(SCRIPT_DIR, "laya_expert_stats.png")
    plt.savefig(stats_path, dpi=150); plt.close()
    print(f"  Stats plot:    {stats_path}")


if __name__ == "__main__":
    main()
