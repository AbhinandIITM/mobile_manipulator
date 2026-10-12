"""
Laya + MLP Policy Grasping Agent
=================================
Uses the trained MLP behavioral cloning policy (grasp_policy.pt) instead of
Laya's zero-shot LLM for continuous movement commands. The MLP runs in <0.1ms
vs Laya's 30ms, and is trained on 575k expert grasp trajectories.

OpenCV wrist camera feed with red-blob detection overlaid.
Saves laya_mlp_analysis.png on exit.

Usage:
    python main_mlp_policy.py
"""

import os, sys, json, time
import numpy as np
import mujoco
import mujoco.viewer
import cv2
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UR5E_DIR   = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(UR5E_DIR, "ur5e_grasp_scene.xml")
sys.path.insert(0, UR5E_DIR)

from robot_skills import solve_ik_vla, get_target_rot_down
from perception import get_scene_state_json, detect_red_blob


# ─── MLP Policy (must match train_policy.py architecture) ─────────────────────
class GraspPolicyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(6, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.xyz_head     = nn.Linear(128, 3)
        self.gripper_head = nn.Linear(128, 1)

    def forward(self, x):
        feat    = self.backbone(x)
        xyz     = self.xyz_head(feat)
        gripper = torch.sigmoid(self.gripper_head(feat))
        return xyz, gripper


def load_policy(weights_path):
    model = GraspPolicyMLP()
    ckpt  = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded policy  (val_loss={ckpt['val_loss']:.4f}  "
          f"xyz_mae={ckpt['xyz_mae']:.3f}cm  "
          f"grip_acc={ckpt['grip_acc']:.1f}%)")
    return model


def state_to_tensor(state_dict):
    """Convert state dict to the 6-element float tensor the MLP expects."""
    s = state_dict
    vec = np.array([
        s["dist_xy_meters"],
        s["height_above_block_meters"],
        float(s["gripper_is_open"]),
        float(s["target_dx_cm"]),
        float(s["target_dy_cm"]),
        float(s["target_dz_cm"]),
    ], dtype=np.float32)
    return torch.from_numpy(vec).unsqueeze(0)  # (1, 6)


# ─── Load everything ──────────────────────────────────────────────────────────
policy_path = os.path.join(SCRIPT_DIR, "grasp_policy.pt")
print("Loading MLP policy...")
policy = load_policy(policy_path)

print("Loading MuJoCo scene...")
mujoco_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(mujoco_model)

home_qpos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
data.qpos[:6] = home_qpos
data.ctrl[:6] = home_qpos
data.ctrl[6]  = 0.0
mujoco.mj_forward(mujoco_model, data)

renderer = mujoco.Renderer(mujoco_model, 240, 320)

# ─── Logging lists ────────────────────────────────────────────────────────────
t_list       = []
block_x_list = []; block_y_list = []; block_z_list = []
gripper_x_list = []; gripper_y_list = []; gripper_z_list = []
cmd_dx = []; cmd_dy = []; cmd_dz = []; cmd_close = []
step_counter = 0

# ─── Main loop ────────────────────────────────────────────────────────────────
print("Starting MLP Policy Loop...")
print("(Close the viewer window to save analysis plots)")
time.sleep(1)

# Simple namespace to hold mutable loop state (gripper latch)
class _State: pass
main_loop_state = _State()

with mujoco.viewer.launch_passive(mujoco_model, data) as viewer:
    viewer.cam.azimuth   = 110.0
    viewer.cam.elevation = -20.0
    viewer.cam.distance  = 1.5
    viewer.cam.lookat[:] = [-0.1, 0.5, 0.1]

    while viewer.is_running():
        # 1. Perception
        state_json = get_scene_state_json(mujoco_model, data)
        state_dict = json.loads(state_json)

        # 2. MLP Policy inference (microseconds, no LLM needed!)
        with torch.no_grad():
            state_tensor = state_to_tensor(state_dict)
            pred_xyz, pred_grip = policy(state_tensor)

        dx_cm = float(pred_xyz[0, 0].item())
        dy_cm = float(pred_xyz[0, 1].item())
        dz_cm = float(pred_xyz[0, 2].item())
        
        # Clamp to action space bounds
        dx_cm = np.clip(dx_cm, -4, 5)
        dy_cm = np.clip(dy_cm, -4, 5)
        dz_cm = np.clip(dz_cm, -4, 5)
        
        close_gripper = pred_grip[0, 0].item() > 0.5

        print(f"[{step_counter}] Z={state_dict['height_above_block_meters']:.3f}m | "
              f"dx={dx_cm:.1f} dy={dy_cm:.1f} dz={dz_cm:.1f} grip={'CLOSE' if close_gripper else 'OPEN'}")

        # 3. Log
        t_list.append(data.time)
        block_pos = data.xpos[mujoco_model.body('target_block').id]
        pinch_pos = data.site_xpos[mujoco_model.site('pinch').id]
        block_x_list.append(block_pos[0]); block_y_list.append(block_pos[1]); block_z_list.append(block_pos[2])
        gripper_x_list.append(pinch_pos[0]); gripper_y_list.append(pinch_pos[1]); gripper_z_list.append(pinch_pos[2])
        cmd_dx.append(dx_cm); cmd_dy.append(dy_cm); cmd_dz.append(dz_cm)
        cmd_close.append(1.0 if close_gripper else 0.0)

        # 4. Execute: MLP output = desired offset FROM block to gripper target
        # i.e. target = block_position + MLP_delta
        # This matches how the expert dataset was generated (block - pinch -> desired motion)
        block_pos_world = data.xpos[mujoco_model.body('target_block').id].copy()
        target_pos = block_pos_world + np.array([dx_cm / 100.0, dy_cm / 100.0, dz_cm / 100.0])
        target_pos[2] = max(target_pos[2], 0.02)

        # Latch gripper once closed (don't oscillate)
        if not hasattr(main_loop_state, 'gripper_latched'):
            main_loop_state.gripper_latched = False
        if close_gripper:
            main_loop_state.gripper_latched = True
        gripper_cmd = 255.0 if main_loop_state.gripper_latched else 0.0

        target_qpos = solve_ik_vla(mujoco_model, data, target_pos, get_target_rot_down())

        # 5. Step simulation
        for _ in range(5):
            data.ctrl[:6] = data.qpos[:6] + 0.1 * (target_qpos - data.qpos[:6])
            data.ctrl[6]  = gripper_cmd
            mujoco.mj_step(mujoco_model, data)
            viewer.sync()

            renderer.update_scene(data, camera="d435i_color")
            img_rgb = renderer.render()
            _, _, img_bgr, _ = detect_red_blob(img_rgb)
            cv2.imshow("Wrist Camera [MLP Policy]", img_bgr)
            cv2.waitKey(1)

        step_counter += 1

cv2.destroyAllWindows()

# ─── Analysis Plot ────────────────────────────────────────────────────────────
if len(t_list) > 10:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("MLP Behavioral Cloning Policy - Grasping Analysis", fontsize=16, fontweight="bold")

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(block_x_list, block_y_list, block_z_list, "r--", label="Block", linewidth=1.5)
    ax1.plot(gripper_x_list, gripper_y_list, gripper_z_list, "b-", label="Gripper", linewidth=1.5)
    ax1.set_xlabel("X [m]"); ax1.set_ylabel("Y [m]"); ax1.set_zlabel("Z [m]")
    ax1.set_title("3D Search & Grasp Path"); ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t_list, block_z_list,   "g-",  label="Object Z",   linewidth=2)
    ax2.plot(t_list, gripper_z_list, "b-",  label="Actual EEF Z", linewidth=2)
    ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Z [m]")
    ax2.set_title("Z Elevation (EEF vs Object)"); ax2.legend(); ax2.grid(True)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t_list, cmd_dx, "m-",  label="Cmd dX [cm]", alpha=0.7)
    ax3.plot(t_list, cmd_dy, "c-",  label="Cmd dY [cm]", alpha=0.7)
    ax3.plot(t_list, cmd_dz, "y-",  label="Cmd dZ [cm]", alpha=0.7)
    ax3.axhline(0, color="k", linestyle="--", linewidth=1)
    ax3.set_xlabel("Time [s]"); ax3.set_ylabel("Delta [cm]")
    ax3.set_title("MLP Policy Output Commands"); ax3.legend(); ax3.grid(True)

    ax4 = fig.add_subplot(2, 2, 4)
    cmd_close_scaled = [c * 255.0 for c in cmd_close]
    ax4.plot(t_list, cmd_close_scaled, "k-", label="Gripper CMD (0-255)")
    ax4.set_xlabel("Time [s]"); ax4.set_ylabel("Command Signal")
    ax4.set_title("Gripper Actuation"); ax4.legend(); ax4.grid(True)

    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    out_path = os.path.join(SCRIPT_DIR, "laya_mlp_analysis.png")
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"\nSaved analysis plot to: {out_path}")
