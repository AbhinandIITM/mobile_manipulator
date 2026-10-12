import mujoco
import mujoco.viewer
import time
import json
import random
import os
os.environ["HF_HUB_OFFLINE"] = "1"
from laya import Router

from robot_skills import solve_ik_vla, get_target_rot_down
from perception import get_scene_state_json, detect_red_blob

print("Loading Laya Decision Engine...")
# Initialize Laya Router
router = Router()

print("Loading MuJoCo scene...")
# Note: Since this script is inside laya_agent, the xml is up one directory.
model = mujoco.MjModel.from_xml_path('../ur5e_grasp_scene.xml')
data = mujoco.MjData(model)

home_qpos = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]
mujoco.mj_resetData(model, data)
data.qpos[:6] = home_qpos
mujoco.mj_forward(model, data)

# Laya prompts
from pydantic import BaseModel, Field
from laya.structured import decide

class ContinuousAction(BaseModel):
    dx_cm: int = Field(description="Must match 'target_dx_cm' from the state exactly.", ge=-4, le=5)
    dy_cm: int = Field(description="Must match 'target_dy_cm' from the state exactly.", ge=-4, le=5)
    dz_cm: int = Field(description="Must match 'target_dz_cm' from the state exactly.", ge=-4, le=5)
    close_gripper: bool = Field(description="True if 'dist_xy_meters' < 0.05 and 'height_above_block_meters' < 0.05, else False.")

import cv2
import numpy as np

print("Starting Laya Autonomous Loop...")

# Setup camera rendering using the safe Renderer API
renderer = mujoco.Renderer(model, height=480, width=640)

# Logging for plots
t_list = []
block_x_list = []
block_y_list = []
block_z_list = []
gripper_x_list = []
gripper_y_list = []
gripper_z_list = []
cmd_dx = []
cmd_dy = []
cmd_dz = []
cmd_close = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    
    time.sleep(2)
    step_counter = 0
    
    while viewer.is_running():
        # 1. Perception
        state_json = get_scene_state_json(model, data)
        state_dict = json.loads(state_json)
        
        # 2. Decision via Laya System 1 Model
        action = decide(router, state_json, schema=ContinuousAction, model="multilingual")
        
        print(f"[{step_counter}] Z={state_dict['height_above_block_meters']}m | ACTION: {action}")
        
        # Log data
        t_list.append(data.time)
        block_pos = data.xpos[model.body('target_block').id]
        pinch_pos = data.site_xpos[model.site('pinch').id]
        
        block_x_list.append(block_pos[0])
        block_y_list.append(block_pos[1])
        block_z_list.append(block_pos[2])
        
        gripper_x_list.append(pinch_pos[0])
        gripper_y_list.append(pinch_pos[1])
        gripper_z_list.append(pinch_pos[2])
        
        cmd_dx.append(action["dx_cm"])
        cmd_dy.append(action["dy_cm"])
        cmd_dz.append(action["dz_cm"])
        cmd_close.append(1.0 if action["close_gripper"] else 0.0)
        
        # 3. Execution (Continuous IK target generation)
        dx = action["dx_cm"] / 100.0
        dy = action["dy_cm"] / 100.0
        dz = action["dz_cm"] / 100.0
        
        # Get current gripper position
        pinch_pos = data.site_xpos[model.site('pinch').id].copy()
        
        # Apply deltas
        target_pos = pinch_pos + np.array([dx, dy, dz])
        
        # Keep Z above floor
        if target_pos[2] < 0.02:
            target_pos[2] = 0.02
            
        target_qpos = solve_ik_vla(model, data, target_pos, get_target_rot_down())
        gripper_cmd = 255.0 if action["close_gripper"] else 0.0
            
        # # Disturbance! Every 50 steps, someone kicks the block away!
        # if step_counter > 0 and step_counter % 50 == 0:
        #     print(">>> DISTURBANCE! Block was kicked away! <<<")
        #     body_id = model.body('target_block').id
        #     jnt_id = model.body_jntadr[body_id]
        #     qpos_adr = model.jnt_qposadr[jnt_id]
        #     data.qpos[qpos_adr] += random.uniform(-0.2, 0.2) # X kick
        #     data.qpos[qpos_adr + 1] += random.uniform(-0.2, 0.2) # Y kick
        #     mujoco.mj_forward(model, data)
        
        # Apply the chosen skill for a short duration (open-loop step)
        for _ in range(5):
            data.ctrl[:6] = data.qpos[:6] + 0.1 * (target_qpos - data.qpos[:6])
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Render and display camera safely
            renderer.update_scene(data, camera="d435i_color")
            img_rgb = renderer.render()
            
            # Draw CV bounding box on the image using our perception pipeline
            _, _, img_bgr, _ = detect_red_blob(img_rgb)
            
            cv2.imshow("Wrist Camera", img_bgr)
            cv2.waitKey(1)
            
    
cv2.destroyAllWindows()

# --- Plotting and Metrics ---
if len(t_list) > 10:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Laya Autonomous VLA Grasping Analysis", fontsize=16, fontweight='bold')
    
    # 1. 3D Trajectory (Top-Left)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(block_x_list, block_y_list, block_z_list, label="Block Position", color='red', linestyle='--')
    ax1.plot(gripper_x_list, gripper_y_list, gripper_z_list, label="Gripper Trajectory", color='blue')
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.set_zlabel("Z [m]")
    ax1.set_title("3D Search & Grasp Path")
    ax1.legend()
    
    # 2. Z Elevation (Top-Right)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t_list, block_z_list, label="Object Z", color='green', linewidth=2)
    ax2.plot(t_list, gripper_z_list, label="Actual EEF Z", color='blue', linewidth=2)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Z [m]")
    ax2.set_title("Z Elevation (EEF vs Object)")
    ax2.grid(True)
    ax2.legend()
    
    # 3. Laya Commands (Bottom-Left)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t_list, cmd_dx, label="Cmd dX [cm]", color='m')
    ax3.plot(t_list, cmd_dy, label="Cmd dY [cm]", color='c')
    ax3.plot(t_list, cmd_dz, label="Cmd dZ [cm]", color='y')
    ax3.axhline(0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Delta [cm]")
    ax3.set_title("Laya LLM Output Commands")
    ax3.grid(True)
    ax3.legend()
    
    # 4. Gripper Actuation (Bottom-Right)
    ax4 = fig.add_subplot(2, 2, 4)
    # Scale binary close_gripper (0/1) to 0-255 range for plot visual parity
    cmd_close_scaled = [c * 255.0 for c in cmd_close]
    ax4.plot(t_list, cmd_close_scaled, label="Gripper CMD (0-255)", color='black')
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Command Signal")
    ax4.set_title("Gripper Actuation")
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("laya_continuous_analysis.png")
    print("Saved metrics plot to laya_continuous_analysis.png")
