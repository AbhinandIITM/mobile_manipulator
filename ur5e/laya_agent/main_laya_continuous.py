import mujoco
import mujoco.viewer
import time
import json
import random
import os
os.environ["HF_HUB_OFFLINE"] = "1"
from laya import Router

from robot_skills import solve_ik_vla, get_target_rot_down
from perception import get_scene_state_json

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
    dx_cm: int = Field(description="Target delta movement along X axis in cm (-4 to 5). Negative is left.", ge=-4, le=5)
    dy_cm: int = Field(description="Target delta movement along Y axis in cm (-4 to 5). Negative is back.", ge=-4, le=5)
    dz_cm: int = Field(description="Target delta movement along Z axis in cm (-4 to 5). Negative is down.", ge=-4, le=5)
    close_gripper: bool = Field(description="True if gripper should close (when near the block).")

import cv2
import numpy as np

print("Starting Laya Autonomous Loop...")

# Setup camera rendering using the safe Renderer API
renderer = mujoco.Renderer(model, height=480, width=640)

# Logging for plots
t_list = []
block_z_list = []
gripper_z_list = []
prob_hover_list = []
prob_descend_list = []
prob_close_list = []
prob_lift_list = []

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
        block_z_list.append(data.xpos[model.body('target_block').id][2])
        gripper_z_list.append(data.site_xpos[model.site('pinch').id][2])
        prob_hover_list.append(0.0) # Dummy for plot compatibility
        prob_descend_list.append(0.0)
        prob_close_list.append(1.0 if action["close_gripper"] else 0.0)
        prob_lift_list.append(0.0)
        
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
            cv2.imshow("Wrist Camera", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            
    
cv2.destroyAllWindows()

# --- Plotting and Metrics ---
if len(t_list) > 10:
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 1. Plot trajectories (Block vs Gripper Z)
    axs[0].plot(t_list, block_z_list, label="Block Z", color='red')
    axs[0].plot(t_list, gripper_z_list, label="Gripper Z", color='blue', linestyle='--')
    axs[0].set_ylabel("Height (m)")
    axs[0].set_title("Laya Autonomous Grasping - Z Trajectories")
    axs[0].legend()
    axs[0].grid(True)
    
    # 2. Plot skill probabilities
    axs[1].plot(t_list, prob_hover_list, label="HOVER Prob", color='gray')
    axs[1].plot(t_list, prob_descend_list, label="DESCEND Prob", color='orange')
    axs[1].plot(t_list, prob_close_list, label="CLOSE_GRIPPER Prob", color='green')
    axs[1].plot(t_list, prob_lift_list, label="LIFT Prob", color='purple')
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Probability")
    axs[1].set_title("Laya Skill Probabilities Over Time")
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("laya_grasping_analysis.png")
    print("Saved metrics plot to laya_grasping_analysis.png")
