import mujoco
import mujoco.viewer
import time
import json
import random
import os
os.environ["HF_HUB_OFFLINE"] = "1"
from laya import Router
import cv2
import numpy as np

from robot_skills import (
    skill_hover_over_block, 
    skill_descend, 
    skill_close_gripper, 
    skill_lift
)
from perception import get_scene_state_json, detect_red_blob

print("Loading Laya Decision Engine...")
router = Router()

print("Loading MuJoCo scene...")
model = mujoco.MjModel.from_xml_path('../ur5e_grasp_scene.xml')
data = mujoco.MjData(model)

home_qpos = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]
mujoco.mj_resetData(model, data)
data.qpos[:6] = home_qpos
mujoco.mj_forward(model, data)

QUESTIONS = {
    "skill_choice": {
        "type": "choice",
        "instructions": (
            "You are a robotic control agent guiding a UR5e arm. Evaluate the JSON state and choose the next skill.\n"
            "Rules:\n"
            "1. If height_above_block_meters > 0.05, you must choose DESCEND.\n"
            "2. If height_above_block_meters <= 0.05 and gripper_open is true, you must choose CLOSE_GRIPPER.\n"
            "3. If height_above_block_meters <= 0.05 and gripper_open is false, you must choose LIFT.\n"
            "Choose exactly one."
        ),
        "criteria": ["HOVER", "DESCEND", "CLOSE_GRIPPER", "LIFT"]
    }
}

print("Starting Laya Autonomous Loop...")

renderer = mujoco.Renderer(model, height=480, width=640)

# Logging
t_list = []
block_x_list = []; block_y_list = []; block_z_list = []
gripper_x_list = []; gripper_y_list = []; gripper_z_list = []
prob_hover_list = []; prob_descend_list = []; prob_close_list = []; prob_lift_list = []
cmd_close = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    
    time.sleep(2)
    step_counter = 0
    
    while viewer.is_running():
        # 1. Perception
        state_json = get_scene_state_json(model, data)
        state_dict = json.loads(state_json)
        
        # 2. Decision via Laya System 1 Model
        laya_response = router.predict(state_json, QUESTIONS, model="multilingual")
        
        chosen_skill = laya_response['answers']['skill_choice']['choice']
        probs = laya_response['answers']['skill_choice']['probabilities']
            
        print(f"[{step_counter}] STATE: dist_xy={state_dict['dist_xy_meters']}m, z={state_dict['height_above_block_meters']}m | DECISION: {chosen_skill}")
        
        # Log data
        t_list.append(data.time)
        
        block_pos = data.xpos[model.body('target_block').id]
        pinch_pos = data.site_xpos[model.site('pinch').id]
        
        block_x_list.append(block_pos[0]); block_y_list.append(block_pos[1]); block_z_list.append(block_pos[2])
        gripper_x_list.append(pinch_pos[0]); gripper_y_list.append(pinch_pos[1]); gripper_z_list.append(pinch_pos[2])
        
        prob_hover_list.append(probs.get('HOVER', 0.0))
        prob_descend_list.append(probs.get('DESCEND', 0.0))
        prob_close_list.append(probs.get('CLOSE_GRIPPER', 0.0))
        prob_lift_list.append(probs.get('LIFT', 0.0))
        
        cmd_close.append(1.0 if chosen_skill in ["CLOSE_GRIPPER", "LIFT"] else 0.0)
        
        # 3. Execution (IK target generation)
        if chosen_skill == "HOVER":
            target_qpos = skill_hover_over_block(model, data, state_dict['block_x'], state_dict['block_y'])
        elif chosen_skill == "DESCEND":
            target_qpos = skill_descend(model, data, state_dict['block_x'], state_dict['block_y'])
        elif chosen_skill == "CLOSE_GRIPPER":
            target_qpos = skill_close_gripper(model, data)
        elif chosen_skill == "LIFT":
            target_qpos = skill_lift(model, data)
            
        # Apply the chosen skill for a short duration (open-loop step)
        for _ in range(5):
            data.ctrl[:6] = data.qpos[:6] + 0.1 * (target_qpos - data.qpos[:6])
            # Handle gripper based on skill
            if chosen_skill in ["CLOSE_GRIPPER", "LIFT"]:
                data.ctrl[6] = 255.0
            else:
                data.ctrl[6] = 0.0
                
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Render and display camera safely
            renderer.update_scene(data, camera="d435i_color")
            img_rgb = renderer.render()
            (cx, cy), area, img_bgr, mask = detect_red_blob(img_rgb)
            cv2.imshow("Wrist Camera", img_bgr)
            cv2.waitKey(1)
            
        step_counter += 1
            
cv2.destroyAllWindows()

# --- Plotting and Metrics ---
if len(t_list) > 10:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Laya Autonomous VLA Grasping Analysis (Discrete Skills)", fontsize=16, fontweight="bold")
    
    # 1. 3D Trajectory (Top-Left)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(block_x_list, block_y_list, block_z_list, "r--", label="Block", linewidth=1.5)
    ax1.plot(gripper_x_list, gripper_y_list, gripper_z_list, "b-", label="Gripper", linewidth=1.5)
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
    
    # 3. Laya Skill Probabilities (Bottom-Left)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t_list, prob_hover_list, label="HOVER Prob", color='gray')
    ax3.plot(t_list, prob_descend_list, label="DESCEND Prob", color='orange')
    ax3.plot(t_list, prob_close_list, label="CLOSE_GRIPPER Prob", color='green')
    ax3.plot(t_list, prob_lift_list, label="LIFT Prob", color='purple')
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Probability")
    ax3.set_title("Laya Skill Probabilities Over Time")
    ax3.legend()
    ax3.grid(True)
    
    # 4. Gripper Actuation (Bottom-Right)
    ax4 = fig.add_subplot(2, 2, 4)
    cmd_close_scaled = [c * 255.0 for c in cmd_close]
    ax4.plot(t_list, cmd_close_scaled, label="Gripper CMD (0-255)", color='black')
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Command Signal")
    ax4.set_title("Gripper Actuation")
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("laya_grasping_analysis.png")
    print("Saved metrics plot to laya_grasping_analysis.png")
