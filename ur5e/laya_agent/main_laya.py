import mujoco
import mujoco.viewer
import time
import json
import random
import os
os.environ["HF_HUB_OFFLINE"] = "1"
from laya import Router

from robot_skills import (
    skill_hover_over_block, 
    skill_descend, 
    skill_close_gripper, 
    skill_lift
)
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
        # Laya takes the state string and the question, outputting the decision in milliseconds.
        laya_response = router.predict(state_json, QUESTIONS, model="multilingual")
        
        chosen_skill = laya_response['answers']['skill_choice']['choice']
        probs = laya_response['answers']['skill_choice']['probabilities']
            
        print(f"[{step_counter}] STATE: dist_xy={state_dict['dist_xy_meters']}m, z={state_dict['height_above_block_meters']}m | DECISION: {chosen_skill}")
        
        # Log data
        t_list.append(data.time)
        block_z_list.append(data.xpos[model.body('target_block').id][2])
        gripper_z_list.append(data.site_xpos[model.site('pinch').id][2])
        prob_hover_list.append(probs.get('HOVER', 0.0))
        prob_descend_list.append(probs.get('DESCEND', 0.0))
        prob_close_list.append(probs.get('CLOSE_GRIPPER', 0.0))
        prob_lift_list.append(probs.get('LIFT', 0.0))
        
        # 3. Execution (IK target generation)
        if chosen_skill == "HOVER":
            target_qpos = skill_hover_over_block(model, data, state_dict['block_x'], state_dict['block_y'])
        elif chosen_skill == "DESCEND":
            target_qpos = skill_descend(model, data, state_dict['block_x'], state_dict['block_y'])
        elif chosen_skill == "CLOSE_GRIPPER":
            target_qpos = skill_close_gripper(model, data)
        elif chosen_skill == "LIFT":
            target_qpos = skill_lift(model, data)
            
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
