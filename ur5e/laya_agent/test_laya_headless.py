import mujoco
import time
import json
import random
from laya import Router

from robot_skills import (
    skill_hover_over_block, 
    skill_descend, 
    skill_close_gripper, 
    skill_lift
)
from perception import get_scene_state_json

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
        "instructions": "Given this JSON state, which skill should the robot execute next to grasp the block?",
        "criteria": ["HOVER", "DESCEND", "CLOSE_GRIPPER", "LIFT"]
    }
}

print("Starting Laya Autonomous Loop (Headless Sample)...")

for step_counter in range(15):
    # 1. Perception
    state_json = get_scene_state_json(model, data)
    state_dict = json.loads(state_json)
    
    # 2. Decision via Laya
    laya_response = router.predict(state_json, QUESTIONS, model="multilingual")
    
    chosen_skill = laya_response['answers']['skill_choice']['choice']
        
    print(f"[{step_counter}] STATE: dist_xy={state_dict['dist_xy_meters']}m, z={state_dict['height_above_block_meters']}m | DECISION: {chosen_skill}")
    print(f"       Raw Laya Response: {laya_response}\n")
    
    # 3. Execution (IK target generation)
    if chosen_skill == "HOVER":
        target_qpos = skill_hover_over_block(model, data, state_dict['block_x'], state_dict['block_y'])
    elif chosen_skill == "DESCEND":
        target_qpos = skill_descend(model, data, state_dict['block_x'], state_dict['block_y'])
    elif chosen_skill == "CLOSE_GRIPPER":
        target_qpos = skill_close_gripper(model, data)
    elif chosen_skill == "LIFT":
        target_qpos = skill_lift(model, data)
        
    # Apply the chosen skill for a short duration
    for _ in range(5):
        data.ctrl[:6] = data.qpos[:6] + 0.1 * (target_qpos - data.qpos[:6])
        mujoco.mj_step(model, data)
        
print("Sample finished.")
