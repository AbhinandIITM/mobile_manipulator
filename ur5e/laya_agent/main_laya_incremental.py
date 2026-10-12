import mujoco
import mujoco.viewer
import time
import json
import os
os.environ["HF_HUB_OFFLINE"] = "1"
from laya import Router
import cv2
import numpy as np

from robot_skills import solve_ik_vla, get_target_rot_down
from perception import get_scene_state_json, detect_red_blob

print("Loading Laya Decision Engine (Incremental Cartesian Mode)...")
router = Router()

print("Loading MuJoCo scene...")
model = mujoco.MjModel.from_xml_path('../ur5e_grasp_scene.xml')
data = mujoco.MjData(model)

home_qpos = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]
mujoco.mj_resetData(model, data)
data.qpos[:6] = home_qpos
mujoco.mj_forward(model, data)

# The "Level 2" discrete incremental movement API from the PDF
QUESTIONS = {
    "x_axis": {
        "type": "choice",
        "instructions": "If target_dx_cm > 0, choose 'positive'. If target_dx_cm < 0, choose 'negative'. Otherwise choose 'hold'.",
        "criteria": ["positive", "negative", "hold"]
    },
    "y_axis": {
        "type": "choice",
        "instructions": "If target_dy_cm > 0, choose 'positive'. If target_dy_cm < 0, choose 'negative'. Otherwise choose 'hold'.",
        "criteria": ["positive", "negative", "hold"]
    },
    "z_axis": {
        "type": "choice",
        "instructions": "If dist_xy_meters > 0.05, choose 'hold'. If dist_xy_meters <= 0.05 and target_dz_cm < 0, choose 'negative'. If we need to lift, choose 'positive'.",
        "criteria": ["positive", "negative", "hold"]
    },
    "gripper": {
        "type": "choice",
        "instructions": "If height_above_block_meters <= 0.05 and dist_xy_meters <= 0.03, choose 'close'. Otherwise 'open'.",
        "criteria": ["open", "close"]
    }
}

print("Starting Laya Incremental Loop...")

renderer = mujoco.Renderer(model, height=480, width=640)

# Movement step size (1 cm per decision step)
STEP_SIZE = 0.01

class LoopState:
    def __init__(self):
        self.gripper_closed = False

loop_state = LoopState()

t_list = []
block_z_list = []
gripper_z_list = []

with mujoco.viewer.launch_passive(model, data) as viewer:
    time.sleep(2)
    step_counter = 0
    
    while viewer.is_running():
        # 1. Perception
        state_json = get_scene_state_json(model, data)
        state_dict = json.loads(state_json)
        
        # 2. Decision via LLM
        laya_response = router.predict(state_json, QUESTIONS, model="multilingual")
        ans = laya_response['answers']
        
        act_x = ans['x_axis']['choice']
        act_y = ans['y_axis']['choice']
        act_z = ans['z_axis']['choice']
        act_g = ans['gripper']['choice']
        
        print(f"[{step_counter}] STATE: dx={state_dict['target_dx_cm']} dy={state_dict['target_dy_cm']} dz={state_dict['target_dz_cm']} | "
              f"LLM: X={act_x}, Y={act_y}, Z={act_z}, Grip={act_g}")
              
        # 3. Translation to Cartesian incremental target (The "Controller" part from PDF)
        pinch_pos = data.site_xpos[model.site('pinch').id].copy()
        
        if act_x == "positive": pinch_pos[0] += STEP_SIZE
        elif act_x == "negative": pinch_pos[0] -= STEP_SIZE
        
        if act_y == "positive": pinch_pos[1] += STEP_SIZE
        elif act_y == "negative": pinch_pos[1] -= STEP_SIZE
        
        if act_z == "positive": pinch_pos[2] += STEP_SIZE
        elif act_z == "negative": pinch_pos[2] -= STEP_SIZE
        
        # Safety bound floor
        pinch_pos[2] = max(pinch_pos[2], 0.02)
        
        if act_g == "close":
            loop_state.gripper_closed = True
            
        target_qpos = solve_ik_vla(model, data, pinch_pos, get_target_rot_down())
        gripper_cmd = 255.0 if loop_state.gripper_closed else 0.0
        
        # Log
        t_list.append(data.time)
        block_z_list.append(data.xpos[model.body('target_block').id][2])
        gripper_z_list.append(data.site_xpos[model.site('pinch').id][2])
        
        # 4. MuJoCo Executor
        for _ in range(5):
            data.ctrl[:6] = data.qpos[:6] + 0.1 * (target_qpos - data.qpos[:6])
            data.ctrl[6] = gripper_cmd
            mujoco.mj_step(model, data)
            viewer.sync()
            
            renderer.update_scene(data, camera="d435i_color")
            img_rgb = renderer.render()
            (cx, cy), area, img_bgr, mask = detect_red_blob(img_rgb)
            cv2.imshow("Wrist Camera", img_bgr)
            cv2.waitKey(1)
            
        step_counter += 1

cv2.destroyAllWindows()
