import mujoco
import mujoco.viewer
import time
import json
import numpy as np
import requests

from robot_skills import solve_ik_vla, get_target_rot_down

print("Loading Ollama LLM (Level 3: Cartesian Positions)...")

print("Loading MuJoCo scene...")
model = mujoco.MjModel.from_xml_path('../ur5e_grasp_scene.xml')
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)
home_qpos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
data.qpos[:6] = home_qpos
mujoco.mj_forward(model, data)

# A simple tool schema matching the PDF
TOOL_PROMPT = """
You are a robotic arm task planner. Convert the user's natural language into the following JSON format. Do not include any other text.
{
  "action": "move_to_position",
  "x": float,
  "y": float,
  "z": float
}
"""

def call_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen3-vl:latest",
        "prompt": f"{TOOL_PROMPT}\nUser: {prompt}\nJSON:",
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(url, json=payload).json()
        return json.loads(response['response'])
    except Exception as e:
        print(f"Ollama generation failed: {e}")
        return None

print("\nStarting Level 3 Cartesian LLM control loop...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    time.sleep(2)
    
    # Simulate a user giving Cartesian commands
    user_prompts = [
        "Move the end effector to x=-0.1, y=0.5, z=0.4.",
        "Now move it down to z=0.2, keeping x=-0.1 and y=0.5.",
        "Move it to the right at x=0.2, y=0.5, z=0.2."
    ]
    
    for prompt in user_prompts:
        if not viewer.is_running():
            break
            
        print(f"\nUser: '{prompt}'")
        print("Waiting for Ollama (qwen3-vl:latest)...")
        
        decision = call_ollama(prompt)
        print(f"LLM produced: {decision}")
        
        if not decision or decision.get('action') != 'move_to_position':
            print("Invalid action chosen!")
            continue
            
        # 2. Extract the Cartesian target directly from LLM floats
        target_pos = np.array([decision['x'], decision['y'], decision['z']])
        print(f"MuJoCo: Controller running IK for target {target_pos}...")
        
        # 3. Controller converts Cartesian -> Joint Targets via IK
        # (This protects the robot: the LLM just gave coordinates, the IK does the math safely)
        target_qpos = solve_ik_vla(model, data, target_pos, get_target_rot_down())
        
        # Smooth interpolation to the target over 100 steps
        start_q = data.qpos[:6].copy()
        
        for i in range(100):
            alpha = i / 100.0
            interpolated_q = start_q * (1 - alpha) + target_qpos * alpha
            
            data.ctrl[:6] = interpolated_q
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)
            
        time.sleep(1) # Pause before next command

print("Simulation finished.")
