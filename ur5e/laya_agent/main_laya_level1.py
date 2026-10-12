import mujoco
import mujoco.viewer
import time
import json
import numpy as np
import os
os.environ["HF_HUB_OFFLINE"] = "1"
from laya import Router

print("Loading Laya Decision Engine (Level 1: Predefined Poses)...")
router = Router()

print("Loading MuJoCo scene...")
model = mujoco.MjModel.from_xml_path('../ur5e_grasp_scene.xml')
data = mujoco.MjData(model)

# 1. Define the Robot API (Predefined Joint Poses)
NAMED_POSES = {
    "home":  np.array([-1.57, -1.57,  1.57, -1.57, -1.57, 0.0]),
    "ready": np.array([-1.57, -1.0,   1.0,  -1.57, -1.57, 0.0]),
    "left":  np.array([-0.5,  -1.0,   1.0,  -1.57, -1.57, 0.0]),
    "right": np.array([-2.5,  -1.0,   1.0,  -1.57, -1.57, 0.0])
}

# Move to home initially
mujoco.mj_resetData(model, data)
data.qpos[:6] = NAMED_POSES["home"]
mujoco.mj_forward(model, data)

# 2. Define the LLM tools (Jev / Laya choice)
QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": "The user wants to move the robot. Choose the correct predefined pose.",
        "criteria": ["home", "ready", "left", "right"]
    }
}

print("\nStarting basic LLM control loop...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    time.sleep(2)
    
    # 3. Simulate a user giving a high-level command
    user_prompts = [
        "Put the robot in the ready position.",
        "Move the UR5e to the left.",
        "Move to the right.",
        "Return the robot home."
    ]
    
    t_list = []
    x_list = []; y_list = []; z_list = []
    action_list = []
    
    for prompt in user_prompts:
        if not viewer.is_running():
            break
            
        print(f"\nUser: '{prompt}'")
        
        # Give the prompt to Laya as the "state" context
        state_json = json.dumps({"user_request": prompt})
        
        laya_response = router.predict(state_json, QUESTIONS, model="multilingual")
        chosen_pose = laya_response['answers']['action']['choice']
        
        print(f"LLM produced: {{'action': '{chosen_pose}'}}")
        
        # 4. Robot Controller executes the chosen pose
        if chosen_pose not in NAMED_POSES:
            print("Invalid pose chosen by LLM!")
            continue
            
        target_q = NAMED_POSES[chosen_pose]
        print(f"MuJoCo: UR5e moving to {chosen_pose}...")
        
        # Smooth interpolation to the target over 100 steps
        start_q = data.qpos[:6].copy()
        
        for i in range(100):
            alpha = i / 100.0
            interpolated_q = start_q * (1 - alpha) + target_q * alpha
            
            data.ctrl[:6] = interpolated_q
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Log telemetry
            t_list.append(data.time)
            pinch_pos = data.site_xpos[model.site('pinch').id]
            x_list.append(pinch_pos[0])
            y_list.append(pinch_pos[1])
            z_list.append(pinch_pos[2])
            action_list.append(chosen_pose)
            
            time.sleep(0.01)
            
        time.sleep(1) # Pause before next command

print("Simulation finished.")

# --- Plotting and Metrics ---
if len(t_list) > 0:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Level 1: LLM Predefined Poses - End Effector Trajectory", fontsize=14, fontweight="bold")
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot(x_list, y_list, z_list, "b-", linewidth=2)
    ax1.set_xlabel("X [m]"); ax1.set_ylabel("Y [m]"); ax1.set_zlabel("Z [m]")
    ax1.set_title("3D Path")
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t_list, x_list, "r-", label="X")
    ax2.plot(t_list, y_list, "g-", label="Y")
    ax2.plot(t_list, z_list, "b-", label="Z")
    
    # Mark where actions change
    last_act = None
    for t, act in zip(t_list, action_list):
        if act != last_act:
            ax2.axvline(x=t, color='k', linestyle='--', alpha=0.3)
            ax2.text(t, max(z_list), f" {act}", rotation=90, verticalalignment='top', alpha=0.7)
            last_act = act
            
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Position [m]")
    ax2.set_title("Cartesian Coordinates over Time")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("level1_analysis.png")
    print("Saved telemetry plot to level1_analysis.png")
