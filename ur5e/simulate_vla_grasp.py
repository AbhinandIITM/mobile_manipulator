import mujoco
import mujoco.viewer
import numpy as np
import cv2
import torch
import time
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from PIL import Image
from controllers import solve_ik

# --- 1. Load VLA Model ---
print("Loading OpenVLA in 4-bit (This might take a minute)...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

prompt = "In: What action should the robot take to grasp the red block?\nOut:"

# --- 2. Setup MuJoCo ---
print("Loading MuJoCo scene...")
model = mujoco.MjModel.from_xml_path('ur5e_grasp_scene.xml')
data = mujoco.MjData(model)

# Set initial joint positions (Home)
home_qpos = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]
mujoco.mj_resetData(model, data)
data.qpos[:6] = home_qpos
mujoco.mj_forward(model, data)

# Camera setup
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
cam.fixedcamid = model.camera('d435i_color').id
width, height = 640, 480
scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)

# VLA Action scale (VLA outputs [-1, 1], we map to max 0.05m delta per step)
MAX_XYZ_DELTA = 0.05

def get_camera_image():
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), scene, context)
    mujoco.mjr_readPixels(rgb_buffer, None, mujoco.MjrRect(0, 0, width, height), context)
    return np.flipud(rgb_buffer)

# --- 3. Simulation Loop ---
print("Starting VLA Control Loop...")
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    # Wait for viewer to settle
    time.sleep(2)
    
    while viewer.is_running():
        # Step 1: Capture RGB from wrist camera
        img_np = get_camera_image()
        img_pil = Image.fromarray(img_np)
        
        # Step 2: VLA Inference
        inputs = processor(prompt, img_pil).to("cuda", dtype=torch.bfloat16)
        
        # We sample the next action using bridge dataset unnormalization
        with torch.no_grad():
            action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            
        print(f"VLA Output Action: {action}")
        
        # OpenVLA action format: [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper]
        delta_x, delta_y, delta_z = action[0], action[1], action[2]
        
        # The actions from 'bridge_orig' are unnormalized specifically for the Bridge dataset.
        # But we will clamp and scale them slightly for safety in our sim.
        target_pos = data.site_xpos[model.site('pinch').id].copy()
        target_pos[0] += np.clip(delta_x, -MAX_XYZ_DELTA, MAX_XYZ_DELTA)
        target_pos[1] += np.clip(delta_y, -MAX_XYZ_DELTA, MAX_XYZ_DELTA)
        target_pos[2] += np.clip(delta_z, -MAX_XYZ_DELTA, MAX_XYZ_DELTA)
        
        # Simple downward orientation for top-down grasp
        target_quat = [0, 1, 0, 0] 
        
        # Step 3: Execute IK
        qpos_target = solve_ik(model, data, target_pos, target_quat, 'pinch')
        
        # Smoothly step towards target
        for _ in range(50):
            data.ctrl[:6] = data.qpos[:6] + 0.1 * (qpos_target - data.qpos[:6])
            
            # Gripper control (VLA output: < 0 is close, > 0 is open)
            if action[6] < 0:
                data.ctrl[6] = 255
            else:
                data.ctrl[6] = 0
                
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)
