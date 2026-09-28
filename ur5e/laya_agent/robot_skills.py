import sys
import os
import numpy as np
import mujoco

# Add parent directory to path to import controllers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from controllers import CartesianIKController

# Global IK controller instance so state (q_target) is maintained
_ik_controller = None

def init_ik(model, qpos_init=None):
    global _ik_controller
    _ik_controller = CartesianIKController(model, site_name='pinch')
    if qpos_init is not None:
        _ik_controller.reset(qpos_init)

def get_target_rot_down():
    # Quat to Rot matrix (pointing down roughly)
    # Target rotation: x axis points forward, z axis points down
    R = np.array([
        [-1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0, -1]
    ])
    return R

def solve_ik_vla(model, data, target_pos, target_rot):
    global _ik_controller
    if _ik_controller is None:
        init_ik(model, data.qpos[:6])
    
    # Run IK
    q_cmd, diag = _ik_controller.compute_control(
        data, 
        target_pos=np.array(target_pos), 
        target_vel=np.zeros(3), 
        target_rot=target_rot, 
        target_angvel=np.zeros(3), 
        dt=0.02
    )
    return q_cmd

def skill_hover_over_block(model, data, block_x, block_y):
    # Hover 30cm above the block
    target_pos = [block_x, block_y, 0.30]
    return solve_ik_vla(model, data, target_pos, get_target_rot_down())

def skill_descend(model, data, block_x, block_y):
    # Drop down to grasp height
    target_pos = [block_x, block_y, 0.025]
    return solve_ik_vla(model, data, target_pos, get_target_rot_down())

def skill_close_gripper(model, data):
    data.ctrl[6] = 255 # max force to close
    return data.qpos[:6].copy()

def skill_open_gripper(model, data):
    data.ctrl[6] = 0
    return data.qpos[:6].copy()

def skill_lift(model, data):
    # Lift back up to 40cm, keeping current X, Y
    pinch_id = model.site('pinch').id
    curr_x, curr_y = data.site_xpos[pinch_id][:2]
    target_pos = [curr_x, curr_y, 0.40]
    return solve_ik_vla(model, data, target_pos, get_target_rot_down())
