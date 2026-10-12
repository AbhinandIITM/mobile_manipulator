"""
Debug: trace what the expert does in one episode
"""
import sys, numpy as np, mujoco
sys.path.insert(0, r'D:\Projects\Mobile_manipulator\ur5e')
from controllers import CartesianIKController

model = mujoco.MjModel.from_xml_path(r'D:\Projects\Mobile_manipulator\ur5e\ur5e_grasp_scene.xml')
data  = mujoco.MjData(model)

home_qpos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
mujoco.mj_resetData(model, data)
data.qpos[:6] = home_qpos; data.ctrl[:6] = home_qpos; data.ctrl[6] = 0.0

jnt_id = model.body_jntadr[model.body("target_block").id]
qadr   = model.jnt_qposadr[jnt_id]
data.qpos[qadr]   = -0.1
data.qpos[qadr+1] =  0.5
data.qpos[qadr+2] =  0.025
mujoco.mj_forward(model, data)

target_rot = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=float)
controller = CartesianIKController(model, "pinch", home_qpos, kp_pos=10, kp_ori=10, k_null=1, damping=0.05, max_qdot=1.5)

dt = model.opt.timestep
target_pos = np.array([-0.1, 0.5, 0.5])
gripper_cmd = 0.0
phase = 1  # skip scan, go straight to hover

block_xy = np.array([-0.1, 0.5])

for step in range(int(20/dt)):
    t = data.time
    pinch = data.site_xpos[model.site("pinch").id]
    block = data.xpos[model.body("target_block").id]

    if step % 500 == 0:
        print(f"t={t:.2f}  phase={phase}  pinch={pinch.round(3)}  block={block.round(3)}  target_z={target_pos[2]:.3f}  gripper={gripper_cmd:.0f}")

    if phase == 1:
        target_pos[0] = float(block_xy[0])
        target_pos[1] = float(block_xy[1])
        target_pos[2] = 0.3
        dist_xy = np.linalg.norm(pinch[:2] - target_pos[:2])
        if dist_xy < 0.01:
            phase = 2
            print(f"  --> Phase 2 at t={t:.2f}")
    elif phase == 2:
        target_pos[2] = max(target_pos[2] - 0.003, 0.025)
        if target_pos[2] <= 0.025:
            phase = 3
            phase_start = t
            print(f"  --> Phase 3 at t={t:.2f}  pinch_z={pinch[2]:.3f}  block_z={block[2]:.3f}")
    elif phase == 3:
        gripper_cmd = 255.0
        if t - phase_start > 1.5:
            phase = 4
            phase_start = t
            print(f"  --> Phase 4 at t={t:.2f}  block_z={block[2]:.3f}")
    elif phase == 4:
        target_pos[2] = 0.35
        gripper_cmd = 255.0
        if block[2] > 0.06:
            print(f"  SUCCESS at t={t:.2f}  block_z={block[2]:.3f}")
            break

    q_cmd, _ = controller.compute_control(data=data, target_pos=target_pos, target_vel=np.zeros(3),
                                           target_rot=target_rot, target_angvel=np.zeros(3), dt=dt)
    data.ctrl[:6] = q_cmd
    data.ctrl[6]  = gripper_cmd
    mujoco.mj_step(model, data)

print(f"\nFinal block z: {data.xpos[model.body('target_block').id][2]:.3f}")
print(f"Final gripper ctrl: {data.ctrl[6]:.0f}")
