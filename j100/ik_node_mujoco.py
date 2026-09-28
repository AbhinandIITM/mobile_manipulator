import mujoco
import mujoco.viewer
import numpy as np
import time

def main():
    print("Loading the model...")
    model = mujoco.MjModel.from_xml_path('mobile_manipulator.xml')
    data = mujoco.MjData(model)

    # Find the end effector body/site. In UR5e it's usually "wrist_3_link" or a specific site.
    # We will use "wrist_3_link" as the end-effector.
    ee_id = model.body('wrist_3_link').id
    
    # Set a target position
    target_pos = np.array([0.5, 0.0, 0.5])
    
    print("Starting simulation with IK...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        # Add a visual marker for the target
        viewer.user_scn.ngeom = 1
        mujoco.mjv_initGeom(viewer.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.05, 0.05, 0.05]), target_pos, np.eye(3).flatten(), np.array([1, 0, 0, 1]))
        
        while viewer.is_running():
            step_start = time.time()
            
            # 1. Forward kinematics to get current EE pos
            mujoco.mj_kinematics(model, data)
            current_pos = data.xpos[ee_id]
            
            # 2. Compute error
            err = target_pos - current_pos
            
            # 3. Get Jacobian for the end effector
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, None, ee_id)
            
            # We only want to move the arm, so we ignore the base DOF
            # Free joint (6) + 4 wheels = 10 dofs. Arm starts at index 10 in nv.
            arm_jacp = jacp[:, 10:]
            
            # 4. Compute joint velocities using pseudo-inverse
            # Damped least squares
            lambda_sq = 1e-4
            J_pinv = arm_jacp.T @ np.linalg.inv(arm_jacp @ arm_jacp.T + lambda_sq * np.eye(3))
            
            dq = J_pinv @ (err * 2.0) # Proportional gain
            
            # 5. Apply velocities (as simple P-control for positions)
            # Free joint (7) + 4 wheels = 11. Arm starts at index 11 in qpos.
            # Arm starts at index 4 in ctrl.
            data.ctrl[4:] = data.qpos[11:] + dq * model.opt.timestep
            
            # Slowly move the base forward
            data.ctrl[0] = 0.5
            data.ctrl[1] = 0.5
            data.ctrl[2] = 0.5
            data.ctrl[3] = 0.5

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == '__main__':
    main()
