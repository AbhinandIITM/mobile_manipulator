import os
import time
import numpy as np
import mujoco
import mujoco.viewer

from controllers import CartesianIKController

def get_block_pos(data, model):
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    return data.xpos[block_id].copy()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "ur5e_grasp_scene.xml")

    print(f"Loading grasp scene from: {model_path} ...")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Initial home configuration
    home_qpos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    data.qpos[:6] = home_qpos
    data.ctrl[:6] = home_qpos
    
    # Gripper open
    data.ctrl[6] = 0

    mujoco.mj_forward(model, data)

    # Controller
    controller = CartesianIKController(
        model=model,
        site_name="pinch",
        home_qpos=data.qpos[:6].copy(),
        kp_pos=10.0,
        kp_ori=10.0,
        k_null=1.0,
        damping=0.05,
        max_qdot=1.5,
    )
    
    mocap_id = -1

    dt = model.opt.timestep

    # State Machine
    phase = 0
    phase_start_time = 0.0

    target_rot = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ])

    # Telemetry logging lists
    t_list = []
    target_pos_list = []
    eef_pos_list = []
    pos_err_list = []
    ori_err_list = []
    gripper_cmd_list = []
    block_z_list = []

    # Initialize off-screen renderer for wrist camera
    renderer = mujoco.Renderer(model, 480, 640)

    print("Starting grasp sequence...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 110.0
        viewer.cam.elevation = -20.0
        viewer.cam.distance = 1.5
        viewer.cam.lookat[:] = [-0.1, 0.5, 0.1]

        while viewer.is_running():
            step_start = time.time()
            t = data.time

            block_pos = get_block_pos(data, model)
            
            # Sequence
            target_pos = block_pos.copy()
            gripper_cmd = 0.0
            
            # 1. Approach from above
            if phase == 0:
                target_pos[2] = 0.25
                if t > 3.0:
                    phase = 1
                    phase_start_time = t
                    print("Moving down to grasp...")
                    
                    # Capture wrist camera image before descending
                    renderer.update_scene(data, camera="d435i_color")
                    img = renderer.render()
                    try:
                        from PIL import Image
                        Image.fromarray(img).save("wrist_cam_view.png")
                        print("Saved wrist camera view to 'wrist_cam_view.png'")
                    except ImportError:
                        print("Pillow (PIL) not installed, could not save camera image.")
            
            # 2. Go down
            elif phase == 1:
                target_pos[2] = 0.04  # Just above the floor, fingers around the box
                if t - phase_start_time > 3.0:
                    phase = 2
                    phase_start_time = t
                    print("Closing gripper...")
            
            # 3. Close gripper
            elif phase == 2:
                target_pos[2] = 0.04
                gripper_cmd = 255.0
                if t - phase_start_time > 1.5:
                    phase = 3
                    phase_start_time = t
                    print("Lifting object...")
                    
            # 4. Lift up
            elif phase == 3:
                target_pos[2] = 0.3
                gripper_cmd = 255.0

            # Update mocap visual
            if mocap_id >= 0:
                data.mocap_pos[mocap_id] = target_pos

            # IK control
            q_cmd, diag_info = controller.compute_control(
                data=data,
                target_pos=target_pos,
                target_vel=np.zeros(3),
                target_rot=target_rot,
                target_angvel=np.zeros(3),
                dt=dt,
            )

            # Commands
            data.ctrl[:6] = q_cmd
            data.ctrl[6] = gripper_cmd

            # G. Log Telemetry
            if int(t / dt) % 5 == 0:
                curr_pos = data.site_xpos[controller.site_id].copy()
                t_list.append(t)
                target_pos_list.append(target_pos.copy())
                eef_pos_list.append(curr_pos.copy())
                pos_err_list.append(diag_info["pos_err_norm"])
                ori_err_list.append(diag_info["ori_err_norm"])
                gripper_cmd_list.append(gripper_cmd)
                block_z_list.append(block_pos[2])

            mujoco.mj_step(model, data)
            viewer.sync()

            # Keep real-time pace
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
            # Stop after sequence is done (e.g. at 12 seconds)
            if t > 12.0:
                break

    # --- Plotting and Metrics ---
    if len(t_list) > 10:
        import matplotlib.pyplot as plt

        t_arr = np.array(t_list)
        target_pos_arr = np.array(target_pos_list)
        eef_pos_arr = np.array(eef_pos_list)
        pos_err_cm = np.array(pos_err_list) * 100.0
        ori_err_deg = np.degrees(np.array(ori_err_list))
        block_z_arr = np.array(block_z_list)
        gripper_cmd_arr = np.array(gripper_cmd_list)

        print("\n" + "=" * 70)
        print(" UR5e Grasping Performance Summary")
        print("=" * 70)
        print(f"Total Simulation Time:        {t_arr[-1]:.2f} s")
        print(f"Mean Cartesian Pos Error:     {np.mean(pos_err_cm):.3f} cm")
        print(f"Max Cartesian Pos Error:      {np.max(pos_err_cm):.3f} cm")
        print(f"Final Object Z Elevation:     {block_z_arr[-1]:.3f} m")
        print("=" * 70)

        fig = plt.figure(figsize=(15, 10))
        plt.suptitle("UR5e Pick-and-Place Grasping Analysis", fontsize=16, fontweight="bold")

        # 1. 3D Workspace Path
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        ax1.plot(target_pos_arr[:, 0], target_pos_arr[:, 1], target_pos_arr[:, 2], "r--", label="Target", linewidth=1.5)
        ax1.plot(eef_pos_arr[:, 0], eef_pos_arr[:, 1], eef_pos_arr[:, 2], "b-", label="Actual EEF", linewidth=1.5)
        ax1.set_title("3D End-Effector Path")
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")
        ax1.set_zlabel("Z [m]")
        ax1.legend()
        ax1.grid(True)

        # 2. XYZ Coordinates vs Time
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(t_arr, target_pos_arr[:, 2], "b--", label="Target Z")
        ax2.plot(t_arr, eef_pos_arr[:, 2], "b-", label="Actual Z")
        ax2.plot(t_arr, block_z_arr, "g-", linewidth=2.0, label="Object Z")
        ax2.set_title("Z Elevation (EEF vs Object)")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Z [m]")
        ax2.legend()
        ax2.grid(True)

        # 3. Tracking Errors
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(t_arr, pos_err_cm, "m-", label="Position Error [cm]")
        ax3.set_title("Cartesian Tracking Error")
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Error [cm]")
        ax3.legend()
        ax3.grid(True)

        # 4. Gripper state
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(t_arr, gripper_cmd_arr, "k-", label="Gripper CMD (0-255)")
        ax4.set_title("Gripper Actuation")
        ax4.set_xlabel("Time [s]")
        ax4.set_ylabel("Command Signal")
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        save_path = os.path.join(script_dir, "grasp_analysis.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[Plot Saved] Analysis plot successfully saved to: {save_path}")

if __name__ == "__main__":
    main()
