import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import cv2

from controllers import CartesianIKController

def get_block_pos(data, model):
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    return data.xpos[block_id].copy()

def detect_red_blob(image_rgb):
    """
    Detects the red block in the RGB image.
    Returns (cx, cy) of the centroid, and the area of the contour.
    If no block is found, returns (None, None), 0.
    """
    # Convert RGB to BGR for OpenCV, then to HSV
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    # Red has two ranges in HSV
    lower_red_1 = np.array([0, 100, 50])
    upper_red_1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    
    lower_red_2 = np.array([170, 100, 50])
    upper_red_2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    
    mask = mask1 + mask2
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get largest contour
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > 50:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw for visualization
                cv2.drawContours(bgr, [c], -1, (0, 255, 0), 2)
                cv2.circle(bgr, (cx, cy), 5, (255, 255, 255), -1)
                
                return (cx, cy), area, bgr, mask
    
    return (None, None), 0, bgr, None

def get_object_world_pose_from_pointcloud(model, data, mask, depth_map, width, height):
    """Calculates true 3D world center by projecting the entire segmented mask into a point cloud."""
    fovy = 60.0 * np.pi / 180.0
    f = (height / 2.0) / np.tan(fovy / 2.0)
    
    # Get all pixel coordinates that are part of the red blob
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
        
    z_cams = depth_map[ys, xs]
    
    # 1. Pinhole Model: Get local 3D vectors in camera frame for ALL pixels
    local_xs = (xs - width / 2.0) * z_cams / f
    local_ys = -(ys - height / 2.0) * z_cams / f
    local_zs = -z_cams
    
    # Shape: (3, N)
    local_pts = np.vstack((local_xs, local_ys, local_zs))
    
    # 2. Transform to world coordinates
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "d435i_color")
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    
    world_pts = cam_pos[:, None] + cam_mat @ local_pts
    
    # 3. Find the true geometric center of the 3D point cloud
    min_x = np.min(world_pts[0, :])
    max_x = np.max(world_pts[0, :])
    min_y = np.min(world_pts[1, :])
    max_y = np.max(world_pts[1, :])
    
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    center_z = np.max(world_pts[2, :])  # Top surface Z
    
    return np.array([center_x, center_y, center_z])

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "ur5e_grasp_scene.xml")

    print(f"Loading vision grasp scene from: {model_path} ...")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Initial home configuration
    home_qpos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    data.qpos[:6] = home_qpos
    data.ctrl[:6] = home_qpos
    data.ctrl[6] = 0  # Gripper open

    # Randomize block position over a wide area so the robot has to search
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    data.qpos[6] = -0.3 + np.random.uniform(0, 0.4)  # X between -0.3 and 0.1
    data.qpos[7] = 0.4 + np.random.uniform(0, 0.2)   # Y between 0.4 and 0.6
    
    mujoco.mj_forward(model, data)

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
    
    width, height = 320, 240
    renderer = mujoco.Renderer(model, height, width)

    # State Machine
    phase = 0
    
    # Target always points down
    target_rot = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ])
    
    # Telemetry logging lists
    t_list = []
    target_pos_list = []
    eef_pos_list = []
    gripper_cmd_list = []
    block_z_list = []
    vision_err_x_list = []
    vision_err_y_list = []
    
    # Initial hover target
    target_pos = np.array([-0.1, 0.5, 0.3])
    gripper_cmd = 0.0

    print("Starting vision-guided grasp sequence...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 110.0
        viewer.cam.elevation = -20.0
        viewer.cam.distance = 1.5
        viewer.cam.lookat[:] = [-0.1, 0.5, 0.1]

        last_render_time = -1
        phase_start_time = -1

        while viewer.is_running():
            step_start = time.time()
            t = data.time
            
            # --- Vision Processing (run at ~15 Hz) ---
            if t - last_render_time > 0.06:
                renderer.disable_depth_rendering()
                renderer.update_scene(data, camera="d435i_color")
                img_rgb = renderer.render()
                
                (cx, cy), area, img_bgr, mask = detect_red_blob(img_rgb)
                
                # If object is detected, capture depth!
                depth_map = None
                if cx is not None and cy is not None:
                    renderer.enable_depth_rendering()
                    renderer.update_scene(data, camera="d435i_color")
                    depth_map = renderer.render()
                
                # Show live feed
                cv2.imshow("Wrist Camera Live Feed (Vision Guided)", img_bgr)
                cv2.waitKey(1)
                
                last_render_time = t
                
                # --- Visual Servoing Logic ---
                if phase == 0:
                    # Exploration / Scanning (Sweep the arm across the table)
                    target_pos[0] = -0.15 + 0.25 * np.sin(t * 1.5)
                    target_pos[1] = 0.50 + 0.15 * np.cos(t * 1.0)
                    target_pos[2] = 0.5
                    
                    if cx is not None and cy is not None and area > 10 and depth_map is not None:
                        print(f"[{t:.2f}s] Object spotted in camera feed! Using DEPTH map for exact 3D pose...")
                        
                        world_p = get_object_world_pose_from_pointcloud(model, data, mask, depth_map, width, height)
                        if world_p is not None:
                            est_x, est_y, est_z = world_p
                            
                            print(f"--> Exact Object World Pose (Depth): X={est_x:.3f}, Y={est_y:.3f}, Z={est_z:.3f}")
                            
                            # Move directly to the exact pose (above it)
                            target_pos[0] = est_x
                            target_pos[1] = est_y
                            
                            print("Transitioning to Position-Based Visual Servoing (PBVS)...")
                            phase = 1
                        
                elif phase == 1:
                    # Hover and align using continuous depth feedback
                    target_pos[2] = 0.3
                    
                    eef_x = data.site_xpos[controller.site_id][0]
                    eef_y = data.site_xpos[controller.site_id][1]
                    dist = np.hypot(eef_x - target_pos[0], eef_y - target_pos[1])
                    
                    if cx is not None and cy is not None and depth_map is not None:
                        world_p = get_object_world_pose_from_pointcloud(model, data, mask, depth_map, width, height)
                        if world_p is not None:
                            # Smoothly update target with new continuous measurements (PBVS)
                            target_pos[0] = 0.7 * target_pos[0] + 0.3 * world_p[0]
                            target_pos[1] = 0.7 * target_pos[1] + 0.3 * world_p[1]
                        
                        if dist < 0.005:
                            print("Centered accurately on object! Descending...")
                            phase = 2
                    elif dist < 0.005:
                        print("Lost tracking! Arrived at estimated target X/Y. Descending blindly...")
                        phase = 2
                
                elif phase == 2:
                    # Descend blindly to the exact locked XY coordinates (camera gets too close and FOV cuts off block)
                    target_pos[2] -= 0.003
                    
                    # Stop descending when Z is at the block's true center height (0.025m)
                    if target_pos[2] <= 0.025:
                        target_pos[2] = 0.025
                        print("Reached exact block center height! Closing gripper...")
                        phase = 3
                        phase_start_time = t
                        
                elif phase == 3:
                    # Close gripper
                    gripper_cmd = 255.0
                    if t - phase_start_time > 1.5:
                        print("Lifting object...")
                        phase = 4
                        
                elif phase == 4:
                    # Lift
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

            # Log telemetry
            if int(t / dt) % 10 == 0:
                t_list.append(t)
                target_pos_list.append(target_pos.copy())
                eef_pos_list.append(data.site_xpos[controller.site_id].copy())
                gripper_cmd_list.append(gripper_cmd)
                block_pos = get_block_pos(data, model)
                block_z_list.append(block_pos[2])
                
                # We just log 0 if it's not currently tracked
                err_x = cx - (width / 2) if cx is not None else 0
                err_y = cy - (height / 2) if cy is not None else 0
                vision_err_x_list.append(err_x)
                vision_err_y_list.append(err_y)

            mujoco.mj_step(model, data)
            
            # Sync viewer every 30 steps (~16ms)
            if int(t / dt) % 30 == 0:
                viewer.sync()

            # Keep real-time pace
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
            if t > 25.0:
                break
                
    cv2.destroyAllWindows()
    
    # --- Plotting and Metrics ---
    if len(t_list) > 10:
        import matplotlib.pyplot as plt

        t_arr = np.array(t_list)
        target_pos_arr = np.array(target_pos_list)
        eef_pos_arr = np.array(eef_pos_list)
        block_z_arr = np.array(block_z_list)
        gripper_cmd_arr = np.array(gripper_cmd_list)
        err_x_arr = np.array(vision_err_x_list)
        err_y_arr = np.array(vision_err_y_list)

        fig = plt.figure(figsize=(15, 10))
        plt.suptitle("UR5e + RealSense Vision-Guided Grasping", fontsize=16, fontweight="bold")

        # 1. 3D Workspace Path
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        ax1.plot(target_pos_arr[:, 0], target_pos_arr[:, 1], target_pos_arr[:, 2], "r--", label="Target", linewidth=1.5)
        ax1.plot(eef_pos_arr[:, 0], eef_pos_arr[:, 1], eef_pos_arr[:, 2], "b-", label="Actual EEF", linewidth=1.5)
        ax1.set_title("3D Search & Grasp Path")
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

        # 3. Vision Centering Error
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(t_arr, err_x_arr, "m-", label="Pixel Error X")
        ax3.plot(t_arr, err_y_arr, "c-", label="Pixel Error Y")
        ax3.set_title("Visual Servoing Error")
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Error [pixels]")
        ax3.axhline(0, color='k', linestyle='--', linewidth=1)
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
        save_path = os.path.join(script_dir, "vision_grasp_analysis.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[Plot Saved] Analysis plot successfully saved to: {save_path}")

if __name__ == "__main__":
    main()
