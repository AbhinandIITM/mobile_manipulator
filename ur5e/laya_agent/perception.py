import numpy as np
import json
import cv2

def detect_red_blob(image_rgb):
    """
    Detects the red block in the RGB image.
    Returns (cx, cy) of the centroid, and the area of the contour.
    If no block is found, returns (None, None), 0, bgr, None.
    """
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    lower_red_1 = np.array([0, 100, 50])
    upper_red_1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    
    lower_red_2 = np.array([170, 100, 50])
    upper_red_2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    
    mask = mask1 + mask2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
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

def get_gripper_actuator_idx(model):
    """Returns the actuator index for the gripper (10 in mobile manipulator, 6 in standalone UR5e)."""
    try:
        grip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
        if grip_id >= 0:
            return grip_id
    except Exception:
        pass
    return 10 if model.nu >= 11 else 6


def get_scene_state_json(model, data):
    box_pos = data.geom_xpos[model.geom('block_geom').id]
    pinch_pos = data.site_xpos[model.site('pinch').id]
    
    dist_xy = np.linalg.norm(box_pos[:2] - pinch_pos[:2])
    dist_z = pinch_pos[2] - box_pos[2]
    
    grip_idx = get_gripper_actuator_idx(model)
    gripper_open = bool(data.ctrl[grip_idx] < 120)
    
    dx_cm = int(np.clip(round((box_pos[0] - pinch_pos[0]) * 100), -4, 5))
    dy_cm = int(np.clip(round((box_pos[1] - pinch_pos[1]) * 100), -4, 5))
    dz_cm = int(np.clip(round((box_pos[2] - pinch_pos[2]) * 100), -4, 5))
    
    state = {
        "dist_xy_meters": round(float(dist_xy), 3),
        "height_above_block_meters": round(float(dist_z), 3),
        "gripper_is_open": gripper_open,
        "target_dx_cm": dx_cm,
        "target_dy_cm": dy_cm,
        "target_dz_cm": dz_cm,
        "block_x": round(float(box_pos[0]), 3),
        "block_y": round(float(box_pos[1]), 3),
        "block_z": round(float(box_pos[2]), 3),
    }
    return json.dumps(state)


def get_state_feature_vector(model, data):
    """
    Returns the exact 6D feature vector required by the trained GraspPolicyMLP:
    [dist_xy, height_above_block, gripper_open, target_dx, target_dy, target_dz]
    """
    box_pos = data.geom_xpos[model.geom('block_geom').id]
    pinch_pos = data.site_xpos[model.site('pinch').id]

    dist_xy = float(np.linalg.norm(box_pos[:2] - pinch_pos[:2]))
    dist_z = float(pinch_pos[2] - box_pos[2])
    grip_idx = get_gripper_actuator_idx(model)
    gripper_open = float(data.ctrl[grip_idx] < 120)

    dx_cm = float(np.clip(round((box_pos[0] - pinch_pos[0]) * 100), -4, 5))
    dy_cm = float(np.clip(round((box_pos[1] - pinch_pos[1]) * 100), -4, 5))
    dz_cm = float(np.clip(round((box_pos[2] - pinch_pos[2]) * 100), -4, 5))

    return np.array([dist_xy, dist_z, gripper_open, dx_cm, dy_cm, dz_cm], dtype=np.float32)

