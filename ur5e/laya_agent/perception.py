import numpy as np
import json

def get_scene_state_json(model, data):
    # Extract true physical positions from MuJoCo to act as our perception pipeline.
    # (In real life, this would come from OpenCV/DINO centroid logic).
    box_pos = data.geom_xpos[model.geom('block_geom').id]
    pinch_pos = data.site_xpos[model.site('pinch').id]
    
    # Calculate 2D distance between gripper and block
    dist_xy = np.linalg.norm(box_pos[:2] - pinch_pos[:2])
    # Calculate height of gripper above block
    dist_z = pinch_pos[2] - box_pos[2]
    
    gripper_open = bool(data.ctrl[6] < 120)
    
    state = {
        "dist_xy_meters": round(float(dist_xy), 3),
        "height_above_block_meters": round(float(dist_z), 3),
        "gripper_is_open": gripper_open,
        "block_x": round(float(box_pos[0]), 3),
        "block_y": round(float(box_pos[1]), 3)
    }
    
    return json.dumps(state)
