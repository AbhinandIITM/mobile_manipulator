# Mobile Manipulator in MuJoCo

This project is a MuJoCo-based equivalent of the Gazebo Fortress Mobile Manipulator project. It demonstrates how to assemble a mobile base (Jackal-like) and a robotic arm (UR5e) directly in MuJoCo and control them efficiently using Python.

## Structure
- `build_xml.py`: Python script that dynamically generates the combined `mobile_manipulator.xml` model using the Universal Robots UR5e models from the local `ur5e` directory and the true Clearpath Jackal base from `j100_asm`.
- `mobile_manipulator.xml`: The generated MJCF (MuJoCo XML) file for the mobile manipulator.
- `simulate.py`: A basic simulation script that demonstrates simple joint-level control and diff-drive control over the model, popping up the interactive `mujoco.viewer`.
- `ik_node_mujoco.py`: An Inverse Kinematics (IK) implementation mimicking the `ik_node` from the ROS 2 project. It uses MuJoCo's native Jacobians (`mujoco.mj_jacBody`) to compute joint velocities for the UR5e to track a target Cartesian position while moving the mobile base.

## Requirements
- `mujoco`
- `mujoco-viewer` (built-in passive viewer in newer mujoco)
- `numpy`

## Getting Started
1. **Explore the model**:
   You can drag and drop `mobile_manipulator.xml` into the MuJoCo viewer or run:
   ```bash
   python simulate.py
   ```
2. **Run Inverse Kinematics**:
   To see the IK solver driving the arm to a target Cartesian position (represented by a red sphere) while the base moves, run:
   ```bash
   python ik_node_mujoco.py
   ```
