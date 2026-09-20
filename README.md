# Autonomous Mobile Manipulator in MuJoCo

A high-performance physics-based perception, state estimation, guidance, and manipulation framework for the **Clearpath Jackal (J100)** 4-wheel skid-steer mobile robot and the **Universal Robots UR5e** 6-DOF robotic manipulator in **MuJoCo**.

Detailed mathematical derivations and control theory proofs are documented in [**`METHODOLOGY.md`**](file:///D:/Projects/Mobile_manipulator/METHODOLOGY.md).

---

## 🚀 Key Features

### 1. Control Lyapunov Function (CLF) + Control Barrier Function (CBF) QP Guidance
* **Zero Distance Overshoot**: Enforces a strict safety barrier $h(R) = R - d_{\text{target}} \ge 0$ via forward invariance.
* **Anti-Pitch / Wheelie Suppression**: Dynamic acceleration slew envelope ($\dot{v} \le 1.0\text{ m/s}^2$) prevents motor torque reaction spikes, keeping chassis pitch oscillation $< 0.23^\circ$.
* **Dynamic Moving Target Pursuit**: Kinematic velocity feedforward dynamically matches target orbital speed ($0.80\text{ m/s}$) at exact standoff distance.
* **Ultra-Fast Analytical Active-Set QP Solver**: Evaluates and solves the constrained CLF-CBF QP in $< 1\,\mu\text{s}$ at 500 Hz.

### 2. Multi-Rate RGB-D Perception & IMU State Estimation
* **30 Hz RGB-D Perception**: Pinhole camera geometry extracts Line-of-Sight (LOS) azimuth bearing angle $\lambda$ and median depth-based range $R$.
* **500 Hz IMU Gyro Propagation**: High-frequency kinematic integration ($\dot{\lambda} = -\omega_z$) maintains sub-milliradian tracking between vision frames.
* **Real-time OpenCV HUD Composite**: Dual-panel live visualization (Left: RGB reticle + bounding box + telemetry; Right: JET colorized depth map).

### 3. UR5e 6-DOF Manipulator Cartesian Trajectory Tracking
* **Damped Least Squares (DLS) Inverse Kinematics**: Singularity-robust Cartesian motion control with Levenberg-Marquardt damping.
* **Nullspace Posture Optimization**: Utilizes redundant degrees of freedom to bias joint configuration toward optimal resting posture.
* **5th-Order Minimum-Jerk Trajectory Generators**: Smooth spatial tracking across `figure8`, `circle`, `spiral`, and `waypoints` with steady-state tracking error $< 2.5\text{ mm}$.

---

## 📁 Repository Structure

```
Mobile_manipulator/
│
├── clf_cbf_guidance.py          # CLF-CBF Active-Set QP Guidance Controller
├── simulate_j100_clf_cbf.py     # CLF-CBF Simulation runner & 6-panel analysis generator
│
├── los_guidance.py              # Baseline Line-of-Sight (LOS) Guidance & HUD Stack
├── simulate_j100_los.py         # Multi-rate LOS guidance simulation script
│
├── controller.py                # Skid-steer Pure Pursuit path tracking controller
├── ekf.py                       # Extended Kalman Filter (EKF) sensor fusion
├── sensors.py                   # IMU, wheel encoder, and camera noise models
├── simulate_j100.py             # J100 navigation & odometry benchmark
│
├── METHODOLOGY.md               # Complete mathematical and algorithmic documentation
│
├── ur5e/                        # UR5e Manipulator Subsystem
│   ├── controllers.py           # 6-DOF DLS Inverse Kinematics & Nullspace Control
│   ├── trajectories.py          # 3D Minimum-Jerk spatial trajectories (Figure-8, Circle, Spiral)
│   ├── simulate_ur5e.py         # Interactive UR5e 3D simulation with trajectory ribbons
│   └── ur5e_scene.xml           # Manipulator MJCF scene definition
│
├── j100_asm/                    # Clearpath Jackal (J100) Base Model
│   ├── j100_asm.xml             # Jackal MJCF model with calibrated front RGB-D camera
│   └── meshes/                  # High-fidelity CAD STL meshes
│
└── j100_los_scene.xml           # Multi-rate perception & tracking scene
```

---

## 🛠️ Installation & Setup

Ensure you have Python 3.10+ installed with the following packages:

```bash
pip install mujoco opencv-python numpy matplotlib scipy
```

---

## 🎮 Quickstart & Execution

### 1. Run CLF-CBF Precision Docking & Guidance

```powershell
# A. Static Target Precision Docking (Zero Overshoot & Flat Chassis)
python simulate_j100_clf_cbf.py --scenario static --d_stop 0.80

# B. Moving Target Dynamic Standoff Tracking (0.80 m Distance Hold)
python simulate_j100_clf_cbf.py --scenario moving --d_stop 0.80

# C. Multi-Waypoint Circuit Navigation
python simulate_j100_clf_cbf.py --scenario waypoints
```

### 2. Run UR5e Manipulator Cartesian Spatial Tracking

```powershell
# A. 3D Figure-8 Trajectory (Spatial Cartesian Ribbon)
python ur5e/simulate_ur5e.py --trajectory figure8

# B. 3D Helix / Spiral Trajectory
python ur5e/simulate_ur5e.py --trajectory spiral

# C. Spatial Circular Orbit
python ur5e/simulate_ur5e.py --trajectory circle
```

### 3. Run Pure Pursuit Path Tracking & EKF Odometry

```powershell
python simulate_j100.py --trajectory figure8 --speed 1.2
```

---

## 📊 Performance Benchmarks

| Capability | Metric | Result |
| :--- | :--- | :--- |
| **Static Standoff Convergence** | Final Distance Error | **$0.00\text{ m}$** (Monotonic descent to $0.80\text{ m}$, zero overshoot) |
| **Chassis Pitch Oscillation** | Max Pitch Angle ($\theta_{\text{pitch}}$) | **$0.23^\circ$** (Eliminated front lift-up / wheelie) |
| **Moving Target Range Error** | Steady-State Range ($R$) | **$0.80\text{ m}$** ($0.0\text{ mm}$ error at $0.80\text{ m/s}$ target speed) |
| **LOS Bearing Alignment** | Steady-State Error ($\lambda$) | **$0.22^\circ$** |
| **UR5e Cartesian Tracking** | Steady-State 3D Error | **$2.48\text{ mm}$** |
| **QP Solver Latency** | Computation Time per Cycle | **$< 1\,\mu\text{s}$** at 500 Hz |

---

## 📜 Documentation

For complete mathematical formulations, Lie derivatives, Lyapunov proofs, and sensor noise equations, see:
* [**`METHODOLOGY.md`**](file:///D:/Projects/Mobile_manipulator/METHODOLOGY.md)
