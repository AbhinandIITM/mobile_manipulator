import mujoco
import mujoco.viewer
import time
import numpy as np

def main():
    # Load the combined XML model
    print("Loading the model...")
    model = mujoco.MjModel.from_xml_path('mobile_manipulator.xml')
    data = mujoco.MjData(model)

    # Simulation loop
    print("Starting the simulation...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        start = time.time()
        while viewer.is_running():
            step_start = time.time()

            # Control the wheels to drive forward
            data.ctrl[0] = 2.0  # FL
            data.ctrl[1] = 2.0  # FR
            data.ctrl[2] = 2.0  # RR
            data.ctrl[3] = 2.0  # RL
            
            # Control the UR5e arm joints sinusoidally
            t = data.time
            # The arm joints start from index 4 in the actuators
            for i in range(4, model.nu):
                data.ctrl[i] = np.sin(t + i) * 0.5

            mujoco.mj_step(model, data)
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == '__main__':
    main()
