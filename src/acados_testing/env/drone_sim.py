import mujoco
import numpy as np
import time

# Simulation class for a drone in MuJoCo, used for testing MPC controllers.

class Sim:
    def __init__(self, use_gui=False):
        self.use_gui = use_gui

        path = '/home/ws/src/drone_mujoco/model/scene.xml'

        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        
        if self.use_gui:
            import mujoco_viewer
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        else:
            self.viewer = None
        
        self.model.opt.timestep = 0.005
        self.running = True
        self.prev_time = time.time()
        
        print("Model loaded successfully.")
        print(self.data)

    def set_control(self, motor_commands):
        self.data.ctrl = np.clip(motor_commands, 0, 1)

    def get_state(self):
        # Get position and orientation
        position = self.data.qpos[0:3]
        orientation = self.data.qpos[3:7]
        return position, orientation

    def step(self):
        mujoco.mj_step(self.model, self.data)
        current_time = time.time()
        print(f"Iterations per second: {int(1.0 / (current_time - self.prev_time))}")
        self.prev_time = current_time

        if self.viewer and self.viewer.is_alive:
            self.viewer.render()
        elif self.viewer and not self.viewer.is_alive:
            self.running = False

    def reset(self):
        self.data.qpos[:] = 0.0  # Reset positions
        self.data.qvel[:] = 0.0  # Reset velocities
        self.data.qpos[2] = 1.0  # Set initial height

    def close(self):
        if self.viewer:
            self.viewer.close()
        self.running = False


def main():
    use_gui = True  # Set to False if GUI is not needed
    sim = Sim(use_gui=use_gui)
    sim.reset()

    try:
        while sim.running:
            sim.step()
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()


if __name__ == '__main__':
    main()