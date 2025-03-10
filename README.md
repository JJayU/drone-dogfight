# DRONE DOGFIGHT

## Dependencies

 - [crazyswarm2](https://github.com/IMRCLab/crazyswarm2)
 - [webots_ros2_crazyflie](https://github.com/mwlock/webots_ros2_crazyflie)

## Main goal

The main goal of the project was to develop a control system for the Crazyflie 2.1 drone equipped with a laser, enabling it to precisely track a moving target. The system allowed the drone to ascend to the target's height and rotate toward it, ensuring accurate pointing with the laser. The project aimed to leverage modern programming tools and technologies to create a comprehensive simulation and control environment.


## Setup & Run

###  Install Crazyswarm2 
[Installation — Crazyswarm2 1.0a1 documentation](https://imrclab.github.io/crazyswarm2/installation.html)

### Add to bashrc

```
export PYTHONPATH=/crazyflie-firmware/build:$PYTHONPATH

source /opt/ros/<DISTRO>/setup.bash

source /home/ros2_ws/install/setup.bash
```
### Launch

```
ros2 launch pid_target_tracker main_launch.launch.py
```


## Documentation
### 1. Crazyflie 2.1

The project utilized the Crazyflie 2.1 drone, a lightweight, modular micro-drone equipped with powerful capabilities for autonomous flight. The drone was modified with a forward-facing laser pointer, serving as a precise indicator of the target position.

### 2. Control System

**2.1. Altitude Control** A PID controller was implemented to regulate the drone’s altitude. This controller ensured that the drone ascended or descended to match the height of the moving target accurately. The control loop used real-time feedback from the drone’s onboard sensors to maintain stability and precision.

**2.2. Orientation Control** A second PID controller managed the drone’s orientation, calculating the precise yaw angle required to align the laser pointer with the target. This controller enabled the drone to rotate smoothly and maintain accurate tracking of the target's position.

### 3. Software Implementation 

- **Simulation:** Webots was used as the primary simulation platform to model the drone’s behavior and test the control algorithms in a safe and controlled virtual environment.
- **Control Framework:** ROS2 (Robot Operating System 2) provided the middleware for communication between system components and facilitated real-time data processing and control.
- **Visualization:** RViz was utilized for visualizing the drone’s position, orientation, and the target’s movement, aiding in debugging and analysis.

### 4. Future Enhancements
- Implementation of a proximity control feature that allows the drone to approach the target to a specified minimum distance and maintain it dynamically.
- Integration of obstacle avoidance algorithms to ensure safe operation in cluttered environments.
- Expansion to multi-drone systems for collaborative tracking and target identification.

### Video preview

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/vqMEiRHj4WA/0.jpg)](https://www.youtube.com/watch?v=vqMEiRHj4WA)

## Config files


## Contributors

__JJayU__ - [Github](https://github.com/JJayU) (Jakub Junkiert, 147550)

__Yerbiff__ - [Github](https://github.com/Yerbiff) (Jarosław Kuźma 147617)

__the_HaUBe__ - [Github](https://github.com/theHaUBe) (Hubert Górecki 147599)

## Bugs & Feature Requests

Please report bugs and request features using the [Issue Tracker](https://github.com/JJayU/drone-dogfight/issues).


