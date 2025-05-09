# DRONE DOGFIGHT

## MPC Controller

An MPC (Model Predictive Control) controller is an advanced control strategy that uses a dynamic model of the system to predict and optimize future behavior over a set time horizon. It computes control actions by solving an optimization problem at each time step, ensuring constraints and objectives are met.

## Files structure

 - `src/acados_testing` - Python scripts used for testing of Acados models and their integration with Mujoco simulation environment
 - `crazyflie_ros2_decription` - Visualisation models for Crazyflie2 quadcopter
 - `drone_mujoco` - Simulation environment in Mujoco, integrated with ROS2
 - `drone_tf_publisher` - Transforms publisher and visualisation launch script
 - `mpc_controller` - Contains nodes with implemented MPC controllers

## Usage

Launch visualisation using:
```bash
ros2 launch drone_tf_publisher vis_launch.launch.py
```

Start the simulation:
```bash
ros2 run drone_mujoco sim
```

Start MPC controller (6DOF):
```bash
ros2 run mpc_controller 6dof
```