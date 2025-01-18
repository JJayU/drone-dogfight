# Copyright 2023 Matthew Lock.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bitcraze Crazyflie driver."""

import rclpy
import math
from geometry_msgs.msg import Twist

import cffirmware as firm

class CrazyflieDriver:
    def __init__(self):
        # Additional teleop control variables
        self.current_linear_x = 0.0
        self.current_linear_y = 0.0
        self.current_linear_z = 0.0
        self.current_angular_z = 0.0

    def init(self, webots_node, properties):
        self.__robot = webots_node.robot
        self.__timestep = int(self.__robot.getBasicTimeStep())

        # Sensors
        self.__gps = self.__robot.getDevice('gps')
        self.__gyro = self.__robot.getDevice('gyro')
        self.__imu = self.__robot.getDevice('inertial_unit')

        # Propellers
        self.m1_motor = self.__robot.getDevice('m1_motor')
        self.m2_motor = self.__robot.getDevice('m2_motor')
        self.m3_motor = self.__robot.getDevice('m3_motor')
        self.m4_motor = self.__robot.getDevice('m4_motor')
        
        self.__propellers = [
            self.m1_motor,
            self.m2_motor,
            self.m3_motor,
            self.m4_motor
        ]
        
        for propeller in self.__propellers:
            propeller.setPosition(float('inf'))
            propeller.setVelocity(0)
            
        # Firmware initialization
        firm.controllerPidInit()
        self.state      = firm.state_t()
        self.sensors    = firm.sensorData_t()
        self.setpoint   = firm.setpoint_t()
        self.control    = firm.control_t()
        self.tick       = 100 #this value makes sure that the position controller and attitude controller are always always initiated
        
        # ROS interface
        rclpy.init(args=None)
        self.__node = rclpy.create_node('crazyflie_sil_pid_driver')

        # Create subscriber for twist messages
        self.twist_subscriber = self.__node.create_subscription(
            Twist,
            '/cmd_vel',
            self.twist_callback,
            10
        )

    def twist_callback(self, msg):
        """
        Callback for receiving Twist messages from teleop_twist_keyboard.
        Stores the current linear and angular velocities.
        """
        self.current_linear_x = msg.linear.x
        self.current_linear_y = msg.linear.y
        self.current_linear_z = msg.linear.z
        self.current_angular_z = msg.angular.z
        
        self.__node.get_logger().info(f"Received velocities - Linear: x={msg.linear.x}, y={msg.linear.y}, z={msg.linear.z}, Angular z={msg.angular.z}")

    def smooth_angle_transition(self, current, desired, max_change_rate):
        diff = desired - current
        change = max(min(diff, max_change_rate), -max_change_rate)
        return current + change
    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)
        
        ## Get measurements
        x, y, z                                     = self.__gps.getValues()
        roll_velocity, pitch_velocity, twist_yaw    = self.__gyro.getValues()
        vx, vy, vz                                  = self.__gps.getSpeedVector()
        roll, pitch, yaw                            = self.__imu.getRollPitchYaw()
                
        if math.isnan(vx):
            return
        
       ## Put measurement in state estimate
        self.state.attitude.roll = math.degrees(roll)
        self.state.attitude.pitch = -math.degrees(pitch)
        self.state.attitude.yaw = math.degrees(yaw)
        self.state.position.x = x
        self.state.position.y = y
        self.state.position.z = z
        self.state.velocity.x = vx
        self.state.velocity.y = vy
        self.state.velocity.z = vz
                
        # Put gyro in sensor data
        self.sensors.gyro.x = math.degrees(roll_velocity)
        self.sensors.gyro.y = math.degrees(pitch_velocity)
        self.sensors.gyro.z = math.degrees(twist_yaw)
        
        # Parametry kontroli nachylenia
        max_tilt_angle = 15.0  # maksymalny kąt nachylenia w stopniach
        max_angle_change_rate = 10.0  # stopni na krok czasowy
        
        # Konwersja prędkości liniowych na kąty nachylenia
        desired_pitch = -self.current_linear_x * (max_tilt_angle / 1.0)
        desired_roll = self.current_linear_y * (max_tilt_angle / 1.0)
        
        # Ograniczenie kątów
        desired_pitch = max(min(desired_pitch, max_tilt_angle), -max_tilt_angle)
        desired_roll = max(min(desired_roll, max_tilt_angle), -max_tilt_angle)
        
        # Płynne przejście do pożądanych kątów
        current_pitch = self.state.attitude.pitch
        current_roll = self.state.attitude.roll
        
        smooth_pitch = self.smooth_angle_transition(current_pitch, desired_pitch, max_angle_change_rate)
        smooth_roll = self.smooth_angle_transition(current_roll, desired_roll, max_angle_change_rate)
        
        ## Fill in Setpoints
        # Kontrola wysokości
        self.setpoint.mode.z = firm.modeVelocity
        self.setpoint.velocity.z = self.current_linear_z
        
        # Kontrola nachylenia (pitch i roll)
        self.setpoint.mode.pitch = firm.modeAbs
        self.setpoint.mode.roll = firm.modeAbs
        self.setpoint.attitude.pitch = smooth_pitch
        self.setpoint.attitude.roll = smooth_roll
        
        # Kontrola obrotu (yaw)
        self.setpoint.mode.yaw = firm.modeVelocity
        self.setpoint.attitudeRate.yaw = math.degrees(self.current_angular_z)
        
        ## Firmware PID bindings
        firm.controllerPid(
            self.control, 
            self.setpoint,
            self.sensors,
            self.state,
            self.tick
        )
        
        ## Motor commands
        cmd_roll    =   math.radians(self.control.roll)
        cmd_pitch   =   math.radians(self.control.pitch)
        cmd_yaw     = - math.radians(self.control.yaw)
        cmd_thrust  =   self.control.thrust
        
        ## Motor mixing
        motorPower_m1 =  cmd_thrust - cmd_roll + cmd_pitch + cmd_yaw
        motorPower_m2 =  cmd_thrust - cmd_roll - cmd_pitch - cmd_yaw
        motorPower_m3 =  cmd_thrust + cmd_roll - cmd_pitch + cmd_yaw
        motorPower_m4 =  cmd_thrust + cmd_roll + cmd_pitch - cmd_yaw
        
        scaling = 1000 ##Todo, remove necessity of this scaling (SI units in firmware)
        self.m1_motor.setVelocity(-motorPower_m1/scaling)
        self.m2_motor.setVelocity(motorPower_m2/scaling)
        self.m3_motor.setVelocity(-motorPower_m3/scaling)
        self.m4_motor.setVelocity(motorPower_m4/scaling)
        
        #ros info
        # self.__node.get_logger().info(f" GPS Coordinates frame : {self.__gps.getCoordinateSystem()}")
        # print(f" GPS Coordinates frame : {self.__gps.getCoordinateSystem()}")