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

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Twist

import cffirmware as firm

class CrazyflieDriver:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot
        self.__timestep = int(self.__robot.getBasicTimeStep())
        self.robot_name = self.__robot.getName()

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
        self.__node = rclpy.create_node('crazyflie_sil_pid_attitude_driver')
        
        self.pose_publisher = self.__node.create_publisher(
            PoseStamped,
            f"{self.robot_name}/pose",
            1
        )
        self.twist_publisher = self.__node.create_publisher(
            TwistStamped,
            f"{self.robot_name}/twist",
            1
        )
        self.cmd_vel_subscription = self.__node.create_subscription(
            Twist,
            f"{self.robot_name}/cmd_vel",
            self.cmd_vel_callback,
            1
        )
        
        # States
        self.go_to_start_bool   = True
        self.hovering_time      = 0
        self.start_mission_time = 0
        self.last_time          = self.__robot.getTime()
        self.__node.get_logger().info(f"{str(self.__timestep)}")

        #roslog info
        self.logger = self.__node.get_logger()
        self.logger.debug(f"Robot {self.robot_name} initialized")
        
        # Initial position
        self.des_x = 0
        self.des_y = 0
        self.des_z = 1
        self.yaw_desired_degrees = 0
        
    def cmd_vel_callback(self, msg:Twist):
        
        self.des_x = msg.linear.x
        self.des_y = msg.linear.y
        self.des_z = msg.linear.z
        self.yaw_desired_degrees = msg.angular.z        
        
    def go_to_pose(self):
        
        yawDesired = 0
        
        ## Fill in Setpoints
        self.setpoint.mode.z            =   firm.modeAbs
        self.setpoint.mode.y            =   firm.modeVelocity
        self.setpoint.mode.x            =   firm.modeVelocity
        self.setpoint.mode.yaw          =   firm.modeVelocity
        # self.setpoint.mode.roll         =   firm.modeVelocity
        # self.setpoint.mode.pitch        =   firm.modeVelocity
        # self.setpoint.mode.pitch        =   firm.modeDisable
        
        # self.setpoint.position.x        = self.des_x
        # self.setpoint.position.y        = self.des_y
        self.setpoint.position.z        = self.des_z
        
        # Setpoint of 8 rad/s -> 0.5 rad/s
        rad_conv = 8/0.5        
        yaw_desired_rad = math.radians(self.yaw_desired_degrees*rad_conv)
        
        self.setpoint.velocity.x = self.des_x
        self.setpoint.velocity.y = self.des_y
        self.setpoint.attitudeRate.yaw = math.degrees(yaw_desired_rad)
        
        # self.setpoint.attitudeRate.roll     = 0
        # self.setpoint.attitudeRate.pitch    = 0
        # self.setpoint.attitude.yaw      = math.degrees(yaw_desired_rad)
        
        self.setpoint.velocity_body     = False
        
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
        
        self.scaling = 1000 ##Todo, remove necessity of this scaling (SI units in firmware)
        self.m1_motor.setVelocity(- motorPower_m1/self.scaling)
        self.m2_motor.setVelocity(  motorPower_m2/self.scaling)
        self.m3_motor.setVelocity(- motorPower_m3/self.scaling)
        self.m4_motor.setVelocity(  motorPower_m4/self.scaling)

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)
        
        ## Get measurements
        x, y, z                                         = self.__gps.getValues()
        vx, vy, vz                                      = self.__gps.getSpeedVector()
        roll, pitch, yaw                                = self.__imu.getRollPitchYaw()
        roll_velocity, pitch_velocity, yaw_velocity     = self.__gyro.getValues()
                
        if math.isnan(vx):
            return
        
        ## Put measurement in state estimate
        self.state.attitude.roll    =   math.degrees(roll)
        self.state.attitude.pitch   = - math.degrees(pitch)
        self.state.attitude.yaw     =   math.degrees(yaw)
        self.state.position.x       =   x
        self.state.position.y       =   y
        self.state.position.z       =   z
        self.state.velocity.x       =   vx
        self.state.velocity.y       =   vy
        self.state.velocity.z       =   vz
            
        # Put gyro in sensor data
        self.sensors.gyro.x = math.degrees(roll_velocity)
        self.sensors.gyro.y = math.degrees(pitch_velocity)
        self.sensors.gyro.z = math.degrees(yaw_velocity)
        
        # publish pose
        pose = PoseStamped()
        pose.header.stamp = self.__node.get_clock().now().to_msg()
        pose.header.frame_id = 'map'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = roll
        pose.pose.orientation.y = pitch
        pose.pose.orientation.z = yaw
        pose.pose.orientation.w = 1.0
        self.pose_publisher.publish(pose)
        
        twist = TwistStamped()
        twist.header.stamp = self.__node.get_clock().now().to_msg()
        twist.header.frame_id = 'map'
        twist.twist.linear.x = vx
        twist.twist.linear.y = vy
        twist.twist.linear.z = vz
        twist.twist.angular.x = roll_velocity
        twist.twist.angular.y = pitch_velocity
        twist.twist.angular.z = yaw_velocity
        self.twist_publisher.publish(twist)
        
        self.go_to_pose()