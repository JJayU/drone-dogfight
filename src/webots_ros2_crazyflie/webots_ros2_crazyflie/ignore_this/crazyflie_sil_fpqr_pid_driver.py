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
import numpy as np

from std_msgs.msg import Empty
from nav_msgs.msg import Odometry

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Twist

from ..sil_mtoro_control import SILMotorCtrl
import cffirmware as cffirmware # crazyflie firmware

class CrazyflieDriver(SILMotorCtrl):
    
    def init(self, webots_node, properties):
        
        super().init(webots_node, properties)

         # Declare Subscriptions and Publishers
        self.target_cmd_vel = Twist()
        self.cf_sil_driver.create_subscription(Twist, '/{}/cmd_vel'.format(self.namespace), self.setpoint_callback, 1)
        self.stop_subscription  = self.cf_sil_driver.create_subscription(Empty, '/stop', self.stop, 10)
        self.odom_publisher     = self.cf_sil_driver.create_publisher(Odometry, '/{}/odom'.format(self.namespace), 10)
        
        # Initialize cf classes
        cffirmware.controllerPidInit()
        
    def step(self):

        rclpy.spin_once(self.cf_sil_driver, timeout_sec=0)
        self.time = self.get_time()

        self.update_current_pose()
        self.publish_odometry()
        self.update_cf_state()
        self.update_cf_sensors()
        self.check_safety_area()

        if self.initialization:
            self.initialization = False
            self.init_setpoint()

        self.compute_setpoint()

        ## Firmware PID bindings
        tick = 100 #this value makes sure that the position controller and attitude controller are always always initiated
        cffirmware.controllerPid(self.control, self.setpoint, self.sensors, self.state, tick)

        ## 
        cmd_roll    =  math.radians(self.control.roll)      # rad/s 
        cmd_pitch   =  math.radians(self.control.pitch)     # rad/s
        cmd_yaw     = -math.radians(self.control.yaw)       # rad/s
        cmd_thrust  = self.control.thrust                   # uint (PWM)

        if self.emergency_stop:
            cmd_roll    = 0.0
            cmd_pitch   = 0.0
            cmd_yaw     = 0.0
            cmd_thrust  = 0.0

        string = "Thrust = {:.0f}\tRoll = {:.4f}\tPitch = {:.4f}\tYaw = {:.4f}".format(cmd_thrust, cmd_roll, cmd_pitch, cmd_yaw)
        # print(string)
        
        self.send_motor_cmd(cmd_thrust, cmd_roll, cmd_pitch, cmd_yaw)

        self.past_position = self.current_pose.position
        self.udpate_past_time()
        
    def init_setpoint(self):

        # Initialize setpoint modes
        self.setpoint.mode.roll  = cffirmware.modeVelocity
        self.setpoint.mode.pitch = cffirmware.modeVelocity
        self.setpoint.mode.yaw   = cffirmware.modeVelocity
        self.setpoint.mode.z     = cffirmware.modeDisable    # needed to use thrust as setpoint

    def compute_setpoint(self):
        # Conversion constants
        mg_newton = 0.372652    # Thrust in Newton to take_off
        mg_pwm = 38615.0        # Thrust in PWM units to take_off with mass 0.038 (webots), scaling = 800
        newton2pwm = mg_pwm/mg_newton

        # Thrust limits
        thrust_min = 10001
        thrust_max = 60000
        
        self.setpoint.attitude.roll         = math.degrees(self.target_cmd_vel.angular.x)    # deg/s
        # NOTE: minus @ pitch
        self.setpoint.attitude.pitch        = math.degrees(-self.target_cmd_vel.angular.y)    # deg/s
        self.setpoint.attitudeRate.yaw      = math.degrees(0)    # deg/s
        self.setpoint.thrust                = np.clip(thrust_min, self.target_cmd_vel.linear.z*newton2pwm, thrust_max)   # PWM units

        # self.setpoint.attitudeRate.roll  = math.degrees(self.target_cmd_vel.angular.x)    # deg/s
        # # NOTE: minus @ pitch
        # self.setpoint.attitudeRate.pitch = math.degrees(-self.target_cmd_vel.angular.y)    # deg/s
        # self.setpoint.attitudeRate.yaw   = math.degrees(self.target_cmd_vel.angular.z)    # deg/s
        # self.setpoint.thrust             = np.clip(thrust_min, self.target_cmd_vel.linear.z*newton2pwm, thrust_max)   # PWM units
        
    def setpoint_callback(self, msg):
        self.target_cmd_vel = msg