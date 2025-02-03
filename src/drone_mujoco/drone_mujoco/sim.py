import mujoco
import mujoco_viewer
import numpy as np
import rclpy
from rclpy.node import Node
import os
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Imu

###
# Drone simulation using Mujoco
# Provides a simulation and a ROS2 interface for a Crazyflie 2.1 model in Mujoco
###

class SimNode(Node):
    def __init__(self):
        super().__init__('sim_node')

        path = os.path.join(os.getcwd(), 'build/drone_mujoco/model/scene.xml')

        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        self.model.opt.timestep = 0.01
        self.create_timer(0.01, self.loop)

        self.running = True

        self.motor_subscription = self.create_subscription(
            Float32MultiArray,
            'motor_power',
            self.motor_listener_callback,
            10
        )
        self.motor_subscription

        self.gps_publisher = self.create_publisher(PointStamped, 'gps', 10)
        self.imu_publisher = self.create_publisher(Imu, 'imu', 10)

    def motor_listener_callback(self, msg):
        if len(msg.data) == 4:
            m1, m2, m3, m4 = msg.data
            # self.get_logger().info(f'Otrzymano moce silników: {m1}, {m2}, {m3}, {m4}')
            self.set_control(m1, m2, m3, m4)
        else:
            self.get_logger().warn('Otrzymano niewłaściwą liczbę wartości!')

    def set_control(self, m1, m2, m3 ,m4):
        self.data.ctrl[0] = m1
        self.data.ctrl[1] = m2
        self.data.ctrl[2] = m3
        self.data.ctrl[3] = m4

    def publish_state(self):
        x, y, z = self.data.qpos[0:3]

        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg() 
        msg.header.frame_id = "map"
        msg.point.x, msg.point.y, msg.point.z = x, y, z

        self.gps_publisher.publish(msg)

        quaternion = self.data.qpos[3:7]

        msg2 = Imu()
        msg2.orientation.w = quaternion[0]
        msg2.orientation.x = quaternion[1]
        msg2.orientation.y = quaternion[2]
        msg2.orientation.z = quaternion[3]

        self.imu_publisher.publish(msg2)
        

    def loop(self):

        if self.viewer.is_alive:
            mujoco.mj_step(self.model, self.data)

            # print(self.data.qpos[:])

            self.set_control(0.03, 0.04, 0.03, 0.04)

            self.viewer.render()

            self.publish_state()

        else:
            self.running = False
            self.destroy_node()

def main():
    rclpy.init()
    node = SimNode()
    try:
        while rclpy.ok() and node.running:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
