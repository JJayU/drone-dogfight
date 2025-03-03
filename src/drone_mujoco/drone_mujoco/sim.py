import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
import os
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Imu
from ament_index_python.packages import get_package_share_directory
import time

class SimNode(Node):
    def __init__(self, use_gui=False):
        super().__init__('sim_node')
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
        self.create_timer(self.model.opt.timestep, self.loop)
        self.running = True
        
        self.prev_time = 0.0

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
            self.set_control(msg.data)
        else:
            self.get_logger().warn('Invalid motor command length!')

    def set_control(self, motor_commands):
        for i in range(4):
            self.data.ctrl[i] = np.clip(motor_commands[i], 0, 1)

    def publish_state(self):
        x, y, z = self.data.qpos[0:3]

        gps_msg = PointStamped()
        gps_msg.header.stamp = self.get_clock().now().to_msg()
        gps_msg.header.frame_id = "map"
        gps_msg.point.x, gps_msg.point.y, gps_msg.point.z = x, y, z
        self.gps_publisher.publish(gps_msg)

        quaternion = self.data.qpos[3:7]

        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = "base_link"
        imu_msg.orientation.w = quaternion[0]
        imu_msg.orientation.x = quaternion[1]
        imu_msg.orientation.y = quaternion[2]
        imu_msg.orientation.z = quaternion[3]
        self.imu_publisher.publish(imu_msg)

    def loop(self):
        mujoco.mj_step(self.model, self.data)
        
        print(f"Iterations per second: {int(1.0 / (time.time() - self.prev_time))}")
        self.prev_time = time.time()
        
        if self.viewer and self.viewer.is_alive:
            self.viewer.render()
        elif self.viewer and not self.viewer.is_alive:
            self.running = False
            self.destroy_node()

        self.publish_state()
        

def main():
    rclpy.init()
    use_gui = False  # CHANGE IF GUI NEEDED
    node = SimNode(use_gui=use_gui)
    try:
        while rclpy.ok() and node.running:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
