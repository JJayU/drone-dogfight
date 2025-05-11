import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import zmq
import struct

class MotorCommandRelay(Node):
    def __init__(self):
        super().__init__('motor_command_relay')

        # Subskrypcja tematu /motor_command
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/motor_power',
            self.listener_callback,
            10
        )

        # Konfiguracja ZeroMQ (PUSH socket)
        context = zmq.Context()
        self.socket = context.socket(zmq.PUSH)
        self.socket.bind("tcp://*:5555")  # serwer nasłuchuje

        self.get_logger().info("Node uruchomiony i nasłuchuje na /motor_command")

    def listener_callback(self, msg):
        if len(msg.data) != 4:
            self.get_logger().warn("Otrzymano dane, ale nie mają 4 elementów!")
            return

        # Spakowanie 4 floatów jako bajty (little-endian)
        packed = struct.pack('<4f', *msg.data)
        self.socket.send(packed)
        # self.get_logger().info(f"Wysłano przez ZeroMQ: {msg.data}")

def main(args=None):
    rclpy.init(args=args)
    node = MotorCommandRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.socket.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
