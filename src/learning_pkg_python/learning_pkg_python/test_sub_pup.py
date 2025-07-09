import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PublisherSubscriberNode(Node):

    def __init__(self):
        super().__init__('publisher_subscriber_node')

        self.publisher = self.create_publisher(String, 'out_topic', 10)

        self.subscription = self.create_subscription(
            String,
            'in_topic', 
            self.listener_callback,  
            10
        )

        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello, ROS2 from Publisher!'
        self.publisher.publish(msg) 
        self.get_logger().info(f'Publishing: "{msg.data}"')

    def listener_callback(self, msg):

        self.get_logger().info(f'Received: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)

    node = PublisherSubscriberNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
