import rclpy
from rclpy.node import Node

class TestNode(Node):
    def __init__(self):
        super().__init__("testnode")
        self.last_time = None
        self.timer = self.create_timer(1,self.timer_callback)
    
    def timer_callback(self):
        sec,nanosec = self.get_clock().now().seconds_nanoseconds()
        if self.last_time is None:
            self.last_time = sec + nanosec*1e-9
            return
        dt = sec + nanosec*1e-9 - self.last_time
        self.last_time = sec + nanosec*1e-9
        print(dt)

rclpy.init()
node = TestNode()
rclpy.spin(node)
rclpy.shutdown()