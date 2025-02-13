import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer , Cache , Duration, Time
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import numpy as np
import cv2
import os
import time

class TimeSynchronize(Node):
    # QoS 설정: 신뢰성 있고 모든 메시지 유지 (depth: 100)
    qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_ALL,
        depth=100
    )

    # 카메라 목록 정의 (key는 카메라 번호, value는 카메라 이름)
    all_cameras = {
        0: 'camera0', 
        2: 'camera2', 
        4: 'camera4', 
    }

    def __init__(self):
        super().__init__("time_synchronize")
        self.get_logger().warn("Data Collect gpt Run...")
        self.bridge = CvBridge()
        self.allow_time = 0.5  # Allowed time difference in seconds

        self.SCEN_NAME = 'scen2'
        self.SCEN_NAME = f'{self.SCEN_NAME}_{str(self.allow_time)}GPT'
        self.base_path = f'dataset/{self.SCEN_NAME}'
        self.camera_paths = {
            index: os.path.join(self.base_path, f'cam{index}') for index in TimeSynchronize.all_cameras.keys()
        }
        for path in self.camera_paths.values():
            os.makedirs(path, exist_ok=True)

        self.call_count = 0  # Number of synchronized messages processed

        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_ALL
        )

        # Create subscription for cam0
        self.cam0_sub = self.create_subscription(
            Image, 'camera0/image_raw', self.cam0_callback, qos_profile=self.qos_profile
        )

        # Create message_filters.Subscriber and Cache for cam2 and cam4
        self.cam2_sub = Subscriber(self, Image, 'camera2/image_raw', qos_profile=self.qos_profile)
        self.cam4_sub = Subscriber(self, Image, 'camera4/image_raw', qos_profile=self.qos_profile)

        self.cam2_cache = Cache(self.cam2_sub, cache_size=1000)
        self.cam4_cache = Cache(self.cam4_sub, cache_size=1000)

    def cam0_callback(self, cam0_msg):
        cam0_time = self.get_msg_time(cam0_msg)
        self.get_logger().info(f"Received cam0 message at time {cam0_time.nanoseconds}")
        # Get messages from caches
        lower_bound = cam0_time - Duration(seconds=self.allow_time)
        upper_bound = cam0_time + Duration(seconds=self.allow_time)
        cam2_msgs = self.cam2_cache.getInterval(lower_bound, upper_bound)
        cam4_msgs = self.cam4_cache.getInterval(lower_bound, upper_bound)

        cam2_msg = self.find_closest_message(cam2_msgs, cam0_time)
        cam4_msg = self.find_closest_message(cam4_msgs, cam0_time)

        # Process the messages
        self.process_messages(cam0_msg, cam2_msg, cam4_msg)

    def find_closest_message(self, msgs, target_time):
        if not msgs:
            return None
        closest_msg = min(msgs, key=lambda msg: abs((self.get_msg_time(msg) - target_time).nanoseconds))
        return closest_msg

    def get_msg_time(self, msg):
        return Time.from_msg(msg.header.stamp)

    def process_messages(self, cam0_msg, cam2_msg, cam4_msg):
        cam0_image = self.bridge.imgmsg_to_cv2(cam0_msg, desired_encoding="bgr8")
        if cam0_image is not None:
            cv2.imwrite(os.path.join(self.camera_paths[0], f"{self.call_count}.jpg"), cam0_image)

        if cam2_msg:
            cam2_image = self.bridge.imgmsg_to_cv2(cam2_msg, desired_encoding="bgr8")
            if cam2_image is not None:
                cv2.imwrite(os.path.join(self.camera_paths[2], f"{self.call_count}.jpg"), cam2_image)
                self.get_logger().info(f"Camera 2 message saved")
        else:
            self.get_logger().warn(f"Camera 2 message not found within time tolerance")

        if cam4_msg:
            cam4_image = self.bridge.imgmsg_to_cv2(cam4_msg, desired_encoding="bgr8")
            if cam4_image is not None:
                cv2.imwrite(os.path.join(self.camera_paths[4], f"{self.call_count}.jpg"), cam4_image)
                self.get_logger().info(f"Camera 4 message saved")
        else:
            self.get_logger().warn(f"Camera 4 message not found within time tolerance")

        self.call_count += 1
        self.get_logger().info(f"Processed synchronized messages #{self.call_count}")

def main(args=None):
    rclpy.init(args=args)
    node = TimeSynchronize()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
