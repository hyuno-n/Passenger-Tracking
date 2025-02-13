import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import numpy as np
import cv2
import os


class Raw_Image_Collect(Node):
    qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_ALL,
        depth=1000
    )

    @classmethod
    def get_qos(cls):
        return cls.qos_profile

    def __init__(self):
        super().__init__('Raw_Image_Collect')

        # self.create_subscription(
        #     Image ,
        #     '/camera0/image_raw',
        #     self.call_back_0,
        #     Raw_Image_Collect.qos_profile
        # )

        # self.create_subscription(
        #     Image,
        #     '/camera2/image_raw',
        #     self.call_back_2,
        #     Raw_Image_Collect.qos_profile
        # )

        camera0 = Subscriber(self , Image , '/camera0/image_raw')
        camera2 = Subscriber(self , Image , 'camera2/image_raw')
        self.ts = ApproximateTimeSynchronizer(
            [camera0 , camera2] , queue_size=1000 , slop=0.1
        )

        self.ts.registerCallback(self.stack_callback)

        self.publishers_stack = self.create_publisher(Image , 'Stack' , Raw_Image_Collect.qos_profile)

    def stack_callback(self , cam0 , cam2 ):
        self.get_logger().info(
            f"cam0 Timestamp: {cam0.header.stamp}, cam2 Timestamp: {cam2.header.stamp}"
        )

        cam0 = CvBridge.imgmsg_to_cv2(cam0 , desired_encoding = 'bgr8')
        cam2 = CvBridge.imgmsg_to_cv2(cam2 , desired_encoding = 'bgr8')

        combined_image = np.hstack((cam0 , cam2))

        combined_image_msg = CvBridge.cv2_to_imgmsg(combined_image , encoding='bgr8')

        self.publishers_stack.publish(combined_image_msg) 
        
        


def main(args=None):
    rclpy.init(args=args)

    # TimeSynchronize 노드 생성
    node = Raw_Image_Collect()

    try:
        rclpy.spin(node)  # 노드 실행
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()