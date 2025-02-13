from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import numpy as np
import rclpy
import cv2
import os

# class Raw_Image_Collect(Node):
#     qos_profile = QoSProfile(
#         reliability=ReliabilityPolicy.RELIABLE,
#         history=HistoryPolicy.KEEP_ALL,
#         depth=1000
#     )

#     camea_list = ['door1','under','door1_back','door2','under_2']

#     SCEN_NAME = 'SCEN1'

#     def __init__(self):
#         super().__init__('raw_file')
#         self.get_logger().info("Raw_Image_Collect run...")

#         self.bridge = CvBridge()  # CvBridge 인스턴스 생성

#         camera0 = Subscriber(self, Image, '/camera0/image_raw')
#         camera2 = Subscriber(self, Image, '/camera2/image_raw')  
#         camera4 = Subscriber(self , Image , '/camera4/image_raw')
#         camera6 = Subscriber(self, Image , '/camera6/image_raw')
#         camera8 = Subscriber(self , Image , '/camera8/image_raw')


#         self.ts = ApproximateTimeSynchronizer(
#             [camera0, camera2 , camera4 , camera6  ,camera8], queue_size=1000, slop=1
#         )

#         self.ts.registerCallback(self.stack_callback)

#         self.publishers_stack = self.create_publisher(Image, 'Stack', Raw_Image_Collect.qos_profile)


#         self.count = 0
#         self.saved = False

#     def stack_callback(self, cam0, cam2 , cam4 , cam6 , cam8):
#         self.get_logger().info(
#             f"cam0 Timestamp: {cam0.header.stamp.sec}, cam2 Timestamp: {cam2.header.stamp.sec}"
#         )


#         cam0 = self.bridge.imgmsg_to_cv2(cam0, desired_encoding='bgr8')  # 인스턴스 사용
#         cam2 = self.bridge.imgmsg_to_cv2(cam2, desired_encoding='bgr8')  # 인스턴스 사용

#         if self.saved:
#             os.makedirs(self.door1_name , exist_ok = True)
#             os.makedirs(self.under_name , exist_ok = True)
#             cv2.imwrite(os.path.join(self.door1_name , f"{self.count}.jpg"), cam0)
#             cv2.imwrite(os.path.join(self.under_name , f"{self.count}.jpg") , cam2)

#         self.count += 1 

#         combined_image = np.hstack((cam0, cam2))

#         combined_image_msg = self.bridge.cv2_to_imgmsg(combined_image, encoding='bgr8')

#         self.publishers_stack.publish(combined_image_msg)

# def main(args=None):
#     rclpy.init(args=args)

#     # Raw_Image_Collect 노드 생성
#     node = Raw_Image_Collect()

#     try:
#         rclpy.spin(node)  # 노드 실행
#     except KeyboardInterrupt:
#         node.get_logger().info('Shutting down...')
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()



import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import os
import cv_bridge
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from collections import deque
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import os
import cv_bridge
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from collections import deque

class Raw_Image_Collect(Node):

    SAVED = True
    all_cameras = {
        0 : 'camera0', 
        2 : 'camera2', 
        4 : 'camera4', 
        6 : 'camera6', 
        8 : 'camera8'}

    SYNC_TIME = 1.0  # 동기화 시간 범위 (초)
    BUFFER_SIZE = 100  # 버퍼 크기

    SCEN_NAME = 'scen2_TEST'

    BASE_PATH = os.path.join('dataset', "TEST", f"{SCEN_NAME}")

    for i in all_cameras.values():
        os.makedirs(os.path.join('dataset', 'TEST', f'{SCEN_NAME}', f'{i}'), exist_ok=True)

    def __init__(self):
        super().__init__('Data_collect')
        self.get_logger().warn("Data collect run ...")
        self.sync_count = 0

        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_ALL,
            depth=1000,
        )

        # 각 카메라의 메시지 버퍼를 저장할 딕셔너리
        self.buffers = {
            cam_id: deque(maxlen=self.BUFFER_SIZE) for cam_id in self.all_cameras.keys()
        }
        self.cvbridge = cv_bridge.CvBridge()

        # 각 카메라에 대한 구독자 생성
        for cam_id in self.all_cameras.keys():
            topic_name = f'/camera{cam_id}/image_raw'
            self.create_subscription(
                Image,
                topic_name,
                lambda msg, cam_id=cam_id: self.store_message(msg, cam_id),
                self.qos_profile
            )

    def store_message(self, msg, cam_id):
        """각 카메라의 메시지를 버퍼에 저장하고 cam0 메시지 수신 시 처리"""
        self.buffers[cam_id].append(msg)
        if cam_id == 0:
            self.process_messages()

    def get_msg_timestamp(self, msg):
        """메시지의 타임스탬프를 초 단위로 반환"""
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def process_messages(self):
        """cam0 메시지 수신 시 다른 카메라의 메시지와 동기화 시도"""
        cam0_msg = self.buffers[0][-1]
        cam0_stamp = self.get_msg_timestamp(cam0_msg)
        synced_messages = {'camera0': cam0_msg}

        for cam_id in self.all_cameras.keys():
            if cam_id == 0:
                continue
            best_msg = None
            min_time_diff = float('inf')
            for msg in self.buffers[cam_id]:
                msg_stamp = self.get_msg_timestamp(msg)
                time_diff = abs(cam0_stamp - msg_stamp)
                if time_diff <= self.SYNC_TIME and time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_msg = msg
            if best_msg:
                synced_messages[f'camera{cam_id}'] = best_msg
            else:
                self.get_logger().warn(f'Camera{cam_id}에서 동기화 가능한 메시지를 찾지 못했습니다.')
        
        # 동기화된 메시지 처리
        self.process_synced_messages(synced_messages)

    def process_synced_messages(self, messages):
        """동기화된 메시지를 처리하고 저장"""
        timestamp_sec = messages['camera0'].header.stamp.sec
        self.get_logger().info(f"camera0 타임스탬프 (sec): {timestamp_sec}")

        for camera, msg in messages.items():
            try:
                image = self.cvbridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                folder_name = self.all_cameras[int(camera[-1])]
                save_name = os.path.join(
                    self.BASE_PATH,
                    f"{folder_name}",
                    f"{self.sync_count}.jpg",
                )
                cv2.imwrite(save_name, image)
            except Exception as e:
                self.get_logger().error(f"{camera} 처리 중 오류 발생: {e}")
        self.sync_count += 1

def main(args=None):
    rclpy.init(args=args)
    node = Raw_Image_Collect()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

