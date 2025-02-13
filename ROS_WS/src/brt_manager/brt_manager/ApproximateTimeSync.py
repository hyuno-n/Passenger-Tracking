import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import os

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
        self.get_logger().info("ApproximateTime sync...")
        # 메시지 처리 카운터
        self.call_count = 0

        # CvBridge 초기화
        self.bridge = CvBridge()

        # 각 카메라 구독자 생성 (message_filters.Subscriber 사용)
        self.subscribers = {}
        for index, name in TimeSynchronize.all_cameras.items():
            topic_name = f'{name}/image_raw'
            self.subscribers[name] = Subscriber(
                self , Image, topic_name  
            )

        # ApproximateTimeSynchronizer 설정
        self.ts = ApproximateTimeSynchronizer(
            list(self.subscribers.values()),  # 구독자 리스트
            queue_size=1000, 
            slop= 0.5,  # 허용 시간 차이 (초)
        )

        # 동기화된 메시지 콜백 등록
        self.ts.registerCallback(self.time_callback)

        # 결합된 이미지 발행자 생성
        self.publisher_stack = self.create_publisher(Image, "Time_Synchronize", TimeSynchronize.qos_profile)

        # 저장할 폴더 설정
        self.SCEN_NAME = 'ApporximateTimeSync_scen2'
        self.base_path = f'dataset/{self.SCEN_NAME}'

        # 카메라 경로를 딕셔너리에 저장
        self.camera_paths = {
            0: f'{self.base_path}/cam0',
            2: f'{self.base_path}/cam2',
            4: f'{self.base_path}/cam4',
        }

        # 폴더 생성 (for 문 사용)
        for path in self.camera_paths.values():
            os.makedirs(path, exist_ok=True)

        self.saved = True  # 이미지 저장 여부

    def time_callback(self, *msgs):
        """동기화된 메시지 처리 콜백"""
        # cam0 메시지 추출 (첫 번째 메시지가 cam0 기준)
        cam0_msg = msgs[0] if len(msgs) > 0 else None

        # 다른 카메라의 메시지를 순서대로 할당 (누락된 경우 None)
        other_msgs = msgs[1:]

        received_msg = []
        omit_msg = []
        for index , msgs in enumerate(other_msgs):
            msg_idx = list(self.camera_paths.keys())[index + 1]
            if msgs:
                received_msg.append(msg_idx)
            if msgs is None:
                omit_msg.append(msg_idx)

        for i in received_msg:
            self.get_logger().info(f"{i} message received")
        for i in omit_msg:
            self.get_logger().warn(f"{i} message lose")

        # cam0 메시지가 없으면 경고 로그
        if cam0_msg is None:
            self.get_logger().fatal("cam0 메시지가 없습니다.")
            return

        # cam0 기준이미지 msg -> img
        cam0_image = self.bridge.imgmsg_to_cv2(cam0_msg, desired_encoding="bgr8")

        # 다른 카메라 이미지 처리 (없으면 빈 이미지로 대체)
        other_images = [
            self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8") 
            if msg else np.zeros_like(cam0_image)
            for msg in other_msgs
        ]

        # cam0 + cam2
        combined_image = np.hstack((cam0_image , other_images[0]))

        # 결합된 이미지를 ROS Image 메시지로 변환
        combined_image_msg = self.bridge.cv2_to_imgmsg(combined_image, encoding="bgr8")

        # 저장 설정이 활성화된 경우 이미지 저장
        if self.saved:
            # cam0 이미지 저장
            cv2.imwrite(os.path.join(self.camera_paths[0], f"{self.call_count}.jpg"), cam0_image)

            # 다른 카메라 이미지 저장 (빈 이미지가 아닌 경우만)
            for i, img in enumerate(other_images):
                if not np.all(img == 0):  # 빈 이미지가 아닌 경우에만 저장
                    camera_index = list(self.camera_paths.keys())[i + 1]  # i+1 번째 카메라 인덱스
                    camera_path = self.camera_paths[camera_index]
                    cv2.imwrite(os.path.join(camera_path, f"{self.call_count}.jpg"), img)

        # 결합된 이미지 발행
        self.publisher_stack.publish(combined_image_msg)
        self.call_count += 1  # 카운터 증가

    def pass_(self , msg):
        pass
def main(args=None):
    rclpy.init(args=args)
    node = TimeSynchronize()

    try:
        rclpy.spin(node)  # 노드 실행
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
