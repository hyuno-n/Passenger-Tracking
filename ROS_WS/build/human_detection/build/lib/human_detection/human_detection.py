import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import cv2
import torch
class HumanDetection(Node):
    def __init__(self):
        super().__init__('human_detection')

        self.subscription = self.create_subscription(
            Image,
            '/camera0/image_raw',  # Assuming the topic name is image_raw
            self.image_callback,1)
        self.subscription  # prevent unused variable warning
        
        self.subscription2 = self.create_subscription(
            Image,
            '/camera2/image_raw',  # Assuming the topic name is image_raw
            self.image_callback2,1)
        self.subscription2  # prevent unused variable warning
        
        self.subscription3 = self.create_subscription(
            Image,
            '/camera4/image_raw',  # Assuming the topic name is image_raw
            self.image_callback3,1)
        self.subscription3  # prevent unused variable warning
        
        self.subscription4 = self.create_subscription(
            Image,
            '/camera6/image_raw',  # Assuming the topic name is image_raw
            self.image_callback4,1)
        self.subscription4  # prevent unused variable warning
        
        
        self.subscription5 = self.create_subscription(
            Image,
            '/camera8/image_raw',  # Assuming the topic name is image_raw
            self.image_callback5,1)
        self.subscription5  # prevent unused variable warning
        

        self.bridge = CvBridge()

        self.model = YOLO('yolov8x.pt')

        self.publisher_1 = self.create_publisher(Image,'detection_result1', 1)
        self.publisher_2 = self.create_publisher(Image,'detection_result2', 1)        
        self.publisher_3 = self.create_publisher(Image,'detection_result3', 1)
        self.publisher_4 = self.create_publisher(Image,'detection_result4', 1)
        self.publisher_5 = self.create_publisher(Image,'detection_result5', 1)
        


    def image_callback(self, msg):
        # ROS 메시지를 OpenCV 이미지로 변환
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 이미지 크기 조정을 위한 목표 크기 설정
        target_width = 640
        target_height = 480

        # cv2.resize 함수를 사용하여 이미지 크기 조정
        resized_frame = cv2.resize(frame, (target_width, target_height))

        # YOLOv8 모델에 크기가 조정된 프레임을 사용
        results = self.model(resized_frame, classes=0)

        person = []

        # 결과에서 사람 객체를 추출
        for r in results:
            boxes = r.boxes

            for box in boxes:
                b = box.xyxy[0]
                x1, y1, x2, y2 = b.tolist()

                # 조정된 크기를 기준으로 객체의 좌표 계산
                person.append([int(x1), int(y1), int(x2), int(y2)])


        print(person)

        detection_image = results[0].plot()
        # cv2.imshow("plot", plots)

        detection_msg = self.bridge.cv2_to_imgmsg(detection_image, encoding='bgr8')

        self.publisher_1.publish(detection_msg)

    def image_callback2(self, msg):
        # ROS 메시지를 OpenCV 이미지로 변환
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 이미지 크기 조정을 위한 목표 크기 설정
        target_width = 640
        target_height = 480

        # cv2.resize 함수를 사용하여 이미지 크기 조정
        resized_frame = cv2.resize(frame, (target_width, target_height))

        # YOLOv8 모델에 크기가 조정된 프레임을 사용
        results = self.model(resized_frame, classes=0)

        person = []

        # 결과에서 사람 객체를 추출
        for r in results:
            boxes = r.boxes

            for box in boxes:
                b = box.xyxy[0]
                x1, y1, x2, y2 = b.tolist()

                # 조정된 크기를 기준으로 객체의 좌표 계산
                person.append([int(x1), int(y1), int(x2), int(y2)])
                print(person)

        detection_image = results[0].plot()
        # cv2.imshow("plot", plots)

        detection_msg = self.bridge.cv2_to_imgmsg(detection_image, encoding='bgr8')

        self.publisher_2.publish(detection_msg)
        
                
    def image_callback3(self, msg):
        # ROS 메시지를 OpenCV 이미지로 변환
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 이미지 크기 조정을 위한 목표 크기 설정
        target_width = 640
        target_height = 480

        # cv2.resize 함수를 사용하여 이미지 크기 조정
        resized_frame = cv2.resize(frame, (target_width, target_height))

        # YOLOv8 모델에 크기가 조정된 프레임을 사용
        results = self.model(resized_frame, classes=0)

        person = []

        # 결과에서 사람 객체를 추출
        for r in results:
            boxes = r.boxes

            for box in boxes:
                b = box.xyxy[0]
                x1, y1, x2, y2 = b.tolist()

                # 조정된 크기를 기준으로 객체의 좌표 계산
                person.append([int(x1), int(y1), int(x2), int(y2)])            

        print(person)

        detection_image = results[0].plot()
        # cv2.imshow("plot", plots)

        detection_msg = self.bridge.cv2_to_imgmsg(detection_image, encoding='bgr8')

        self.publisher_3.publish(detection_msg)

    def image_callback4(self, msg):
        # ROS 메시지를 OpenCV 이미지로 변환
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 이미지 크기 조정을 위한 목표 크기 설정
        target_width = 640
        target_height = 480

        # cv2.resize 함수를 사용하여 이미지 크기 조정
        resized_frame = cv2.resize(frame, (target_width, target_height))

        # YOLOv8 모델에 크기가 조정된 프레임을 사용
        results = self.model(resized_frame, classes=0)

        person = []

        # 결과에서 사람 객체를 추출
        for r in results:
            boxes = r.boxes

            for box in boxes:
                b = box.xyxy[0]
                x1, y1, x2, y2 = b.tolist()

                # 조정된 크기를 기준으로 객체의 좌표 계산
                person.append([int(x1), int(y1), int(x2), int(y2)])
                
        detection_image = results[0].plot()
        # cv2.imshow("plot", plots)

        detection_msg = self.bridge.cv2_to_imgmsg(detection_image, encoding='bgr8')

        self.publisher_4.publish(detection_msg)

    def image_callback5(self, msg):
        # ROS 메시지를 OpenCV 이미지로 변환
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 이미지 크기 조정을 위한 목표 크기 설정
        target_width = 640
        target_height = 480

        # cv2.resize 함수를 사용하여 이미지 크기 조정
        resized_frame = cv2.resize(frame, (target_width, target_height))

        # YOLOv8 모델에 크기가 조정된 프레임을 사용
        results = self.model(resized_frame, classes=0)

        person = []

        # 결과에서 사람 객체를 추출
        for r in results:
            boxes = r.boxes

            for box in boxes:
                b = box.xyxy[0]
                x1, y1, x2, y2 = b.tolist()

                # 조정된 크기를 기준으로 객체의 좌표 계산
                person.append([int(x1), int(y1), int(x2), int(y2)])

        print(person)

        detection_image = results[0].plot()
        # cv2.imshow("plot", plots)

        detection_msg = self.bridge.cv2_to_imgmsg(detection_image, encoding='bgr8')

        self.publisher_5.publish(detection_msg)

def main(args=None):
    # ROS 초기화
    rclpy.init(args=args)

    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 노드 생성
    node = HumanDetection()

    # ROS 노드 실행
    rclpy.spin(node)

    # 노드 종료
    node.destroy_node()
    
    # ROS 종료
    rclpy.shutdown()


if __name__ == '__main__':
    main()