import rclpy
from sensor_msgs.msg import Image
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
import numpy as np

class Alight(Node):
    def __init__(self):
        super().__init__('alight_node')
        self.bridge = CvBridge()
        self.subscription0 = self.create_subscription(
            Image,
            '/door1_out', 
            self.call_back,
            1
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.yolo_model = YOLO('yolov8x.pt').to(self.device)
        self.publisher_ = self.create_publisher(Image, 'test_image', 1)
    
    def call_back(self, msg):
        # ROS 메시지를 OpenCV 이미지로 변환
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    

        image = self.run_yolo(cv_image)
        
        # BGR 이미지를 RGB로 변환
        half_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # RGB 이미지를 ROS 이미지 메시지로 변환
        half_image_msg = self.bridge.cv2_to_imgmsg(half_image_rgb, encoding='rgb8')
        self.publisher_.publish(half_image_msg)
        
        # 로그 메시지 출력
        self.get_logger().info("Published Test image")
    
    
    def run_yolo(self,image):
        results = self.yolo_model(image , classes = 0 , verbose=False)
        boxes = results[0].boxes
        
        for box in boxes:
            b = box.xyxy[0]
            x1 , y1 , x2 ,y2 = b.tolist()
            cv2.rectangle(image , (int(x1) , int(y1)) , (int(x2) , int(y2)) , (0,0,255),2)
            center_x = int((x1 + x2)//2)
            center_y = int((y1+y2)//2)
            
            
        return image
            
        

def main(args=None):
    rclpy.init(args=args)
    
    alight_node = Alight()
    
    rclpy.spin(alight_node)
    
    alight_node.destroy_node()
    rclpy.shutdown()
    
if __name__ == "__main__":
    main()
