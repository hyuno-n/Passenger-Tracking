import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
from ultralytics.solutions import object_counter
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def return_bbox(results, frame):

    people = []

    for r in results:
        boxes = r.boxes

        for box in boxes:
            b = box.xyxy[0]
            x1, y1, x2, y2 = b.tolist()

            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)

            people.append([x, y])

            frame = cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    return people, frame

class InOutCounting(Node):
    def __init__(self):
        super().__init__('inout_counting')

        self.subscription_door1 = self.create_subscription(
            Image,
            'door1_under', 
            self.door1_callback,
            1)
        self.subscription_door1

        self.subscription_door2 = self.create_subscription(
            Image,
            'door2_under', 
            self.door2_callback,
            1)
        self.subscription_door2
        '''
        self.subscription_door3 = self.create_subscription(
            Image,
            'door3_under', 
            self.door3_callback,
            1)
        self.subscription_door3
        '''
        self.people_door1 = []
        self.people_door2 = []
        '''
        self.people_door3 = []

        '''
        self.frame_door1 = None
        self.frame_door2 = None
        '''
        self.frame_door3 = None
        '''
        self.incnt_door1 = 0
        self.incnt_door2 = 0
        #self.incnt_door3 = 0
        self.outcnt_door1 = 0
        self.outcnt_door2 = 0
        #self.outcnt_door3 = 0

        self.bridge = CvBridge()

        self.model = YOLO('yolov8x.pt')

        self.timer = self.create_timer(0.02, self.inout_counting_pub)

        self.publisher_door1_detect = self.create_publisher(Image,'inout_cnt_door1', 1)
        self.publisher_door2_detect = self.create_publisher(Image,'inout_cnt_door2', 1)
        '''
        self.publisher_door3_detect = self.create_publisher(Image,'inout_cnt_door3', 1)
        '''

        self.publisher_in_count = self.create_publisher(Int8, 'in_count')
        self.publisher_out_count = self.create_publisher(Int8, 'out_count')

    def door1_callback(self, msg):

        self.frame_door1 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        w = self.frame_door1.shape[1]

        # Define region points (수정해야할수도?)
        region_points = [(0, 450), (0, 0), (w, 0), (w, 450)]

        # Init Object Counter
        counter = object_counter.ObjectCounter()[0]
        self.incnt_door1 = object_counter.ObjectCounter()[1]
        self.outcnt_door1 = object_counter.ObjectCounter()[2]

        counter.set_args(view_img=True,
                        reg_pts=region_points,
                        classes_names=self.model.names,
                        draw_tracks=True)
        
        tracks = self.model.track(self.frame_door1, persist=True, show=False, classes = 0)

        counter.start_counting(self.frame_door1, tracks)      

        detection_msg = self.bridge.cv2_to_imgmsg(self.frame_door1, encoding='bgr8')

        self.publisher_door1_detect.publish(detection_msg)

    def door2_callback(self, msg):

        self.frame_door2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        w = self.frame_door2.shape[1]

        # Define region points (수정해야할수도?)
        region_points = [(0, 450), (0, 0), (w, 0), (w, 450)]

        # Init Object Counter
        counter = object_counter.ObjectCounter()[0]
        self.incnt_door2 = object_counter.ObjectCounter()[1]
        self.outcnt_door2 = object_counter.ObjectCounter()[2]
        counter.set_args(view_img=True,
                        reg_pts=region_points,
                        classes_names=self.model.names,
                        draw_tracks=True)
        
        tracks = self.model.track(self.frame_door2, persist=True, show=False, classes = 0)

        counter.start_counting(self.frame_door2, tracks)     

        detection_msg = self.bridge.cv2_to_imgmsg(self.frame_door2, encoding='bgr8')

        self.publisher_door2_detect.publish(detection_msg)

    '''
    def door3_callback(self, msg):

        self.frame_door3 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        w = self.frame_door3.shape[1]

        # Define region points (수정해야할수도?)
        region_points = [(0, 450), (0, 0), (w, 0), (w, 450)]

        # Init Object Counter
        counter = object_counter.ObjectCounter()[0]
        self.incnt_door3 = object_counter.ObjectCounter()[1]
        self.outcnt_door3 = object_counter.ObjectCounter()[2]
        counter.set_args(view_img=True,
                        reg_pts=region_points,
                        classes_names=self.model.names,
                        draw_tracks=True)
        
        tracks = self.model.track(self.frame_door3, persist=True, show=False, classes = 0)

        counter.start_counting(self.frame_door3, tracks)     

        detection_msg = self.bridge.cv2_to_imgmsg(self.frame_door3, encoding='bgr8')

        self.publisher_door3_detect.publish(detection_msg)
    '''

    def inout_counting_pub(self):
        
        # 나중에 door3까지 추가해서 더하기
        self.publisher_in_count.publish(self.incnt_door1 + self.incnt_door2)
        self.publisher_out_count.publish(self.outcnt_door1 + self.outcnt_door2)


def main(args=None):
    rclpy.init(args=args)

    node = InOutCounting()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()