import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import cv2
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

def return_bbox(results, frame):
    people = []

    for r in results:
        boxes = r.boxes

        for box in boxes:
            b = box.xyxy[0]
            x1, y1, x2, y2 = list(map(lambda x : int(x) , b.tolist()))

            # 박스 중심점 계산
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)
            people.append([x, y])

            # 감지된 사람 주위에 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 녹색 박스로 표시

            # 중심점 표시 (선택적)
            frame = cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # 빨간색 점

    return people, frame

def new_return_bbox(boxes , frame):
    people = []

    for i in boxes:
        x1 , y1 , x2 , y2 = i
        c_x , c_y = (x1 + x2) // 2 , (y1 + y2) // 2
        people.append([c_x , c_y])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 녹색 박스로 표시

            # 중심점 표시 (선택적)
        frame = cv2.circle(frame, (c_x, c_y), 3, (0, 0, 255), -1)  # 빨간색 점

    return people , frame

import os
class SitRecognition(Node):
    qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_ALL,
        depth=100
    )
    @classmethod
    def get_qos(cls):
        return cls.qos_profile
    def __init__(self):
        super().__init__('sit_recognition')
    
        self.subscription_door1 = self.create_subscription(
            Image,
            'door1_in',  
            self.door1_callback,
            SitRecognition.get_qos())
        self.subscription_door1 # 문1(1량 승차문)의 버스 내부 sub

        self.subscription_door2 = self.create_subscription(
            Image,
            'door2_in', 
            self.door2_callback,
            1)
        self.subscription_door2 # 문2(1량 하차문)의 버스 내부 sub
        
        self.subscription_door3 = self.create_subscription(
            Image,
            'door3_in',  
            self.door3_callback,
            1)
        self.subscription_door3 # 문3(2량 하차문)의 버스 내부 sub

        self.subscription_bus1_under = self.create_subscription(
            Image,
            'bus1_under',  
            self.bus1_front_callback,
            SitRecognition.get_qos())
        self.subscription_bus1_under # 1량 가운데의 버스 내부 sub

        self.subscription_bus1_out = self.create_subscription(
            Image,
            'bus1_out',  
            self.bus1_back_callback,
            1)
        self.subscription_bus1_out # 1량 가운데의 버스 out sub
        
        self.subscription_bus2_in = self.create_subscription(
            Image,
            'bus2_in',  
            self.bus2_callback_front,
            1)
        self.subscription_bus2_in # 2량 가운데의 버스 내부 sub
        
        self.subscription_bus2_out = self.create_subscription(
            Image,
            'bus2_out',  
            self.bus2_callback_back,
            1)
        self.subscription_bus2_out # 2량 가운데의 버스 내부 sub
        
        self.bridge = CvBridge()
        self.model = YOLO('yolo11x.pt')
        
        # 각 카메라별 감지 결과를 저장하는 리스트 초기화
        self.people_door1 = []
        self.people_door2 = []
        self.people_door3 = []
        self.people_bus1_under = []
        self.people_bus1_out = []
        self.people_bus2_in = []
        self.people_bus2_out = []

        self.frame_door1 = None
        self.frame_door2 = None
        self.frame_door3 = None
        self.frame_bus1_under = None
        self.frame_bus1_out = None
        self.frame_bus2_in = None
        self.frame_bus2_out = None

        # 좌석 좌표 초기화
        self.sits_door1 = [[(409, 247), (501, 338)], [(376, 141), (471, 227)], [(245, 249), (340, 337)], 
                           [(239, 173), (343, 245)], [(94, 248), (187, 359)], [(101, 157), (199, 243)]]
        
    
        
        self.sits_bus1_under = [[(421, 180), (586, 336)],[(170, 375), (264, 557)],[(14, 382), (156, 484)],
                                [(453, 412), (597, 598)]]
        
        self.sits_bus1_out = [[(303, 309), (411, 489)],[(419, 298), (508, 490)],[(2, 263), (166, 430)]]
        
        self.sits_door2 = [[(376, 170), (560, 338)], [(159, 136), (357, 327)]]
        
        self.sits_door3 = [[(130, 328), (223, 492)], [(15, 316), (104, 485)], [(250, 182), (357, 270)], 
                           [(244, 97), (366, 179)],[(576, 341), (595, 386)], [(488, 334), (593, 478)], 
                           [(357, 342), (482, 525)], [(84, 186), (213, 279)], [(89, 96), (210, 182)]]
        
        self.sits_bus2_in =[[(131, 392), (210, 572)], [(2, 389), (105, 552)],
                            [(459, 399), (586, 534)], [(348, 403), (449, 559)]]
        
        self.sits_bus2_out = [[(389, 217), (453, 382)],[(485, 217), (560, 393)],
                              [(61, 221), (153, 387)],[(157, 216), (250, 387)],
                              [(396, 117), (493, 169)], [(383, 65), (503, 101)], [(376, 21), (494, 60)], 
                              [(132, 108), (225, 164)], [(154, 66), (231, 104)], [(169, 26), (244, 64)]]
    
    
        self.sits_bus1_under_cam1 = [[(384, 170), (495, 251)],[(407, 262), (529, 340)],[(254, 188), (368, 265)],
                                     [(232,272), (374, 343)]]
        
    
    
    
    
    
    
        
        self.full_sit = cv2.imread("/home/krri/Desktop/ros_ws/simulation_img/full_sit.png")
        self.sit_h, self.sit_w = self.full_sit.shape[:2]
        self.full_sit_simul = cv2.resize(self.full_sit, (int(self.sit_w*0.2), int(self.sit_h*0.2)))
        self.new_sit_h, self.new_sit_w = self.full_sit_simul.shape[:2]
        self.empty_sit_simul = cv2.cvtColor(self.full_sit_simul, cv2.COLOR_BGR2GRAY)
        self.empty_sit_simul = cv2.cvtColor(self.empty_sit_simul, cv2.COLOR_GRAY2BGR)
        
        self.sits_simul_door1 = [(115, 140), (115, 175),
                                 (115+self.new_sit_w+15, 140), (115+self.new_sit_w+15, 175),
                                 (115+self.new_sit_w*2+15*2, 140), (115+self.new_sit_w*2+15*2, 175)]
        
        self.sits_simul_bus1_under = [(115+self.new_sit_w*2+15*2, 35),(115+self.new_sit_w*3+15*3, 140), 
                                      (115+self.new_sit_w*3+15*3, 175),(115+self.new_sit_w*3+15*3, 35)]
        
        self.sits_simul_bus1_out = [(115+self.new_sit_w*4+15*4, 140), (115+self.new_sit_w*4+15*4, 175),
                                    (115+self.new_sit_w*4+15*4, 35)]
        
        self.sits_simul_door2 = [(115+self.new_sit_w*5+15*5, 175), (115+self.new_sit_w*6+15*6, 175)]
        
        self.sits_simul_door3 = [(800, 140), (800, 175),
                                 (800+self.new_sit_w+15, 140), (800+self.new_sit_w+15, 175),
                                 (775, 35), (775+self.new_sit_h, 35), (775+self.new_sit_h*2, 35),
                                 (800+self.new_sit_w*2+15*2, 140), (800+self.new_sit_w*2+15*2, 175)]
        
        self.sits_simul_bus2_in = [(800+self.new_sit_w*3+15*3, 140),(800+self.new_sit_w*3+15*3, 175),
                                   (800+self.new_sit_w*3+15*3, 35), (800+self.new_sit_w*3+15*3, 70)]
        
        self.sits_simul_bus2_out = [(800+self.new_sit_w*4+15*4, 140), (800+self.new_sit_w*4+15*4, 175),
                                    (800+self.new_sit_w*4+15*4, 35), (800+self.new_sit_w*4+15*4, 70),
                                    (800+self.new_sit_w*5+15*5, 160), (800+self.new_sit_w*5+15*5+self.new_sit_h, 160),
                                    (800+self.new_sit_w*5+15*5+self.new_sit_h*2, 160), 
                                    (800+self.new_sit_w*5+15*5, 35), (800+self.new_sit_w*5+15*5+self.new_sit_h, 35), 
                                    (800+self.new_sit_w*5+15*5+self.new_sit_h*2, 35)]

        self.all_sits = (len(self.sits_simul_door1) + len(self.sits_simul_door2) + len(self.sits_simul_door3) + 
                         len(self.sits_simul_bus1_under) + len(self.sits_simul_bus2_in) + len(self.sits_bus2_out))
        
        self.timer = self.create_timer(0.02, self.sit_simulation_pub)

        self.publisher_door1_detect = self.create_publisher(Image,'detect_door1_in', 1)
        self.publisher_door2_detect = self.create_publisher(Image,'detect_door2_in', 1)
        self.publisher_door3_detect = self.create_publisher(Image,'detect_door3_in', 1)
        self.publisher_bus1_under_detect = self.create_publisher(Image,'detect_bus1_under', 1)
        self.publisher_bus1_out_detect = self.create_publisher(Image,'detect_bus1_out',1)
        self.publisher_bus2_in_detect = self.create_publisher(Image,'detect_bus2_in', 1)
        self.publisher_bus2_out_detect = self.create_publisher(Image,'detect_bus2_out', 1)
        self.publisher_sit_simulation = self.create_publisher(Image, 'bus_illustration', 1)

    def apply_nms(self, boxes, scores, iou_threshold=0.6):
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.3, nms_threshold = iou_threshold)
        if isinstance(indices, list) and len(indices) > 0:
            return [boxes[i[0]] for i in indices]
        elif isinstance(indices, np.ndarray) and indices.size > 0:
            return [boxes[i] for i in indices.flatten()]
        else:
            return []
        
    
    def nmx_box_to_cv2_loc(self , boxes):
        x1 , y1 , w, h = boxes
        x2 = x1 + w
        y2 = y1 + h

        return [x1 , y1 , x2 , y2]

  

    def door1_callback(self, msg):
        self.frame_door1 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # if self.saved:
        #     cv2.imwrite(os.path.join(self.door1_name ,f"{self.call_count}.jpg"),self.frame_door1)
        # self.call_count += 1
        results = self.model(self.frame_door1, classes=0)
        boxes = []
        scores = []
        for res in results:
            for box in res.boxes:
                if int(box.cls) == 0:
                    x1 , y1 , w , h = box.xyxy[0].tolist()
                    score = box.conf
                    boxes.append([int(x1), int(y1), int(w - x1), int(h - y1)])
                    scores.append(float(score.detach().cpu().numpy()))

        nmx_boxes = self.apply_nms(boxes , scores)
        nmx_boxes = list(map(self.nmx_box_to_cv2_loc , nmx_boxes))
        self.people_door1 , self.frame_door1 = new_return_bbox(nmx_boxes , self.frame_door1)

        # self.people_door1 = return_bbox(results, self.frame_door1)[0]
        # self.frame_door1 = return_bbox(results, self.frame_door1)[1]

        detection_msg = self.bridge.cv2_to_imgmsg(self.frame_door1, encoding='bgr8')
        self.publisher_door1_detect.publish(detection_msg)

    def door2_callback(self, msg):
        self.frame_door2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(self.frame_door2, classes=0)
        self.people_door2 = return_bbox(results, self.frame_door2)[0]
        self.frame_door2 = return_bbox(results, self.frame_door2)[1]
        detection_msg = self.bridge.cv2_to_imgmsg(self.frame_door2, encoding='bgr8')
        self.publisher_door2_detect.publish(detection_msg)
    
    def door3_callback(self, msg):
        self.frame_door3 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(self.frame_door3 , classes=0)
        self.people_door3 = return_bbox(results, self.frame_door3)[0]
        self.frame_door3 = return_bbox(results, self.frame_door3)[1]
        detection_msg = self.bridge.cv2_to_imgmsg(self.frame_door3, encoding='bgr8')
        self.publisher_door3_detect.publish(detection_msg)

    def bus1_front_callback(self, msg):
        self.frame_bus1_under = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # if self.saved:
        #     cv2.imwrite(os.path.join(self.under_name , f'{self.call_count2}.jpg') , self.frame_bus1_under)
        # self.call_count2 += 1
        results = self.model(self.frame_bus1_under, classes=0)
        boxes = []
        scores = []
        for res in results:
            for box in res.boxes:
                if int(box.cls) == 0:
                    x1 , y1 , w , h = box.xyxy[0].tolist()
                    score = box.conf
                    boxes.append([int(x1), int(y1), int(w - x1), int(h - y1)])
                    scores.append(float(score.detach().cpu().numpy()))

        nmx_boxes = self.apply_nms(boxes , scores)
        nmx_boxes = list(map(self.nmx_box_to_cv2_loc , nmx_boxes))
        self.people_bus1_under , self.frame_bus1_under = new_return_bbox(nmx_boxes , self.frame_bus1_under)


        # self.people_bus1_under = return_bbox(results, self.frame_bus1_under)[0]
        # self.frame_bus1_under = return_bbox(results, self.frame_bus1_under)[1]


        detection_msg = self.bridge.cv2_to_imgmsg(self.frame_bus1_under, encoding='bgr8')
        self.publisher_bus1_under_detect.publish(detection_msg)
    
    def bus1_back_callback(self, msg):
        self.frame_bus1_out = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(self.frame_bus1_out, classes=0)
        self.people_bus1_out = return_bbox(results, self.frame_bus1_out)[0]
        self.frame_bus1_out = return_bbox(results, self.frame_bus1_out)[1]
        detection_msg = self.bridge.cv2_to_imgmsg(self.frame_bus1_out, encoding='bgr8')
        self.publisher_bus1_out_detect.publish(detection_msg)    

    def bus2_callback_front(self, msg):
        self.frame_bus2_in = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(self.frame_bus2_in, classes=0)
        self.people_bus2_in  = return_bbox(results, self.frame_bus2_in)[0]
        self.frame_bus2_in = return_bbox(results, self.frame_bus2_in)[1]
        detection_msg = self.bridge.cv2_to_imgmsg(self.frame_bus2_in, encoding='bgr8')
        self.publisher_bus2_in_detect.publish(detection_msg) 
        
    def bus2_callback_back(self, msg):
        self.frame_bus2_out = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(self.frame_bus2_out, classes=0)
        self.people_bus2_out = return_bbox(results, self.frame_bus2_out)[0]
        self.frame_bus2_out = return_bbox(results, self.frame_bus2_out)[1]
        detection_msg = self.bridge.cv2_to_imgmsg(self.frame_bus2_out, encoding='bgr8')
        self.publisher_bus2_out_detect.publish(detection_msg)

    def sit_simulation_pub(self):
        brt_simul = cv2.imread("/home/krri/Desktop/ros_ws/simulation_img/brt_simulation.png")
        
        sits_simul_door1 = [self.empty_sit_simul] * len(self.sits_door1)
        sits_simul_door2 = [self.empty_sit_simul] * len(self.sits_door2)
        sits_simul_door3 = [self.empty_sit_simul] * len(self.sits_door3)
        sits_simul_bus1_under = [self.empty_sit_simul] * len(self.sits_bus1_under)
        sits_simul_bus1_out = [self.empty_sit_simul] * len(self.sits_bus1_out)
        sits_simul_bus2_in = [self.empty_sit_simul] * len(self.sits_bus2_in)
        sits_simul_bus2_out = [self.empty_sit_simul] * len(self.sits_bus2_out)

        # 좌석 점유 상태를 결합하여 판단
        all_seats_occupied = []

        for s, sit in enumerate(self.sits_door1):
            if self.is_seat_occupied_combined(sit, [self.people_door1, self.people_bus1_under]):
                sits_simul_door1[s] = self.full_sit_simul
                all_seats_occupied.append(sit)

        for s, sit in enumerate(self.sits_door2):
            if self.is_seat_occupied_combined(sit, [self.people_door2, self.people_bus1_out]):
                sits_simul_door2[s] = self.full_sit_simul
                all_seats_occupied.append(sit)

        for s, sit in enumerate(self.sits_door3):
            if self.is_seat_occupied_combined(sit, [self.people_door3, self.people_bus2_in]):
                sits_simul_door3[s] = self.full_sit_simul
                all_seats_occupied.append(sit)

        for s, sit in enumerate(self.sits_bus1_under):
            if self.is_seat_occupied_combined(sit, [self.people_bus1_under, self.people_bus1_out]):
                sits_simul_bus1_under[s] = self.full_sit_simul
                all_seats_occupied.append(sit)

        for s, sit in enumerate(self.sits_bus1_out):
            if self.is_seat_occupied_combined(sit, [self.people_bus1_out, self.people_bus2_in]):
                sits_simul_bus1_out[s] = self.full_sit_simul
                all_seats_occupied.append(sit)

        for s, sit in enumerate(self.sits_bus2_in):
            if self.is_seat_occupied_combined(sit, [self.people_bus2_in, self.people_bus2_out]):
                sits_simul_bus2_in[s] = self.full_sit_simul
                all_seats_occupied.append(sit)

        for s, sit in enumerate(self.sits_bus2_out):
            if self.is_seat_occupied_combined(sit, [self.people_bus2_out, self.people_door3]):
                sits_simul_bus2_out[s] = self.full_sit_simul
                all_seats_occupied.append(sit)

        full_sits_door1 = sum(1 for sit in sits_simul_door1 if np.array_equal(sit, self.full_sit_simul))
        full_sits_door2 = sum(1 for sit in sits_simul_door2 if np.array_equal(sit, self.full_sit_simul))
        full_sits_door3 = sum(1 for sit in sits_simul_door3 if np.array_equal(sit, self.full_sit_simul))
        full_sits_bus1_under = sum(1 for sit in sits_simul_bus1_under if np.array_equal(sit, self.full_sit_simul))
        full_sits_bus1_out = sum(1 for sit in sits_simul_bus1_out if np.array_equal(sit, self.full_sit_simul))
        full_sits_bus2_in = sum(1 for sit in sits_simul_bus2_in if np.array_equal(sit, self.full_sit_simul))
        full_sits_bus2_out = sum(1 for sit in sits_simul_bus2_out if np.array_equal(sit, self.full_sit_simul))

        full_sits = (full_sits_door1 + full_sits_door2 + full_sits_door3 +
                     full_sits_bus1_under + full_sits_bus1_out +
                     full_sits_bus2_in + full_sits_bus2_out)

        if self.all_sits > 0:
            sit_percentage = (full_sits / self.all_sits) * 100
        else:
            sit_percentage = 0

        for s in range(len(self.sits_simul_door1)):
            brt_simul[self.sits_simul_door1[s][1]:self.sits_simul_door1[s][1]+self.new_sit_h, self.sits_simul_door1[s][0]:self.sits_simul_door1[s][0]+self.new_sit_w] = sits_simul_door1[s]

        for s in range(len(self.sits_simul_door2)):
            brt_simul[self.sits_simul_door2[s][1]:self.sits_simul_door2[s][1]+self.new_sit_h, self.sits_simul_door2[s][0]:self.sits_simul_door2[s][0]+self.new_sit_w] = sits_simul_door2[s]

        for s in range(len(self.sits_simul_door3)):
            brt_simul[self.sits_simul_door3[s][1]:self.sits_simul_door3[s][1]+self.new_sit_h, self.sits_simul_door3[s][0]:self.sits_simul_door3[s][0]+self.new_sit_w] = sits_simul_door3[s]

        for s in range(len(self.sits_simul_bus1_under)):
            brt_simul[self.sits_simul_bus1_under[s][1]:self.sits_simul_bus1_under[s][1]+self.new_sit_h, self.sits_simul_bus1_under[s][0]:self.sits_simul_bus1_under[s][0]+self.new_sit_w] = sits_simul_bus1_under[s]

        for s in range(len(self.sits_simul_bus1_out)):
            brt_simul[self.sits_simul_bus1_out[s][1]:self.sits_simul_bus1_out[s][1]+self.new_sit_h, self.sits_simul_bus1_out[s][0]:self.sits_simul_bus1_out[s][0]+self.new_sit_w] = sits_simul_bus1_out[s]

        for s in range(len(self.sits_simul_bus2_in)):
            brt_simul[self.sits_simul_bus2_in[s][1]:self.sits_simul_bus2_in[s][1]+self.new_sit_h, self.sits_simul_bus2_in[s][0]:self.sits_simul_bus2_in[s][0]+self.new_sit_w] = sits_simul_bus2_in[s]

        for s in range(len(self.sits_simul_bus2_out)):
            brt_simul[self.sits_simul_bus2_out[s][1]:self.sits_simul_bus2_out[s][1]+self.new_sit_h, self.sits_simul_bus2_out[s][0]:self.sits_simul_bus2_out[s][0]+self.new_sit_w] = sits_simul_bus2_out[s]

        brt_simul = cv2.putText(brt_simul,str(self.all_sits - full_sits),(613, 165),cv2.FONT_HERSHEY_SIMPLEX ,0.6,(0,0,0),2)
        brt_simul = cv2.putText(brt_simul,str(full_sits),(646, 193),cv2.FONT_HERSHEY_SIMPLEX ,0.6,(0,0,255),2)

        sit_percentage_text = f'Sit: {sit_percentage:.2f}%'
        brt_simul = cv2.putText(brt_simul, sit_percentage_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        brt_simul = self.bridge.cv2_to_imgmsg(brt_simul, encoding='bgr8')

        self.publisher_sit_simulation.publish(brt_simul)

    def is_seat_occupied_combined(self, seat, people_lists):
        for people in people_lists:
            for p in people:
                x, y = p[0], p[1]
                if seat[0][0] <= x <= seat[1][0] and seat[0][1] <= y <= seat[1][1]:
                    return True
        return False

    def remove_duplicates(self, people, distance_threshold):
        unique_people = []
        for person in people:
            if not any(np.linalg.norm(np.array(person) - np.array(unique_person)) < distance_threshold for unique_person in unique_people):
                unique_people.append(person)
        return unique_people

def main(args=None):
    rclpy.init(args=args)

    node = SitRecognition()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
