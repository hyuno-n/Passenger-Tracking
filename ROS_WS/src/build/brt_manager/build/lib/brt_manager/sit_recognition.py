import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import cv2

def return_bbox(results, frame):
    people = []

    for r in results:
        boxes = r.boxes

        for box in boxes:
            b = box.xyxy[0]
            x1, y1, x2, y2 = b.tolist()

            # 박스 중심점 계산
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)
            people.append([x, y])

            # 감지된 사람 주위에 박스 그리기
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 녹색 박스로 표시

            # 중심점 표시 (선택적)
            frame = cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # 빨간색 점

    return people, frame



class SitRecognition(Node):
    def __init__(self):
        super().__init__('sit_recognition')

        self.subscription_door1 = self.create_subscription(
            Image,
            'door1_in',  
            self.door1_callback,
            1)
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
            1)
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

        self.bridge = CvBridge()

        self.model = YOLO('yolov8x.pt')


        ############### 실험실/현장실험에서 위치 조절 필요 ##################
        # 좌석 좌표 위치 순서 정하고 이를 암기해서 좌표값 넣을 때 꼭 순서대로 배열 
        # 문1(1량 승차문)에서의 좌석 위치
        self.sits_door1 = [[(409, 247), (501, 338)], [(376, 141), (471, 227)], [(245, 249), (340, 337)], 
                           [(239, 173), (343, 245)], [(94, 248), (187, 359)], [(101, 157), (199, 243)],
                            ]
                        #    [(421, 180), (586, 336)]]
            
            
        self.sits_bus1_under = [[(421, 180), (586, 336)],[(170, 375), (264, 557)],[(14, 382), (156, 484)],
                                
                                # [(191, 397), (276, 606)],[(64, 400), (156, 575)],
                                [(453, 412), (597, 598)]]
        # ,[(456, 448), (598, 570)]]
        # 큰 버스에서 5,6번 <-> 7,8번 교환 됐다.
        self.sits_bus1_out = [[(303, 309), (411, 489)],[(419, 298), (508, 490)],[(2, 263), (166, 430)]]
        
        self.sits_door2 = [[(287, 49), (416, 176)], [(132, 107), (266, 242)]] # 문2(1량 하차문)에서의 좌석 위치
        
        self.sits_door3 = [[(130, 328), (223, 492)], [(15, 316), (104, 485)], [(250, 182), (357, 270)], 
                           [(244, 97), (366, 179)],[(576, 341), (595, 386)], [(488, 334), (593, 478)], 
                           [(357, 342), (482, 525)], [(84, 186), (213, 279)], [(89, 96), (210, 182)]] ## 잘되는 좌표
            
        self.sits_bus2_in =[[(131, 392), (210, 572)], [(2, 389), (105, 552)],
                            [(459, 399), (586, 534)], [(348, 403), (449, 559)]]
                            
                           
            
        self.sits_bus2_out = [[(389, 217), (453, 382)],[(485, 217), (560, 393)],
                            [(61, 221), (153, 387)],[(157, 216), (250, 387)],

                            #2량 맨 뒤 6자리
                            [(396, 117), (493, 169)], [(383, 65), (503, 101)], [(376, 21), (494, 60)], 
                            [(132, 108), (225, 164)], [(154, 66), (231, 104)], [(169, 26), (244, 64)]] # 문3(2량 하차문)에서의 좌석 위치
        
        
        
        # 일단 맨 뒤에 세개 붙어있는 좌석은 생략함 #
        
        
        ### 시뮬레이션 화면의 착석/빈좌석 구성 ###
        self.full_sit = cv2.imread("/home/krri/Desktop/ros_ws/simulation_img/full_sit.png")
        self.sit_h, self.sit_w = self.full_sit.shape[:2]
        self.full_sit_simul = cv2.resize(self.full_sit, (int(self.sit_w*0.2), int(self.sit_h*0.2)))
        self.new_sit_h, self.new_sit_w = self.full_sit_simul.shape[:2]

        # 빈좌석: 착석좌석을 grayscale로 변환
        self.empty_sit_simul = cv2.cvtColor(self.full_sit_simul, cv2.COLOR_BGR2GRAY) # bgr(3) -> gray(1) channel, grayscale
        self.empty_sit_simul = cv2.cvtColor(self.empty_sit_simul, cv2.COLOR_GRAY2BGR) # gray(1) -> bgr(3) channel, grayscale

        ### 버스 그림에 좌석 삽입 -> 수정 필요 없음 ###
        self.sits_simul_door1 = [(115, 140), (115, 175),
                        (115+self.new_sit_w+15, 140), (115+self.new_sit_w+15, 175),
                        (115+self.new_sit_w*2+15*2, 140), (115+self.new_sit_w*2+15*2, 175)]
                        # (115+self.new_sit_w*2+15*2, 35)] # 버스 1량 내부 앞쪽 좌석 배치

        self.sits_simul_bus1_under = [(115+self.new_sit_w*2+15*2, 35),(115+self.new_sit_w*3+15*3, 140), (115+self.new_sit_w*3+15*3, 175),
                        # (115+self.new_sit_w*4+15*4, 140), (115+self.new_sit_w*4+15*4, 175),
                        # (115+self.new_sit_w*5+15*5, 175), 
                        # (115+self.new_sit_w*6+15*6, 175),
                        (115+self.new_sit_w*3+15*3, 35)] 
                        # (115+self.new_sit_w*4+15*4, 35)] # 버스 1량 내부 뒤쪽 좌석 배치
        
        self.sits_simul_bus1_out = [(115+self.new_sit_w*4+15*4, 140), (115+self.new_sit_w*4+15*4, 175),(115+self.new_sit_w*4+15*4, 35)]
        
        self.sits_simul_door2 = [(115+self.new_sit_w*5+15*5, 175), (115+self.new_sit_w*6+15*6, 175)]
        

        self.sits_simul_door3 = [(800, 140), (800, 175),
                        (800+self.new_sit_w+15, 140), (800+self.new_sit_w+15, 175),
                        (775, 35), (775+self.new_sit_h, 35), (775+self.new_sit_h*2, 35),
                        (800+self.new_sit_w*2+15*2, 140), (800+self.new_sit_w*2+15*2, 175)]
                        
        self.sits_simul_bus2_in = [(800+self.new_sit_w*3+15*3, 140),(800+self.new_sit_w*3+15*3, 175),
                                #  5,6번  (800+self.new_sit_w*4+15*4, 140), (800+self.new_sit_w*4+15*4, 175),
                                   (800+self.new_sit_w*3+15*3, 35), (800+self.new_sit_w*3+15*3, 70)]
                                #  9,10번  (800+self.new_sit_w*4+15*4, 35), (800+self.new_sit_w*4+15*4, 70)]                
                        
        self.sits_simul_bus2_out = [(800+self.new_sit_w*4+15*4, 140), (800+self.new_sit_w*4+15*4, 175),(800+self.new_sit_w*4+15*4, 35), (800+self.new_sit_w*4+15*4, 70),
                                    (800+self.new_sit_w*5+15*5, 160), (800+self.new_sit_w*5+15*5+self.new_sit_h, 160),(800+self.new_sit_w*5+15*5+self.new_sit_h*2, 160), (800+self.new_sit_w*5+15*5, 35), (800+self.new_sit_w*5+15*5+self.new_sit_h, 35), (800+self.new_sit_w*5+15*5+self.new_sit_h*2, 35)]                        
                         # 버스 2량 내부 좌석 배치

        self.all_sits = len(self.sits_simul_door1) + len(self.sits_simul_door2) + len(self.sits_simul_door3) + len(self.sits_simul_bus1_under) + len(self.sits_simul_bus2_in) + len(self.sits_bus2_out)# 1량 좌석 수 + 2량 좌석 수 => 전체 좌석 수
        self.empty_sits = self.all_sits
        self.prev_empty_sits = self.all_sits

        self.timer = self.create_timer(0.02, self.sit_simulation_pub)

        self.publisher_door1_detect = self.create_publisher(Image,'detect_door1_in', 1)
        self.publisher_door2_detect = self.create_publisher(Image,'detect_door2_in', 1)
        self.publisher_door3_detect = self.create_publisher(Image,'detect_door3_in', 1)
        
        self.publisher_bus1_under_detect = self.create_publisher(Image,'detect_bus1_under', 1)
        self.publisher_bus1_out_detect = self.create_publisher(Image,'detect_bus1_out',1)
        self.publisher_bus2_in_detect = self.create_publisher(Image,'detect_bus2_in', 1)
        self.publisher_bus2_out_detect = self.create_publisher(Image,'detect_bus2_out', 1)


        self.publisher_sit_simulation = self.create_publisher(Image, 'bus_illustration', 1)

        

    def door1_callback(self, msg):

        self.frame_door1 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run YOLOv8 inference on the frame
        results = self.model(self.frame_door1, classes=0)
        self.people_door1 = return_bbox(results, self.frame_door1)[0]
        self.frame_door1 = return_bbox(results, self.frame_door1)[1]

        detection_msg = self.bridge.cv2_to_imgmsg(self.frame_door1, encoding='bgr8')

        self.publisher_door1_detect.publish(detection_msg)

    def door2_callback(self, msg):

        self.frame_door2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run YOLOv8 inference on the frame
        results = self.model(self.frame_door2, classes=0)
        self.people_door2 = return_bbox(results, self.frame_door2)[0]
        self.frame_door2 = return_bbox(results, self.frame_door2)[1]

        detection_msg = self.bridge.cv2_to_imgmsg(self.frame_door2, encoding='bgr8')

        self.publisher_door2_detect.publish(detection_msg)
    
    def door3_callback(self, msg):

        self.frame_door3 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run YOLOv8 inference on the frame
        results = self.model(self.frame_door3 , classes=0)
        self.people_door3 = return_bbox(results, self.frame_door3)[0]
        self.frame_door3 = return_bbox(results, self.frame_door3)[1]

        detection_msg = self.bridge.cv2_to_imgmsg(self.frame_door3, encoding='bgr8')

        self.publisher_door3_detect.publish(detection_msg)

    def bus1_front_callback(self, msg):

        self.frame_bus1_under = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run YOLOv8 inference on the frame
        results = self.model(self.frame_bus1_under, classes=0)
        self.people_bus1_under = return_bbox(results, self.frame_bus1_under)[0]
        self.frame_bus1_under = return_bbox(results, self.frame_bus1_under)[1]

        detection_msg = self.bridge.cv2_to_imgmsg(self.frame_bus1_under, encoding='bgr8')

        self.publisher_bus1_under_detect.publish(detection_msg)
    
    def bus1_back_callback(self, msg):

        self.frame_bus1_out = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run YOLOv8 inference on the frame
        results = self.model(self.frame_bus1_out, classes=0)
        self.people_bus1_out = return_bbox(results, self.frame_bus1_out)[0]
        self.frame_bus1_out = return_bbox(results, self.frame_bus1_out)[1]

        detection_msg = self.bridge.cv2_to_imgmsg(self.frame_bus1_out, encoding='bgr8')

        self.publisher_bus1_out_detect.publish(detection_msg)    

    def bus2_callback_front(self, msg):

        self.frame_bus2_in = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run YOLOv8 inference on the frame
        results = self.model(self.frame_bus2_in, classes=0)
        self.people_bus2_in  = return_bbox(results, self.frame_bus2_in)[0]
        self.frame_bus2_in = return_bbox(results, self.frame_bus2_in)[1]
        
        detection_msg = self.bridge.cv2_to_imgmsg(self.frame_bus2_in, encoding='bgr8')

        self.publisher_bus2_in_detect.publish(detection_msg) 
        
    def bus2_callback_back(self, msg):

        self.frame_bus2_out = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run YOLOv8 inference on the frame
        results = self.model(self.frame_bus2_out, classes=0)
        self.people_bus2_out = return_bbox(results, self.frame_bus2_out)[0]
        self.frame_bus2_out = return_bbox(results, self.frame_bus2_out)[1]

        detection_msg = self.bridge.cv2_to_imgmsg(self.frame_bus2_out, encoding='bgr8')

        self.publisher_bus2_out_detect.publish(detection_msg)

    def sit_simulation_pub(self):
        
        

        # 시뮬레이션에 쓸 이미지 정의 (버스, 좌석)
        brt_simul = cv2.imread("/home/krri/Desktop/ros_ws/simulation_img/brt_simulation.png")
        
        sits_simul_door1 = []
        sits_simul_door2 = []
        sits_simul_door3 = []
        sits_simul_bus1_under = []
        sits_simul_bus1_out = []
        sits_simul_bus2_in = []
        sits_simul_bus2_out = []
        
        for s in range(len(self.sits_door1)):
            # frame = cv2.rectangle(frame, sits[s][0], sits[s][1], (255, 0, 0), 1, cv2.LINE_8)
            sits_simul_door1.append(self.empty_sit_simul)
        for s in range(len(self.sits_door2)):
            # frame = cv2.rectangle(frame, sits[s][0], sits[s][1], (255, 0, 0), 1, cv2.LINE_8)
            sits_simul_door2.append(self.empty_sit_simul)
        for s in range(len(self.sits_door3)):
            # frame = cv2.rectangle(frame, sits[s][0], sits[s][1], (255, 0, 0), 1, cv2.LINE_8)
            sits_simul_door3.append(self.empty_sit_simul)
        for s in range(len(self.sits_bus1_under)):
            # frame = cv2.rectangle(frame, sits[s][0], sits[s][1], (255, 0, 0), 1, cv2.LINE_8)
            sits_simul_bus1_under.append(self.empty_sit_simul)
        for s in range(len(self.sits_bus1_out)):
            # frame = cv2.rectangle(frame, sits[s][0], sits[s][1], (255, 0, 0), 1, cv2.LINE_8)
            sits_simul_bus1_out.append(self.empty_sit_simul)    
        for s in range(len(self.sits_bus2_in)):
            # frame = cv2.rectangle(frame, sits[s][0], sits[s][1], (255, 0, 0), 1, cv2.LINE_8)
            sits_simul_bus2_in.append(self.empty_sit_simul)
        for s in range(len(self.sits_bus2_out)):
            # frame = cv2.rectangle(frame, sits[s][0], sits[s][1], (255, 0, 0), 1, cv2.LINE_8)
            sits_simul_bus2_out.append(self.empty_sit_simul)


        
        for s in range(len(self.sits_simul_door1)):
            for p in self.people_door1:
                x, y = p[0], p[1]
                if (self.sits_door1[s][0][0] <= x <= self.sits_door1[s][1][0] and self.sits_door1[s][0][1] <= y <= self.sits_door1[s][1][1]):
                    # Change the color of the rectangle to yellow
                    sits_simul_door1[s] = self.full_sit_simul
                    break
                
        for s in range(len(self.sits_simul_door2)):
            for p in self.people_door2:
                x, y = p[0], p[1]
                # print(x,y)
                if (self.sits_door2[s][0][0] <= x <= self.sits_door2[s][1][0] and self.sits_door2[s][0][1] <= y <= self.sits_door2[s][1][1]):
                    # Change the color of the rectangle to yellow
                    sits_simul_door2[s] = self.full_sit_simul
                    break
        for s in range(len(self.sits_simul_bus1_under)):
            for p in self.people_bus1_under:
                x, y = p[0], p[1]
                print(x,y)
                if (self.sits_bus1_under[s][0][0] <= x <= self.sits_bus1_under[s][1][0] and self.sits_bus1_under[s][0][1] <= y <= self.sits_bus1_under[s][1][1]):
                    # Change the color of the rectangle to yellow
                    sits_simul_bus1_under[s] = self.full_sit_simul
                    break
        for s in range(len(self.sits_simul_bus1_out)):
            for p in self.people_bus1_out:
                x, y = p[0], p[1]
                print(x,y)
                if (self.sits_bus1_out[s][0][0] <= x <= self.sits_bus1_out[s][1][0] and self.sits_bus1_out[s][0][1] <= y <= self.sits_bus1_out[s][1][1]):
                    # Change the color of the rectangle to yellow
                    sits_simul_bus1_out[s] = self.full_sit_simul
                    break
                
        for s in range(len(self.sits_simul_door3)):
            for p in self.people_door3:
                x, y = p[0], p[1]
                print('??')
                print(s)
                print(np.shape(self.sits_door3))
                
                if (self.sits_door3[s][0][0] <= x <= self.sits_door3[s][1][0] and self.sits_door3[s][0][1] <= y <= self.sits_door3[s][1][1]):
                    # Change the color of the rectangle to yellow
                    sits_simul_door3[s] = self.full_sit_simul
                    break
        for s in range(len(self.sits_simul_bus2_in)):
            for p in self.people_bus2_in:
                x, y = p[0], p[1]
                print('??')
                print(s)
                print(np.shape(self.sits_simul_bus2_in))
                
                if (self.sits_bus2_in[s][0][0] <= x <= self.sits_bus2_in[s][1][0] and self.sits_bus2_in[s][0][1] <= y <= self.sits_bus2_in[s][1][1]):
                    # Change the color of the rectangle to yellow
                    sits_simul_bus2_in[s] = self.full_sit_simul
                    break                      
        for s in range(len(self.sits_simul_bus2_out)):
            for p in self.people_bus2_out:
                x, y = p[0], p[1]
                print('??')
                print(s)
                print(np.shape(self.sits_simul_bus2_out))
                
                if (self.sits_bus2_out[s][0][0] <= x <= self.sits_bus2_out[s][1][0] and self.sits_bus2_out[s][0][1] <= y <= self.sits_bus2_out[s][1][1]):
                    # Change the color of the rectangle to yellow
                    sits_simul_bus2_out[s] = self.full_sit_simul
                    break                
        
        
        full_sits = np.count_nonzero(sits_simul_door3 == self.full_sit_simul)
        full_sits += np.count_nonzero(sits_simul_door1 == self.full_sit_simul)
        full_sits += np.count_nonzero(sits_simul_door2 == self.full_sit_simul)
        full_sits += np.count_nonzero(sits_simul_bus1_under == self.full_sit_simul)
        full_sits += np.count_nonzero(sits_simul_bus1_out == self.full_sit_simul)
        full_sits += np.count_nonzero(sits_simul_bus2_in == self.full_sit_simul)
        full_sits += np.count_nonzero(sits_simul_bus2_out == self.full_sit_simul)
                     
            
                # 버스 그림에 좌석 그림 세팅 #
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
            
        # 빈 좌석 / 착석한 좌석 개수
        brt_simul = cv2.putText(brt_simul,str(self.all_sits - full_sits),(613, 165),cv2.FONT_HERSHEY_SIMPLEX ,0.6,(0,0,0),2)
        brt_simul = cv2.putText(brt_simul,str(full_sits),(646, 193),cv2.FONT_HERSHEY_SIMPLEX ,0.6,(0,0,255),2)
        
        brt_simul = self.bridge.cv2_to_imgmsg(brt_simul, encoding='bgr8')

        self.publisher_sit_simulation.publish(brt_simul)



def main(args=None):
    rclpy.init(args=args)

    node = SitRecognition()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    
    
    
