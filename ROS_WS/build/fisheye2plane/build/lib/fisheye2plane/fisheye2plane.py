import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import cv2
from scipy.ndimage import map_coordinates
import time
import os
import natsort
import glob
import collections 
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.clock import Clock  

def get_rotation_matrix(rad, ax):
    ax = np.array(ax)
    assert len(ax.shape) == 1 and ax.shape[0] == 3
    ax = ax / np.sqrt((ax**2).sum())
    R = np.diag([np.cos(rad)] * 3)
    R = R + np.outer(ax, ax) * (1.0 - np.cos(rad))

    ax = ax * np.sin(rad)
    R = R + np.array([[0, -ax[2], ax[1]],
                    [ax[2], 0, -ax[0]],
                    [-ax[1], ax[0], 0]])

    return R



# o_fov: output fov(보고 싶은 화각으로 설정)
# o_u: move right(왼쪽으로 돌릴 각도 크기), o_v: move down(아래로 돌릴 각도 크기) -> 범위 -90 ~ 90
        
def grid_in_3d_to_project(o_fov, o_sz, o_u, o_v):

    z = 1
    L = np.tan(o_fov / 2) / z
    x = np.linspace(L, -L, num=o_sz, dtype=np.float64)
    y = np.linspace(-L, L, num=o_sz, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = np.ones_like(x_grid)
 
    Rx = get_rotation_matrix(o_v, [1, 0, 0])
    Ry = get_rotation_matrix(o_u, [0, 1, 0])
    
    tic2 = time.time()
    #0.03     
    xyz_grid = np.stack([x_grid, y_grid, z_grid], -1).dot(Rx).dot(Ry)
    print("remap : ", time.time()-tic2)

    return [xyz_grid[..., i] for i in range(3)]


    # o_fov: output fov, o_u: move right, o_v: move down 
    # 1. o_fov 120
    # 2. o_fov 90, o_u 45
    # 3. o_fov 90, o_v 45

def fisheye_to_plane(frame, ih, iw, i_fov, o_fov, o_sz, o_u, o_v):

    # Convert degree to radian
    i_fov = i_fov * np.pi / 180
    o_fov = o_fov * np.pi / 180
    o_u = o_u * np.pi / 180
    o_v = o_v * np.pi / 180
    
    #0.03
    # 각도조절?? 
    x_grid, y_grid, z_grid = grid_in_3d_to_project(o_fov, o_sz, o_u, o_v)
    
    theta = np.arctan2(y_grid, x_grid)
    c_grid = np.sqrt(x_grid**2 + y_grid**2)
    rho = np.arctan2(c_grid, z_grid)
    r = rho * min(ih, iw) / i_fov
    coor_x = r * np.cos(theta) + iw / 2
    coor_y = r * np.sin(theta) + ih / 2

    # 0.04
    # out = np.stack([
    #     map_coordinates(frame[..., ich], [coor_y, coor_x], order=1)
    #     for ich in range(frame.shape[-1])
    # ], axis=-1)
    #0.003
    transformed_image = cv2.remap(frame, coor_x.astype(np.float32), coor_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)



    out = np.fliplr(transformed_image)

    return out


def fisheye_to_plane_info(ih, iw, i_fov, o_fov, o_sz, o_u, o_v):

    # Convert degree to radian
    i_fov = i_fov * np.pi / 180
    o_fov = o_fov * np.pi / 180
    o_u = o_u * np.pi / 180
    o_v = o_v * np.pi / 180
    
    #0.03
    # 각도조절?? 
    x_grid, y_grid, z_grid = grid_in_3d_to_project(o_fov, o_sz, o_u, o_v)
    
    theta = np.arctan2(y_grid, x_grid)
    c_grid = np.sqrt(x_grid**2 + y_grid**2)
    rho = np.arctan2(c_grid, z_grid)
    r = rho * min(ih, iw) / i_fov
    coor_x = r * np.cos(theta) + iw / 2
    coor_y = r * np.sin(theta) + ih / 2

    return coor_x.astype(np.float32), coor_y.astype(np.float32)


def fisheye_remap(frame, image_info):
    
    transformed_image = cv2.remap(frame, image_info.coor_x.astype(np.float32), image_info.coor_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)

    out = np.fliplr(transformed_image)

    return out


class ImageInfo():
    coor_x = None
    coor_y = None

class Fisheye2Plane(Node):
    qos_profile = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_ALL,
                depth=50
            )
    
    def __init__(self):
        super().__init__('fisheye2plane')

        self.call_count = 0
        self.call_count2 = 0
        
        self.SCEN_NAME = 'RELIABLE'
        self.door1_name = f'dataset/{self.SCEN_NAME}/door1'
        self.under_name = f'dataset/{self.SCEN_NAME}/under'

        os.makedirs(self.door1_name , exist_ok = True)
        os.makedirs(self.under_name , exist_ok = True)

        self.saved = False


        # door1 
        self.subscription0 = self.create_subscription(
            Image,
            '/camera0/image_raw', 
            self.image_callback0,
            Fisheye2Plane.qos_profile)
        
        self.subscription0 # cam0 sub


        # door2
        self.subscription2 = self.create_subscription(
            Image,
            '/camera4/image_raw', 
            self.image_callback2,
            1)
        self.subscription2 # cam2 sub
        
        # bus_door_3
        self.subscription4 = self.create_subscription(
            Image,
            '/camera6/image_raw', 
            self.image_callback4,
            1)
        self.subscription4 # cam4 sub
        #bus_1_in
        self.subscription6 = self.create_subscription(
            Image,
            '/camera2/image_raw', 
            self.image_callback6,
            Fisheye2Plane.qos_profile)  
        self.subscription6 # cam6 sub

        # bus_2_in
        self.subscription8 = self.create_subscription(
            Image,
            '/camera8/image_raw', 
            self.image_callback8,
            1)
        self.subscription8 # cam8 sub
        

        self.bridge = CvBridge()

        # publish
        self.publisher_in0 = self.create_publisher(Image,'door1_in', 1) # 문1(1량 승차문)의 버스 내부
        self.publisher_in2 = self.create_publisher(Image,'door2_in', 1) # 문2(1량 하차문)의 버스 내부
        self.publisher_in4 = self.create_publisher(Image,'door3_in', 1) # 문3(2량 하차문)의 버스 내부
        
        self.publisher_in6 = self.create_publisher(Image,'bus1_under', 1) # 1량 가운데의 버스 내부
        self.publisher_out6 = self.create_publisher(Image,'bus1_out', 1) # 1량 가운데의 버스 내부
        
        # tic1 = time.time()
        # print(time.time()-tic1)
        self.publisher_under0 = self.create_publisher(Image,'door1_under', 1) # 문1(1량 승차문)의 버스 문 아래
        self.publisher_under2 = self.create_publisher(Image,'door2_under', 1) # 문2(1량 하차문)의 버스 문 아래
        self.publisher_under4 = self.create_publisher(Image,'door3_under', 1) # 문3(2량 하차문)의 버스 문 아래
        
        self.publisher_out0 = self.create_publisher(Image,'door1_out', 1) # 문1(1량 승차문)의 버스 외부
        self.publisher_out2 = self.create_publisher(Image,'door2_out', 1) # 문2(1량 하차문)의 버스 외부
        self.publisher_out4 = self.create_publisher(Image,'door3_out', 1) # 문3(2량 히차문)의 버스 외부
        
        
        self.publisher_in8 = self.create_publisher(Image,'bus2_in', 1) # 2량 가운데의 버스 내부     
        self.publisher_out8 = self.create_publisher(Image,'bus2_out',1) # 2량 가운데의 버스 내부
        
        
        self.image_info={"image0_in" : ImageInfo(),
                        "image0_under" : ImageInfo(),
                        "image0_out" : ImageInfo(),
                        "image2_in" : ImageInfo(),
                        "image2_under" : ImageInfo(),
                        "image2_out" : ImageInfo(),
                        "image4_in" : ImageInfo(),
                        "image4_under" : ImageInfo(),
                        "image4_out" : ImageInfo(),                        
                        "image6_in" : ImageInfo(),
                        "image6_under" : ImageInfo(),
                        "image6_out" : ImageInfo(),                          
                        "image8_in" : ImageInfo(),
                        "image8_under" : ImageInfo(),
                        "image8_out" : ImageInfo()}
        
      
        # image 변환 실시간 수정용
        
                ############# 실험실/현장실험에서 위치 조절 필요 ################
        # [i_fov, o_fov, o_sz, o_u, o_v]
        # i_fov: input fov (180도 카메라이므로 180으로 고정), o_sz: 해상도 설정              90이상으로 올리면 더 넓지만 펴지는건 덜하다
        # o_fov: output fov(보고 싶은 화각으로 설정)
        # o_u: move right(왼쪽으로 돌릴 각도 크기), o_v: move down(아래로 돌릴 각도 크기) -> 범위 -90 ~ 90



        # image0
        
        _1, _2 = fisheye_to_plane_info(1920, 1920, 180, 90, 600, -40 , 0)
        self.image_info["image0_in"].coor_x = _1
        self.image_info["image0_in"].coor_y = _2
        _1, _2 = fisheye_to_plane_info(1920, 1920, 180, 90, 600, 0, 0)
        self.image_info["image0_under"].coor_x = _1
        self.image_info["image0_under"].coor_y = _2     
        #_1, _2 = fisheye_to_plane_info(1920, 1920, 180, 90, 600, 40, 0) #원본 scen1 화각
        # _1 , _2 = fisheye_to_plane_info(1920 , 1920 , 180 , 90 , 600 , 65 , 20) # test
        _1, _2 = fisheye_to_plane_info(1920, 1920, 180, 90, 600, -30, 7)  #가현님 시나리오 화각 사용
        self.image_info["image0_out"].coor_x = _1
        self.image_info["image0_out"].coor_y = _2        
        

      
        # image2
        
        _1, _2 = fisheye_to_plane_info(1920, 1920, 180, 90, 600, -40, 0)
        self.image_info["image2_in"].coor_x = _1
        self.image_info["image2_in"].coor_y = _2

        _1, _2 = fisheye_to_plane_info(1920, 1920, 180, 90, 600, 0, 0)
        self.image_info["image2_under"].coor_x = _1
        self.image_info["image2_under"].coor_y = _2
        
        _1, _2 = fisheye_to_plane_info(1920, 1920, 180, 90, 600, 40, 0)
        self.image_info["image2_out"].coor_x = _1
        self.image_info["image2_out"].coor_y = _2  
             
        # image4
        
        _1, _2 = fisheye_to_plane_info(1920, 1920, 180, 90, 600, -40, 0)
        self.image_info["image4_in"].coor_x = _1
        self.image_info["image4_in"].coor_y = _2

        _1, _2 = fisheye_to_plane_info(1920, 1920, 180, 90, 600, 0, 0)
        self.image_info["image4_under"].coor_x = _1
        self.image_info["image4_under"].coor_y = _2
        
        _1, _2 = fisheye_to_plane_info(1920, 1920, 180, 90, 600, 40, 0)
        self.image_info["image4_out"].coor_x = _1
        self.image_info["image4_out"].coor_y = _2

        # image6
        
        _1, _2 = fisheye_to_plane_info(1920, 1920, 180, 90, 600, -40, 0)
        self.image_info["image6_in"].coor_x = _1
        self.image_info["image6_in"].coor_y = _2

        _1, _2 = fisheye_to_plane_info(1920, 1920, 180, 90, 600, 0, 0)
        self.image_info["image6_under"].coor_x = _1
        self.image_info["image6_under"].coor_y = _2
        
        _1, _2 = fisheye_to_plane_info(1920, 1920, -180, 90, 600, -25, 0)
        self.image_info["image6_out"].coor_x = _1
        self.image_info["image6_out"].coor_y = _2
        
        #image8
        
        _1, _2 = fisheye_to_plane_info(1920, 1920, 180, 90, 600, 40, 0)
        self.image_info["image8_in"].coor_x = _1
        self.image_info["image8_in"].coor_y = _2

        _1, _2 = fisheye_to_plane_info(1920, 1920, 180, -90, 600, 0, 0)
        self.image_info["image8_under"].coor_x = _1
        self.image_info["image8_under"].coor_y = _2
        
        _1, _2 = fisheye_to_plane_info(1920, 1920, 180, 90, 600, -28, 0)
        self.image_info["image8_out"].coor_x = _1
        self.image_info["image8_out"].coor_y = _2
        
                
        self.queue = collections.deque()    
        self.queue2= collections.deque()    
        
    def message_callback(self , msg):
        self.queue.append(msg)
        self.process_queue()

    def process_queue(self):
        while self.queue:
            msg = self.queue.popleft()
            self.image_callback0(msg)


    def message_callback2(self , msg):
        self.queue2.append(msg)
        self.process_queue2()

    def process_queue2(self):
        while self.queue2:
            msg = self.queue2.popleft()
            self.image_callback6(msg)
    # cam0
    def image_callback0(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        w = int(frame.shape[1])
        h = int(frame.shape[0])

        black = np.zeros((int((w-h)/2), w, 3), np.uint8)

        frame_new = cv2.vconcat( [black, frame] )
        frame_new = cv2.vconcat( [frame_new, black] )

        w = int(frame_new.shape[1])
        h = int(frame_new.shape[0])

        ############# 실험실/현장실험에서 위치 조절 필요 ################
        # [i_fov, o_fov, o_sz, o_u, o_v]
        # i_fov: input fov (180도 카메라이므로 180으로 고정), o_sz: 해상도 설정              90이상으로 올리면 더 넓지만 펴지는건 덜하다
        # o_fov: output fov(보고 싶은 화각으로 설정)
        # o_u: move right(왼쪽으로 돌릴 각도 크기), o_v: move down(아래로 돌릴 각도 크기) -> 범위 -90 ~ 90
        cam0_in = np.array(fisheye_remap(frame_new, self.image_info["image0_in"]))
        cam0_under = np.array(fisheye_remap(frame_new, self.image_info["image0_under"]))
        cam0_out = np.array(fisheye_remap(frame_new, self.image_info["image0_out"]))

        cam0_out = cv2.rotate(cam0_out, cv2.ROTATE_90_CLOCKWISE)
        cam0_in = cv2.rotate(cam0_in, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.saved:
            cv2.imwrite(os.path.join(f'{self.door1_name}/{self.call_count}.jpg') ,  cam0_in)
            print(f"SAVE {self.call_count}jpg")
        print("door1 : : ",self.call_count)

        cam0_in_msg = self.bridge.cv2_to_imgmsg(cam0_in, encoding='bgr8')
        cam0_under_msg = self.bridge.cv2_to_imgmsg(cam0_under, encoding='bgr8')
        cam0_out_msg = self.bridge.cv2_to_imgmsg(cam0_out, encoding='bgr8')

        now = Clock().now().to_msg()
        cam0_in_msg.header.stamp = now
        cam0_under_msg.header.stamp = now
        cam0_out_msg.header.stamp = now

        self.publisher_in0.publish(cam0_in_msg)
        self.publisher_under0.publish(cam0_under_msg)
        self.publisher_out0.publish(cam0_out_msg)
        self.call_count += 1

    # cam2
    def image_callback2(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        w = int(frame.shape[1])
        h = int(frame.shape[0])

        black = np.zeros((int((w-h)/2), w, 3), np.uint8)

        frame_new = cv2.vconcat( [black, frame] )
        frame_new = cv2.vconcat( [frame_new, black] )

        w = int(frame_new.shape[1])
        h = int(frame_new.shape[0])

        ############# 실험실/현장실험에서 위치 조절 필요 ################
        # [i_fov, o_fov, o_sz, o_u, o_v]
        # i_fov: input fov (180도 카메라이므로 180으로 고정), o_sz: 해상도 설정
        # o_fov: output fov(보고 싶은 화각으로 설정)
        # o_u: move right(왼쪽으로 돌릴 각도 크기), o_v: move down(아래로 돌릴 각도 크기) -> 범위 -90 ~ 90
        cam2_in = np.array(fisheye_remap(frame_new, self.image_info["image2_in"]))
        cam2_under = np.array(fisheye_remap(frame_new, self.image_info["image2_under"]))
        cam2_out = np.array(fisheye_remap(frame_new, self.image_info["image2_out"]))

        cam2_out = cv2.rotate(cam2_out, cv2.ROTATE_90_CLOCKWISE)
        cam2_in = cv2.rotate(cam2_in, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cam2_in_msg = self.bridge.cv2_to_imgmsg(cam2_in, encoding='bgr8')
        cam2_under_msg = self.bridge.cv2_to_imgmsg(cam2_under, encoding='bgr8')
        cam2_out_msg = self.bridge.cv2_to_imgmsg(cam2_out, encoding='bgr8')

        self.publisher_in2.publish(cam2_in_msg)
        self.publisher_under2.publish(cam2_under_msg)
        self.publisher_out2.publish(cam2_out_msg)
    
    # cam4
    def image_callback4(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        w = int(frame.shape[1])
        h = int(frame.shape[0])

        black = np.zeros((int((w-h)/2), w, 3), np.uint8)

        frame_new = cv2.vconcat( [black, frame] )
        frame_new = cv2.vconcat( [frame_new, black] )

        w = int(frame_new.shape[1])
        h = int(frame_new.shape[0])       
        
        cam4_in = np.array(fisheye_remap(frame_new, self.image_info["image4_in"]))
        cam4_under = np.array(fisheye_remap(frame_new, self.image_info["image4_under"]))
        cam4_out = np.array(fisheye_remap(frame_new, self.image_info["image4_out"]))
        
        
        # cam4_in = np.array(fisheye_to_plane(frame_new, h, w, 180, 90, 600, 40, 0))
        # cam4_under = np.array(fisheye_to_plane(frame_new, h, w, 180, 90, 600, 0, 0))
        # cam4_out = np.array(fisheye_to_plane(frame_new, h, w, 180, 90, 600, -40, 0))        
        

        cam4_out = cv2.rotate(cam4_out, cv2.ROTATE_90_CLOCKWISE)
        cam4_in = cv2.rotate(cam4_in, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cam4_in_msg = self.bridge.cv2_to_imgmsg(cam4_in, encoding='bgr8')
        cam4_under_msg = self.bridge.cv2_to_imgmsg(cam4_under, encoding='bgr8')
        cam4_out_msg = self.bridge.cv2_to_imgmsg(cam4_out, encoding='bgr8')

        self.publisher_in4.publish(cam4_in_msg)
        self.publisher_under4.publish(cam4_under_msg)
        self.publisher_out4.publish(cam4_out_msg)

    # cam6
    def image_callback6(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        w = int(frame.shape[1])
        h = int(frame.shape[0])

        black = np.zeros((int((w-h)/2), w, 3), np.uint8)

        frame_new = cv2.vconcat( [black, frame] )
        frame_new = cv2.vconcat( [frame_new, black] )

        w = int(frame_new.shape[1])
        h = int(frame_new.shape[0])

        ############# 실험실/현장실험에서 위치 조절 필요 ################
        # [i_fov, o_fov, o_sz, o_u, o_v]
        # i_fov: input fov (180도 카메라이므로 180으로 고정), o_sz: 해상도 설정
        # o_fov: output fov(보고 싶은 화각으로 설정)
        # o_u: move right(왼쪽으로 돌릴 각도 크기), o_v: move down(아래로 돌릴 각도 크기) -> 범위 -90 ~ 90
        # cam6_in = np.array(fisheye_to_plane(frame_new, h, w, 180, 90, 600, 0, 0))

        cam6_in = np.array(fisheye_remap(frame_new, self.image_info["image6_in"]))
        cam6_out = np.array(fisheye_remap(frame_new, self.image_info["image6_out"]))


        cam6_in = cv2.rotate(cam6_in, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cam6_out = cv2.rotate(cam6_out, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        if self.saved:
            cv2.imwrite(os.path.join(f'{self.under_name}/{self.call_count2}.jpg') , cam6_in)
        print("under : ",self.call_count2)
        cam6_in_msg = self.bridge.cv2_to_imgmsg(cam6_in, encoding='bgr8')
        cam6_out_msg = self.bridge.cv2_to_imgmsg(cam6_out, encoding='bgr8')
        
        
        now = Clock().now().to_msg()
        cam6_in_msg.header.stamp = now
        cam6_out_msg.header.stamp = now

        self.publisher_in6.publish(cam6_in_msg)
        self.publisher_out6.publish(cam6_out_msg)
        self.call_count2 += 1
        
    # cam8
    def image_callback8(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        w = int(frame.shape[1])
        h = int(frame.shape[0])

        black = np.zeros((int((w-h)/2), w, 3), np.uint8)

        frame_new = cv2.vconcat( [black, frame] )
        frame_new = cv2.vconcat( [frame_new, black] )

        w = int(frame_new.shape[1])
        h = int(frame_new.shape[0])

        ############# 실험실/현장실험에서 위치 조절 필요 ################
        # [i_fov, o_fov, o_sz, o_u, o_v]
        # i_fov: input fov (180도 카메라이므로 180으로 고정), o_sz: 해상도 설정
        # o_fov: output fov(보고 싶은 화각으로 설정)
        # o_u: move right(왼쪽으로 돌릴 각도 크기), o_v: move down(아래로 돌릴 각도 크기) -> 범위 -90 ~ 90
        # cam8_in = np.array(fisheye_to_plane(frame_new, h, w, 180, 90, 600, -20, 0))
        
        # cam8_out = np.array(fisheye_to_plane(frame_new, h, w, 180, 90, 600, 20, 0))

        cam8_in = np.array(fisheye_remap(frame_new, self.image_info["image8_in"]))
        cam8_out = np.array(fisheye_remap(frame_new, self.image_info["image8_out"]))


        cam8_in = cv2.rotate(cam8_in, cv2.ROTATE_180)
        cam8_in = cv2.rotate(cam8_in, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cam8_out = cv2.rotate(cam8_out, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        
        cam8_in_msg = self.bridge.cv2_to_imgmsg(cam8_in, encoding='bgr8')
        cam8_out_msg = self.bridge.cv2_to_imgmsg(cam8_out, encoding='bgr8')
        
        self.publisher_in8.publish(cam8_in_msg)
        self.publisher_out8.publish(cam8_out_msg)

def main(args=None):

    rclpy.init(args=args)

    node = Fisheye2Plane()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin() 
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    
    main()