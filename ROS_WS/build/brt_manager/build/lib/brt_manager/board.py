import rclpy
from sensor_msgs.msg import Image
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
import numpy as np
import os
import sys


current_path = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.abspath(os.path.join(current_path, '../../../../../../'))
sys.path.append(target_path)

from src.pose_recognition.tools.hboe import hboe

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

from collections import deque
import itertools
os.environ["QT_LOGGING_RULES"] = "*.debug=false"

class Board(Node):
    def __init__(self):
        super().__init__('board_node')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.yolo_model = YOLO('yolov8x.pt').to(self.device)
        self.bridge = CvBridge()
        self.message_queue = deque(maxlen=1000)
        self.subscription0 = self.create_subscription(
            Image,
            '/door1_out', 
            self.queue_message_callback,
            1
        )
        
        self.publish_origin_image = self.create_publisher(Image , 'target_image',1000)
        self.publisher_ = self.create_publisher(Image, 'test_image', 1000)
        self.publisher__ = self.create_publisher(Image , 'plot',1000)
        self.call_count = 1
        self.save_folder = 'node_image'
        self.generate_save_folder()
        self.Plot__ = Plot__()


    def generate_save_folder(self):
        os.makedirs(self.save_folder , exist_ok=True)
        sublist = ['NON_IMAGE','IMAGE','BEV','PLOT']
        for sub in sublist:
            mf = os.path.join(self.save_folder , sub)
            os.makedirs(mf , exist_ok=True)

    def queue_message_callback(self , msg):
        self.message_queue.append(msg)
        self.process_oldest_message()
        
    def process_oldest_message(self):
        if self.message_queue:
            oldest_message = self.message_queue.popleft()
            self.call_back(oldest_message)
            self.call_count += 1

    def call_back(self, msg):
        
         
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') # msg to cv Image
        cv_image = cv_image[90: , 50 :]
        cv_image = cv2.resize(cv_image , dsize=(600,600))
        cv2.imwrite(os.path.join(self.save_folder,'NON_IMAGE',f"{self.call_count}.jpg"), cv_image)
        image , ax_image , preprocessing_image = self.run_yolo(cv_image)

        print("image shape  :",image.shape)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(self.save_folder, "BEV", f"{self.call_count}.jpg"), cv2.cvtColor(image_rgb,cv2.COLOR_BGR2RGB))
        ros_image_msg = self.bridge.cv2_to_imgmsg(image_rgb, encoding='rgb8')
        self.publisher_.publish(ros_image_msg)
        
        

       
        ax_image = cv2.cvtColor(ax_image , cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(self.save_folder ,'PLOT',f"{self.call_count}.jpg"),cv2.cvtColor(ax_image,cv2.COLOR_BGR2RGB))
        ros_image_msg = self.bridge.cv2_to_imgmsg(ax_image , encoding='rgb8')
        self.publisher__.publish(ros_image_msg)
        
        
        pr_image = cv2.cvtColor(preprocessing_image , cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(self.save_folder , 'IMAGE' , f"{self.call_count}.jpg"),cv2.cvtColor(pr_image,cv2.COLOR_BGR2RGB))
        ros_image_msg = self.bridge.cv2_to_imgmsg(pr_image , encoding='rgb8')
        self.publish_origin_image.publish(ros_image_msg)

        self.get_logger().info("Published image")
    
    
    
    def run_yolo(self,image):
        people_bottom = []
        people_center = []
        people_ori = []
        image_copy = image.copy()
        
        results = self.yolo_model(image , classes = 0 , verbose=False)
        boxes = results[0].boxes
        
        for box in boxes:
            b = box.xyxy[0]
            x1 , y1 , x2 ,y2 = b.tolist()
            cv2.rectangle(image_copy , (int(x1) ,int(y1)) , (int(x2),int(y2)),color=(0,0,255),thickness=2)
            
            
            """ Human Oritention get"""
            center_x = int((x1 + x2)//2)
            center_y = int((y1+y2)//2)
            crop_person = image_copy[int(y1):int(y2) , int(x1):int(x2)] # h w c 

            person_ori = int(hboe(crop_person))
            
            if int((x1+x2)/2) >= 60 and int(y2) >= 125:
                people_center.append([center_x, center_y])
                people_ori.append(person_ori)
                
                if y1 + (x2-x1)*2.3 > 600 or y2-y1 > (x2-x1)*2:
                    people_bottom.append([[int((x1+x2)/2), int(y2)]])
                else:
                    people_bottom.append([[int((x1+x2)/2), int(y1 + (x2-x1)*2.3)]])
                
                #bottom = [[int(x1 +x2) // 2 , int(y2)]]
                #people_bottom.append(bottom)                

                buffer = tuple(itertools.chain.from_iterable(people_bottom[-1]))
                cv2.circle(image_copy , buffer , radius=6 , color=(0,255,0) , thickness= -1)
        
        IPM_IMAGE , tb = self.IPM_Adative(image, people_bottom)
        
        #IPM_IMAGE = np.full(shape=(600,600),fill_value=255 ,dtype=np.uint8)
        #ax_image = np.zeros(shape=(600,600),dtype=np.uint8)
        ax_image = self.Plot__(tb , people_ori , people_center)
        return IPM_IMAGE , ax_image , image_copy
        
    def IPM_Adative(self , image:np.array , people_bottom : list) -> np.array:

        raw_img = image
        people_head = np.array(people_bottom)
        people_head[: , 0 , 0] += 630
        people_head[ : , 0 , 1] += 600
        
        people_head = people_head.tolist()
        topLeft = [625, 725]
        bottomRight = [1800, 1200]  
        topRight = [1125, 725]  
        bottomLeft = [0, 1200]  

        rows, cols = raw_img.shape[:2]

        black = np.zeros((rows*3, rows*3, 3), np.uint8)

        bg_height, bg_width = black.shape[:2]
        overlay_height, overlay_width = raw_img.shape[:2]

        x_offset = (bg_width - overlay_width) // 2
        y_offset = (bg_height - overlay_height) // 2

        black[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = raw_img

        img = black       
        
        w1 = abs(bottomRight[0] - bottomLeft[0])
        w2 = abs(topRight[0] - topLeft[0])
        h1 = abs(topRight[1] - bottomRight[1])
        h2 = abs(topLeft[1] - bottomLeft[1])
        width = int(max([w1, w2])) 
        height = int(max([h1, h2]))  

        pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])
        pts2 = np.float32([[0, 0], [width - 1, 0],[width - 1, height - 1], [0, height - 1]])

        mtrx = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img, mtrx, (width, height))
        r , c = result.shape[:2]
        
        transformed_people_head = cv2.perspectiveTransform(np.float32(people_head) , mtrx )
        
        result = cv2.resize(result , (600,600) , interpolation=cv2.INTER_LINEAR)
        transformed_people_head[:, :, 1] *= 600/r
        transformed_people_head[:, :, 0] *= 600/c


        for i in transformed_people_head:
            cv2.circle(result , np.int32(i[0]) , radius=4 , color=(0,255,0) , thickness=-1)

        return result , transformed_people_head.tolist()

from typing import List
from io import BytesIO

class Plot__():
    def __init__(self):
        self.boarding_state = 'start'
        self.break_count = 0
        
    def __call__(self , people_bt : List , people_or:List , people_ct :List):
        transformed_people_bottom = people_bt
        people_ori = people_or
        people_center = people_ct
        queue = 0
        eps = 0.8
        colors = ['blue', 'green', 'purple', 'pink', 'orange', 'yellow']
        ori = []
        data = []
        noise = []
        n_ori = []
        cnt = 0

        fig = plt.figure(figsize=(8,8))
        ax2 = plt.subplot(1,1,1)
        reg_x1, reg_x2, reg_y1, reg_y2 = 150, 450, 515, 595
        rect = Rectangle((reg_x1/100, 6-reg_y2/100), (reg_x2-reg_x1)/100, (reg_y2-reg_y1)/100, linewidth=1, edgecolor='g', facecolor='none')
        ax2.add_patch(rect)
        transformed_people_bottom = list(itertools.chain.from_iterable(transformed_people_bottom))

        if self.boarding_state == 'start':
            for person_bottom, person_ori in zip(transformed_people_bottom, people_ori):

                bottom_x, bottom_y = person_bottom

                if 140 < person_ori < 220:
                    ori.append(person_ori)
                    data.append([bottom_x, bottom_y])

                else:
                    bx = bottom_x/100
                    by = 6-bottom_y/100
                    p_ori = person_ori-270 if person_ori+90>=360 else person_ori+90
                    noise.append((bx, by)) 
                    n_ori.append(p_ori)
                    
            points = self.__calculate_new_data(noise, n_ori, eps) 
                        
            for i, v in enumerate(points): 
                end = self.__calculate_new_data_reverse(v, n_ori[i], eps)
                ax2.plot(end[0], end[1], marker='x', color='grey', linestyle='') # 그래프 초기화
                self.__draw_arrow(end[0], end[1], n_ori[i], eps, ax2)
            
            noise = []
            n_ori = []
            points  = []
            data = [[x / 100, 6 - y / 100] for x, y in data]

            new_ori = self.__calculate_new_ori(ori)

            A = np.array(data)
            B = np.array(new_ori)
            

            print("============= 140 < ori < 220 ======================:",A)
            print("a shape",A.shape)
            sorted_indices = np.argsort(-A[:, 1]) # descending sort
            data = A[sorted_indices].tolist()
            new_ori = B[sorted_indices].tolist()

            points = self.__calculate_new_data(data, new_ori, eps) 
                
            min_y_value = float(10) 
            min_x_value = float(10)

            cluster = self.__cluster_new(data , points , eps)

            for c in cluster:
                if len(c) <= 1:
                    cluster.remove(c)
                    noise.append(c[0])

            cnt = []
            queue = 0
            cluster2 = list(itertools.chain.from_iterable(cluster))

            for index, coord in enumerate(cluster2):
                x, y = coord
                if y < min_y_value:
                    min_y_value = y
                    min_x_value = x

            min_y_queue = 0

            for i in cluster:
                if (min_x_value , min_y_value) in i:
                    cnt = len(i)
                    queue = i
                    break
                else:
                    cnt = 0

            t_people_bottom = [[x / 100, 6 - y / 100] for x, y in transformed_people_bottom]
            for i, c in enumerate(cluster): 
                for j, v in enumerate(c):
                    if v == (min_x_value, min_y_value):
                        min_y_queue = v[1]
                    
                    idx = data.index(list(v))
                    idx2 = t_people_bottom.index([v[0], v[1]])
                    self.__draw_arrow(v[0], v[1], new_ori[idx], eps, ax2)

                    if c == queue:
                        complete_time = 0
                        self.__draw_eps(points[idx][0], points[idx][1], eps*2, 'red', ax2)
                        ax2.plot(v[0], v[1], marker='o', color='red', linestyle='') 
                    else:
                        self.__draw_eps(points[idx][0], points[idx][1], eps*2, colors[i], ax2)
                        ax2.plot(v[0], v[1], marker='.', color=colors[i], linestyle='') 
                        
            for i, v in enumerate(noise):
                idx = data.index(list(v))
                ax2.plot(v[0], v[1], marker='x', color='grey', linestyle='') 
                self.__draw_arrow(v[0], v[1], new_ori[idx], eps, ax2)
    



            if cnt != 0:
                print('queue counting: '+ str(cnt))
                ax2.set_title('queue counting: '+ str(cnt))
            else:
                print('**queue counting: 0**')
                ax2.set_title('queue counting: 0')


            self.break_count = 0
            try:
                if min_y_queue >=3 or cnt <=1:
                    self.break_count +=1
                    if self.break_count == 2:
                        self.boarding_state = 'END'
                        print("===========================  Boarding state ENDING ============================")
            except Exception as e:
                print(min_y_queue , cnt)
                print(e)


        elif self.boarding_state == 'END':
            if len(transformed_people_bottom) > 0:
                for i , p in enumerate(transformed_people_bottom):
                    bottom_x , bottom_y = p
                    bx = bottom_x / 100
                    by = 6 - bottom_y / 100

                    p_ori = people_ori[i]-270 if people_ori[i]+90>=360 else people_ori[i]+90
  
                    if reg_x1 < bottom_x < reg_x2 and reg_y1 < bottom_y < reg_y2:
                        
                        self.__draw_arrow(bx, by, p_ori, eps, ax2)
                        
                        # if 100 < people_ori[i] < 260:
                        if 140 < people_ori[i] < 220:
                            
                            in_region = True
                            
                            ax2.plot(bx, by, marker='o', color='red', linestyle='') # 그래프 초기화                     
                            
                        else: 
                            ax2.plot(bx, by, marker='o', color='black', linestyle='') # 그래프 초기화
                    else:
                        ax2.plot(bx, by, marker='x', color='grey', linestyle='') # 그래프 초기화
            

        transformed_people_bottom.clear()
        people_ori.clear()
        people_center.clear()
        
        buf = BytesIO()
        fig.savefig(buf , format='png')
        buf.seek(0)
        img_array = np.frombuffer(buf.getvalue() , dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_array , cv2.IMREAD_COLOR)
        
        plt.close(fig)
        return img
    
    def __calculate_new_data_reverse(self , data, angle_deg, length):
        if angle_deg+180>360:
            ori = angle_deg-180
        else:
            ori = angle_deg+180
        angle_rad = np.deg2rad(ori)
        end = (data[0] + length * np.cos(angle_rad), data[1] + length * np.sin(angle_rad))

        return end
    
    def __draw_eps(self , x, y, eps, colors, ax):
        ax.plot(x, y, marker='o', linestyle='', alpha=0.15, color=colors, markersize=eps*53)

    def __cluster_new(self, data, points, eps):
        cluster = []
        for p in points:
            c = []
            for d in data:

                if points.index(p) != data.index(d) :
                    dist = self.__Euclidean(d, p)
                    if dist < eps:
                        c.append(d)
                else:
                    c.append(d)
            cluster.append(c)
        cluster = self.__merge_lists(cluster)
        return cluster
    
    def __calculate_new_ori(self , ori):
        new_ori = []
        for o in ori:
            if o+90>=360:
                new_ori.append(o-270)
            else:
                new_ori.append(o+90)
        return new_ori
    
    
    def __calculate_new_data(self , data, angle_deg, length):
        new_data = []
        for d in range(len(data)):
            angle_rad = np.deg2rad(angle_deg[d])
            end = (data[d][0] + length * np.cos(angle_rad), data[d][1] + length * np.sin(angle_rad))
            new_data.append(end)
        return new_data

    def __draw_arrow(self,x, y, angle_deg, length, ax):
        angle_rad = np.deg2rad(angle_deg)
        dx = length * np.cos(angle_rad)
        dy = length * np.sin(angle_rad)
        ax.arrow(x, y, dx, dy, head_width=0.07, head_length=0.07, fc='black', ec='black')

    def __Euclidean(self,v1, v2): 
        import math
        sum = 0
        for i in range(len(v1)):
            sum +=  (v1[i]-v2[i])**2
        return round(math.sqrt(sum), 2)
    
    def __merge_lists(self ,lists):
        
        def find_root(roots, i):
            while roots[i] != i:
                i = roots[i]
            return i
        
        def union(roots, ranks, i, j):
            root_i = find_root(roots, i)
            root_j = find_root(roots, j)
            if root_i != root_j:
                if ranks[root_i] > ranks[root_j]:
                    roots[root_j] = root_i
                elif ranks[root_i] < ranks[root_j]:
                    roots[root_i] = root_j
                else:
                    roots[root_j] = root_i
                    ranks[root_i] += 1
                
        n = len(lists)
        roots = list(range(n))
        ranks = [0] * n
        element_to_root = {}  # Dictionary to map element to its root index
        
        for i, lst in enumerate(lists):
            for elem in lst:
                elem_tuple = tuple(elem)  # Convert element to tuple to use as dictionary key
                if elem_tuple in element_to_root:
                    union(roots, ranks, i, element_to_root[elem_tuple])
                element_to_root[elem_tuple] = find_root(roots, i)
        
        index_to_elements = {}
        for i, lst in enumerate(lists):
            root = find_root(roots, i)
            if root not in index_to_elements:
                index_to_elements[root] = set()
            index_to_elements[root].update(tuple(e) for e in lst)
        
        result = [sorted(list(elements)) for elements in index_to_elements.values() if elements]
        return result


def main(args=None):
    rclpy.init(args=args)
    
    board = Board()
    
    rclpy.spin(board)
    
    board.destroy_node()
    rclpy.shutdown()
    
if __name__ == "__main__":
    main()
