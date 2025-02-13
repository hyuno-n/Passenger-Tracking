import matplotlib.pylab
import matplotlib.pyplot
import  numpy as np
from sklearn.cluster import  DBSCAN
from sklearn.preprocessing import  StandardScaler
from ultralytics import  YOLO
import cv2
import  os
import itertools
import  matplotlib.pyplot as plt
import  matplotlib
from src.pose_recognition.tools.hboe import hboe

os.environ["QT_LOGGING_RULES"] = "*.debug=false"

class Scan():
    def __init__(self):
        self.model = YOLO("yolov10x.pt").to("cuda")
        self.color_map = {
                    'blue': (255, 0, 0),
                    'green': (0, 255, 0),
                    'purple': (128, 0, 128),
                    'pink': (255, 192, 203),
                    'orange': (0, 165, 255),
                    'yellow': (0, 255, 255)
                }
        
        self.colors = list(self.color_map.values())
        

    def forward(self , image : np.array):
        

        param_image = image.copy()
        image_copy = image.copy()
        results = self.model(image , classes = 0 , verbose=False)
        boxes = results[0].boxes
        ori_list = []
        bottom_list = []
        for i , box in enumerate(boxes):
            x1 ,y1 , x2 ,y2 = list(map(int , box.xyxy[0]))
            cv2.rectangle(image , (x1,y1) , (x2,y2) , color=self.colors[i%len(self.colors)], thickness=2)
            bottom_x = (x2 + x1) // 2
            bottom_y =  y2
            print(bottom_x , '---', bottom_y)
            cv2.circle(image , (bottom_x , bottom_y) , radius=2 , color=(0,0,255),thickness=2)

            crop_person = param_image[y1 : y2 , x1 : x2]
            ori = hboe(crop_person)
            ori_list.append(ori)

            if int((x1+x2)/2) >= 60 and int(y2) >= 125:
                if y1 + (x2-x1)*2.3 > 600 or y2-y1 > (x2-x1)*2:
                    people_bottom = ([[int((x1+x2)/2), int(y2)]])
                    bottom_list.append(people_bottom)
                else:
                    people_bottom = ([[int((x1+x2)/2), int(y1 + (x2-x1)*2.3)]])
                    bottom_list.append(people_bottom)
                
                x ,y = list(itertools.chain.from_iterable(people_bottom))
                cv2.rectangle(image_copy , (x1,y1) , (x2,y2) ,color=self.colors[i%len(self.colors)] , thickness=2)
                cv2.circle(image_copy , (x,y) ,radius=2 ,color=(0,255,0),thickness=2)

        #cv2.imshow("image_copy",image_copy)
        #cv2.imshow("test",image)
        trans_loc = self.BEV_adative(param_image , bottom_list)

        # for i in bottom_list:
        #     x , y = i[0]
        #     plt.scatter(x ,y , color = 'red')

        # plt.show()
        self.DBSCAN(trans_loc, ori_list)

    def BEV_adative(self , image : np.array , bottom_list : np.array):

        raw_img = image
        people_head = np.array(bottom_list)


        topLeft = [625, 725]
        bottomRight = [1800, 1200]  
        topRight = [1125, 725]  
        bottomLeft = [0, 1200]  

        rows , _  = raw_img.shape[:2]

        black = np.zeros((rows*3, rows*3, 3), np.uint8)
        bg_height, bg_width = black.shape[:2]
        overlay_height, overlay_width = raw_img.shape[:2]

        x_offset = (bg_width - overlay_width) // 2
        y_offset = (bg_height - overlay_height) // 2
        
        people_head[: , 0 , 0] += x_offset
        people_head[ : , 0 , 1] += y_offset
        people_head = people_head.tolist()
        
        black[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = raw_img

        img = black       

        BUFFER = [topLeft , topRight , bottomRight , bottomLeft]
        for i in BUFFER:
            cv2.circle(img , i , radius=10, color=(255, 0 ,0) , thickness=-1)

        for i in people_head:
            i = list(itertools.chain.from_iterable(i))
            cv2.circle(img , i , radius=7 , color=(170,255,195) , thickness=-1)

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


        img = cv2.resize(img , (600,600))
        result = cv2.resize(result , (600,600))
        cv2.imwrite('db_scan_res_target_image.jpg',result)
        #cv2.imshow('original', img)
        #cv2.imshow('result', result)
        # cv2.imwrite("oo.jpg",img)
        # cv2.imwrite('rr.jpg',result)
        #cv2.waitKey(0)
        
        return list(itertools.chain.from_iterable(transformed_people_head.tolist()))


    def DBSCAN(self,cordinate , ori):

        ori = list(map(int , ori))
        cordinate = np.float32(cordinate)
        cordinate[ : , 1] = 600 - cordinate[ : , 1]

        colors = ['red','blue','green','pink']
        fig = plt.figure(figsize=(8,8))
        ax1 = plt.subplot(1,1,1)

        db_cordinate = StandardScaler().fit_transform(cordinate)
        db = DBSCAN(eps=0.9 , min_samples=2)
        labels = db.fit(db_cordinate)

        print(db_cordinate)
        labels = db.labels_

        db_cordinate_np = np.float32(db_cordinate)
        for cordinate , labels , ori in zip(db_cordinate_np , labels , ori):
            x_cordinate , y_cordinate = cordinate
            if labels == -1:
                color = 'black'
                ax1.plot(x_cordinate , y_cordinate , marker = 'x' , color = color)
                self.draw_arrow(x_cordinate , y_cordinate , color , ori , ax1)
                continue
            else:
                color = colors[labels % len(colors)]
            ax1.scatter(x_cordinate  , y_cordinate , color = color)
            self.draw_arrow(x_cordinate , y_cordinate , color ,ori , ax1)
 
        plt.savefig("db_scan_res.png")
        plt.show()
        
        
    def draw_arrow(self ,x , y , color , ori ,ax):
        head_length = 0.07
        ori = ori + 90
        angle_rad = np.deg2rad(ori)
        dx  = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        ax.arrow(x, y, dx, dy, head_width=0.07, head_length=head_length, fc='black', ec='black')

        arrow_end_x = x + dx
        arrow_end_y = y + dy
        norm = np.sqrt(dx**2 + dy**2)

        arrow_head_x = arrow_end_x - (dx / norm) * head_length
        arrow_head_y = arrow_end_y - (dy / norm) * head_length

        if color != 'black':
            self.draw_eps(arrow_head_x , arrow_head_y , color , ax)

    def draw_eps(self ,x , y, color , ax):
        eps = 0.9
        ax.plot(x, y, marker='o', linestyle='', alpha=0.15, color=color, markersize=eps*100)

if __name__ == "__main__":
    image = cv2.imread('node_image/NON_IMAGE/17.jpg')
    if image is None:
        exit
    scan = Scan()
    scan.forward(image)