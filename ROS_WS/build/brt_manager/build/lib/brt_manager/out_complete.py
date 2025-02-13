# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import numpy as np
# import cv2
# import os
# from collections import defaultdict
# from shapely.geometry import Polygon
# from shapely.geometry.point import Point
# from ultralytics import YOLO
# from ultralytics.utils.plotting import Annotator, colors
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# class OutComplete(Node):
#     def __init__(self):
#         super().__init__('out_complete')

#         self.subscription_door1 = self.create_subscription(
#             Image,
#             'door1_under', 
#             self.door1_callback,
#             1)
#         self.subscription_door1

#         self.subscription_door2 = self.create_subscription(
#             Image,
#             'door2_under', 
#             self.door2_callback,
#             1)
#         self.subscription_door2
#         '''
#         self.subscription_door3 = self.create_subscription(
#             Image,
#             'door3_under', 
#             self.door3_callback,
#             1)
#         self.subscription_door3
#         '''
#         self.people_door1 = []
#         self.people_door2 = []
#         '''
#         self.people_door3 = []

#         '''
#         self.frame_door1 = None
#         self.frame_door2 = None
#         '''
#         self.frame_door3 = None
#         '''

#         self.door1_region_cnt = None
#         self.door2_region_cnt = None
#         # self.door3_region_cnt = None

#         self.door1_cnt_3sec = [999, 999, 999]
#         self.door2_cnt_3sec = [999, 999, 999]
#         # self.door3_cnt_3sec = [999, 999, 999]

#         self.door1_out_complete = False
#         self.door2_out_complete = False
#         self.door3_out_complete = False

#         self.bridge = CvBridge()

#         self.model = YOLO('yolov8x.pt')

#         self.timer = self.create_timer(1, self.out_complete_pub)

#         self.publisher_door1_detect = self.create_publisher(Image,'detect_door1_under', 1)
#         self.publisher_door2_detect = self.create_publisher(Image,'detect_door2_under', 1)
#         '''
#         self.publisher_door3_detect = self.create_publisher(Image,'detect_door3_under', 1)
#         '''

#         self.publisher_outout_count = self.create_publisher(Image, 'inout_count')

#     def door1_callback(self, msg):

#         self.frame_door1 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

#         counting_regions = [
#             {
#                 "name": "YOLOv8 Polygon Region",
#                 # polygon out 영역으로 수정 #
#                 "polygon": Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),  # Polygon points
#                 "counts": 0,
#                 "dragging": False,
#                 "region_color": (37, 255, 225),  # BGR Value
#                 "text_color": (0, 0, 0),  # Region Text Color
#             },
#         ]

#         # Extract the results
#         results = self.model.track(self.frame_door1, classes=0)

#         if results[0].boxes.id is not None:
#             boxes = results[0].boxes.xyxy.cpu()
#             track_ids = results[0].boxes.id.int().cpu().tolist()
#             clss = results[0].boxes.cls.cpu().tolist()

#             annotator = Annotator(self.frame_door1, line_width=2, example=self.model.names)

#             for box, track_id, cls in zip(boxes, track_ids, clss):
#                 annotator.box_label(box, str(self.model.names[cls]), color=colors(cls, True))
#                 bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

#                 for region in counting_regions:
#                     if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
#                         region["counts"] += 1

#         # Draw regions (Polygons/Rectangles)
#         for region in counting_regions:
#             region_label = str(region["counts"])
#             region_color = region["region_color"]
#             region_text_color = region["text_color"]

#             polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
#             centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

#             text_size, _ = cv2.getTextSize(
#                 region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2
#             )
#             text_x = centroid_x - text_size[0] // 2
#             text_y = centroid_y + text_size[1] // 2
#             cv2.rectangle(
#                 self.frame_door1,
#                 (text_x - 5, text_y - text_size[1] - 5),
#                 (text_x + text_size[0] + 5, text_y + 5),
#                 region_color,
#                 -1,
#             )
#             cv2.putText(
#                 self.frame_door1, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, 2
#             )
#             cv2.polylines(self.frame_door1, [polygon_coords], isClosed=True, color=region_color, thickness=2)

#         self.door1_region_cnt = region["counts"]            
#         detection_msg = self.bridge.cv2_to_imgmsg(self.frame_door1, encoding='bgr8')

#         self.publisher_door1_detect.publish(detection_msg)

#     def door2_callback(self, msg):

#         self.frame_door2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

#         counting_regions = [
#             {
#                 "name": "YOLOv8 Polygon Region",
#                 # polygon 영역 수정 #
#                 "polygon": Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),  # Polygon points
#                 "counts": 0,
#                 "dragging": False,
#                 "region_color": (37, 255, 225),  # BGR Value
#                 "text_color": (0, 0, 0),  # Region Text Color
#             },
#         ]

#         # Extract the results
#         results = self.model.track(self.frame_door2, classes=0)

#         if results[0].boxes.id is not None:
#             boxes = results[0].boxes.xyxy.cpu()
#             track_ids = results[0].boxes.id.int().cpu().tolist()
#             clss = results[0].boxes.cls.cpu().tolist()

#             annotator = Annotator(self.frame_door2, line_width=2, example=self.model.names)

#             for box, track_id, cls in zip(boxes, track_ids, clss):
#                 annotator.box_label(box, str(self.model.names[cls]), color=colors(cls, True))
#                 bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

#                 for region in counting_regions:
#                     if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
#                         region["counts"] += 1

#         # Draw regions (Polygons/Rectangles)
#         for region in counting_regions:
#             region_label = str(region["counts"])
#             region_color = region["region_color"]
#             region_text_color = region["text_color"]

#             polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
#             centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

#             text_size, _ = cv2.getTextSize(
#                 region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2
#             )
#             text_x = centroid_x - text_size[0] // 2
#             text_y = centroid_y + text_size[1] // 2
#             cv2.rectangle(
#                 self.frame_door2,
#                 (text_x - 5, text_y - text_size[1] - 5),
#                 (text_x + text_size[0] + 5, text_y + 5),
#                 region_color,
#                 -1,
#             )
#             cv2.putText(
#                 self.frame_door2, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, 2
#             )
#             cv2.polylines(self.frame_door2, [polygon_coords], isClosed=True, color=region_color, thickness=2)

#         self.door1_region_cnt = region["counts"] 

#         detection_msg = self.bridge.cv2_to_imgmsg(self.frame_door2, encoding='bgr8')

#         self.publisher_door2_detect.publish(detection_msg)

#     '''
#     def door3_callback(self, msg):

#         self.frame_door3 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

#         counting_regions = [
#             {
#                 "name": "YOLOv8 Polygon Region",
#                 # polygon 영역 수정 #
#                 "polygon": Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),  # Polygon points
#                 "counts": 0,
#                 "dragging": False,
#                 "region_color": (37, 255, 225),  # BGR Value
#                 "text_color": (0, 0, 0),  # Region Text Color
#             },
#         ]

#         # Extract the results
#         results = self.model.track(self.frame_door3, classes=0)

#         if results[0].boxes.id is not None:
#             boxes = results[0].boxes.xyxy.cpu()
#             track_ids = results[0].boxes.id.int().cpu().tolist()
#             clss = results[0].boxes.cls.cpu().tolist()

#             annotator = Annotator(self.frame_door3, line_width=2, example=self.model.names)

#             for box, track_id, cls in zip(boxes, track_ids, clss):
#                 annotator.box_label(box, str(self.model.names[cls]), color=colors(cls, True))
#                 bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

#                 for region in counting_regions:
#                     if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
#                         region["counts"] += 1

#         # Draw regions (Polygons/Rectangles)
#         for region in counting_regions:
#             region_label = str(region["counts"])
#             region_color = region["region_color"]
#             region_text_color = region["text_color"]

#             polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
#             centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

#             text_size, _ = cv2.getTextSize(
#                 region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2
#             )
#             text_x = centroid_x - text_size[0] // 2
#             text_y = centroid_y + text_size[1] // 2
#             cv2.rectangle(
#                 self.frame_door3,
#                 (text_x - 5, text_y - text_size[1] - 5),
#                 (text_x + text_size[0] + 5, text_y + 5),
#                 region_color,
#                 -1,
#             )
#             cv2.putText(
#                 self.frame_door3, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, 2
#             )
#             cv2.polylines(self.frame_door3, [polygon_coords], isClosed=True, color=region_color, thickness=2)

#         self.door1_region_cnt = region["counts"]            
#         detection_msg = self.bridge.cv2_to_imgmsg(self.frame_door3, encoding='bgr8')

#         self.publisher_door3_detect.publish(detection_msg)
#     '''

#     def out_complete_pub(self):
        
#         self.door1_cnt_3sec.pop(0)
#         self.door2_cnt_3sec.pop(0)
#         # self.door3_cnt_3sec.pop(0)
#         self.door1_cnt_3sec.append(self.door1_region_cnt)
#         self.door2_cnt_3sec.append(self.door2_region_cnt)
#         # self.door3_cnt_3sec.append(self.door3_region_cnt)

#         if sum(self.door1_cnt_3sec) == 0 and self.door1_out_complete is False:
#             print("문 1 하차 완료")
#             self.door1_out_complete = True

#         if sum(self.door2_cnt_3sec) == 0 and self.door2_out_complete is False:
#             print("문 2 하차 완료")
#             self.door2_out_complete = True
#         '''
#         if sum(self.door3_cnt_3sec) == 0 and self.door3_out_complete is False:
#             print("문 3 하차 완료")
#             self.door3_out_complete = True
#         '''
#         if self.door1_out_complete is True and self.door2_out_complete is True: # and self.door3_out_complete is True:
#             print("모든 문 하차 완료")

# def main(args=None):
#     rclpy.init(args=args)

#     node = OutComplete()

#     rclpy.spin(node)

#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__maout__':
#     main()