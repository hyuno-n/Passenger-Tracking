# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2

# class ImageSubscriber(Node):
#     def __init__(self):
#         super().__init__('image_subscriber')
#         # 이미지를 구독하는 구독자 생성
#         self.subscription = self.create_subscription(
#             Image,'camera0/image_raw',
#             self.image_callback,10)
#         self.subscription  # prevent unused variable warning
#         self.br = CvBridge()

#         # 이미지를 발행하는 발행자 생성
#         self.publisher = self.create_publisher(
#             Image,
#             '/processed_image',
#             10)

#     def image_callback(self, msg):
#         # 받은 이미지 데이터를 OpenCV 형식으로 변환
#         cv_image = self.br.imgmsg_to_cv2(msg, "bgr8")
#         cv2.imshow("Camera Image", cv_image)
#         cv2.waitKey(1)

#         # 변환된 이미지 데이터를 다시 ROS 메시지로 변환
#         processed_image_msg = self.br.cv2_to_imgmsg(cv_image, "bgr8")

#         # 변환된 이미지 메시지를 새 토픽으로 발행
#         self.publisher.publish(processed_image_msg)

# def main(args=None):
#     rclpy.init(args=args)
#     image_subscriber = ImageSubscriber()
#     rclpy.spin(image_subscriber)
#     image_subscriber.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()
