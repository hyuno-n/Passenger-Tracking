import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import rclpy.subscription
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import os
import threading
import bisect
import time

class TimeSynchronize(Node):
    qos_profile = QoSProfile( # ros2 메세지 최대 보존
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_ALL
    )

    all_cameras = {
        0: 'camera0',
        2: 'camera2',
        4: 'camera4',
    }

    def __init__(self):
        super().__init__("time_synchronize")
        self.get_logger().warn("Test Run...")
        
        self.running = True
        self.call_count = 0
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.camera_buffers = {cam_name : [] for cam_name in TimeSynchronize.all_cameras.keys()}
        
        for index in TimeSynchronize.all_cameras.keys():
            topic_name = f'{TimeSynchronize.all_cameras[index]}/image_raw'
            self.create_subscription(
                Image, topic_name, lambda msg, idx=index: self.camera_callback(msg, idx), qos_profile=TimeSynchronize.qos_profile
            )

        self.allow_time = 0.1
        self.SCEN_NAME = 'scen2'
        self.SCEN_NAME = f'{self.SCEN_NAME}_{str(self.allow_time)}origin'
        self.base_path = f'dataset/{self.SCEN_NAME}'
        self.camera_paths = {
            index: os.path.join(self.base_path, f'cam{index}') for index in TimeSynchronize.all_cameras.keys()
        }

        for path in self.camera_paths.values():
            os.makedirs(path, exist_ok=True)

        self.msg_count = {index : 0 for index in TimeSynchronize.all_cameras.keys()} # 각 토픽 메세지 수신갯수 확인
        self.saved = True
        self.buffer_size = 10000
        self.time_tolerance_ns = self.allow_time * 1e9 # +- 허용범위내에서 기준을 만족하는 메세지 찾기
        self.last_cam0_msg_time = time.time() # cam0 기준으로 cam0의 마지막 수신 시간 기록
        
        self.processing_thread = None
        self.start_processing_thread()
        
    def start_processing_thread(self):
        self.processing_thread = threading.Thread(target=self.process_messages)
        self.processing_thread.daemon = True # demon thread main종료시 데이터 매칭 발생
        self.processing_thread.start()
        
    def camera_callback(self, msg, camera_index):
        with self.lock:
            self.get_logger().info(f"{camera_index} message received { msg.header.stamp.sec}")
            msg_time_ns = self.get_msg_time(msg).nanoseconds
            self.msg_count[camera_index] += 1
            buffer = self.camera_buffers[camera_index]
            bisect.insort(buffer, (msg_time_ns, msg)) # 이진탐색 적용 buffer 리스트에서 msg_time_ns보다 큰처음 만족 인덱스에 삽입
            if len(buffer) > self.buffer_size:
                buffer.pop(0)

            if camera_index == 0:
                self.last_cam0_msg_time = time.time() # cam0 수신시간 기록

    def process_messages(self):
        
        try:
            
            while rclpy.ok() and self.running: # ros2 TimeSync 노드가 생존동안.
                time.sleep(1)
                current_time = time.time()
                time_since_last_cam0 = current_time - self.last_cam0_msg_time

                if time_since_last_cam0 >= 10:
                    self.get_logger().info(f"매칭을 시작합니다..")
                    print(current_time)
                    print(self.last_cam0_msg_time)
                    print('====================================================================')

                    with self.lock:
                        if not self.camera_buffers[0]:
                            self.get_logger().fatal("cam0 메시지가 없습니다 종료...")
                            self.running = False
                            return
                                                
                        while self.camera_buffers[0]:
                            cam0_time_ns, cam0_msg = self.camera_buffers[0].pop(0)
                            self.process_cam0_message(cam0_time_ns, cam0_msg)

                        self.get_logger().info("SAVED FINISHED")
                        for index , count  in self.msg_count.items():
                            self.get_logger().info(f"cam{index} message {count}")                    
                        else:
                            self.running = False
                            return
                        
            return

        except Exception as e:
            self.get_logger().warn(f"Exception Process_messages : {str(e)}")
        finally:
            self.shutdown_node()

           
    def process_cam0_message(self, cam0_time_ns, cam0_msg):
        cam0_image = self.bridge.imgmsg_to_cv2(cam0_msg, desired_encoding="bgr8")
        if self.saved:
            cv2.imwrite(os.path.join(self.camera_paths[0], f"{self.call_count}.jpg"), cam0_image)
        
        self.get_logger().info(f"Camera0 {cam0_msg.header.stamp.sec} matced start")
        for index in TimeSynchronize.all_cameras.keys():
            if index == 0:
                continue
            else:
                msg = self.find_closest_message(
                    self.camera_buffers[index], cam0_time_ns
                )
                if msg:
                    image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    if self.saved:
                        cv2.imwrite(os.path.join(self.camera_paths[index], f"{self.call_count}.jpg"), image)
                    self.get_logger().info(f"Camera {index} message saved")
                else:
                    self.get_logger().warn(f"Camera {index} message None")

        self.call_count += 1

    def find_closest_message(self, buffer, target_time_ns):
        """ buffer : 2 4 6 8 
            target_time_ns : 검사 cam0 ns
        """
        if not buffer:
            return None

        lower_bound = target_time_ns - self.time_tolerance_ns # 하한선
        upper_bound = target_time_ns + self.time_tolerance_ns # 상한선
        times = [item[0] for item in buffer] # idx카메라의 모든 msg_time_ns를 저장
        left = bisect.bisect_left(times, lower_bound) # 하한선보다 times의 인덱스
        right = bisect.bisect_right(times, upper_bound) # 상한선보다 조건 처음만족하는 times의 인덱스

        if left >= right:
            return None

        closest_msg = min(
            buffer[left:right],
            key=lambda x: abs(x[0] - target_time_ns)
        )[1]

        return closest_msg
    
    def shutdown_node(self):
        """노드와 스레드를 안전하게 종료"""
        self.running = False 

        if threading.current_thread() is not self.processing_thread:
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join()  
                self.get_logger().info("Shutting down thread...")               

    def get_msg_time(self, msg):
        return rclpy.time.Time.from_msg(msg.header.stamp)

    def destroy_node(self):
        self.processing_thread.join(timeout=1)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = TimeSynchronize()

    try:
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        node.shutdown_node()
        node.destroy_node()
        exit(0)
        
    finally:
        
        node.shutdown_node()
        node.destroy_node()
        

if __name__ == '__main__':
    main()