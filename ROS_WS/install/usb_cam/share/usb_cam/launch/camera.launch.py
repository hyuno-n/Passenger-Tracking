from launch import LaunchDescription
from launch_ros.actions import Node
#


def generate_launch_description():
    camera_configs = {
        "camera0": "/dev/video0",
        "camera2": "/dev/video2",
        "camera4": "/dev/video4",
        "camera6": "/dev/video6",
        "camera8": "/dev/video8",
        "camera10": "/dev/video10"
    }

    nodes = []
    for camera_name, video_device in camera_configs.items():
        params = {
            'video_device': video_device,
            'framerate': 10.0,
            'pixel_format': "uyvy",
            'image_width': 1920,
            'image_height': 1080,
            'camera_name': camera_name,
            'camera_info_url': f"package://usb_cam/config/{camera_name}_info.yaml",
            # 추가 파라미터 설정
        }

        node = Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            parameters=[params],
            remappings=[
                ('image_raw', f'{camera_name}/image_raw'),
                ('image_raw/compressed', f'{camera_name}/image_raw/compressed'),
            ],
        )
        

        nodes.append(node)

    return LaunchDescription(nodes)