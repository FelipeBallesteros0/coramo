"""Publica las dos cámaras CSI de la cabeza a 640x480 y 15 FPS con camera_ros."""
from launch import LaunchDescription
from launch_ros.actions import Node


def camara(nombre, indice):
    return Node(package="camera_ros", executable="camera_node", namespace=f"/head/{nombre}", name="camera",
                parameters=[{"camera": indice, "width": 640, "height": 480, "format": "YUYV",
                             "FrameDurationLimits": [66666, 66666]}])  # 15 FPS en microsegundos


def generate_launch_description():
    return LaunchDescription([camara("cam_left", 0), camara("cam_right", 1)])
