"""Publica las dos cámaras CSI de la cabeza a 640x480 y 15 FPS con camera_ros.

Se identifican por su ID de libcamera, no por índice: el orden de enumeración
puede cambiar entre arranques y left/right quedarían intercambiadas.
"""
from launch import LaunchDescription
from launch_ros.actions import Node

CAM_LEFT = "/base/axi/pcie@1000120000/rp1/i2c@88000/ov5647@36"
CAM_RIGHT = "/base/axi/pcie@1000120000/rp1/i2c@80000/ov5647@36"


def camara(nombre, camera_id):
    return Node(
        package="camera_ros",
        executable="camera_node",
        namespace="/head",
        name=nombre,
        parameters=[{
            "camera": camera_id,
            "width": 640,
            "height": 480,
            "FrameDurationLimits": [66666, 66666],  # 15 FPS, en microsegundos
        }],
        output="screen",
    )


def generate_launch_description():
    return LaunchDescription([
        camara("cam_left", CAM_LEFT),
        camara("cam_right", CAM_RIGHT),
    ])
