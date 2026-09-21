# src/coramo_bringup/launch/cerebro.launch.py
"""Levanta el cerebro completo. El cuerpo va simulado salvo en el perfil xeon."""
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _armar(context):
    perfil = LaunchConfiguration("perfil").perform(context)
    params = str(Path(__file__).resolve().parents[1] / "params" / f"{perfil}.yaml")
    comunes = dict(package="coramo_brain", parameters=[params], output="screen")
    nodos = [Node(executable=e, name=n, **comunes) for e, n in
             [("speech", "speech"), ("agent", "agent"), ("tts", "tts"),
              ("safety", "safety"), ("supervisor", "supervisor")]]
    if perfil != "xeon":
        nodos.append(Node(executable="body_bridge_sim", name="body_bridge", **comunes))
    return nodos


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("perfil", default_value="dev-sin-robot",
                              description="xeon, dev-sin-robot o dev-sin-gpu"),
        OpaqueFunction(function=_armar),
    ])
