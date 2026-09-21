cmake_minimum_required(VERSION 3.8)
project(coramo_msgs)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Transcript.msg"
  "msg/BodyCommand.msg"
  "msg/Event.msg"
  "msg/State.msg"
  DEPENDENCIES std_msgs builtin_interfaces)
ament_package()
