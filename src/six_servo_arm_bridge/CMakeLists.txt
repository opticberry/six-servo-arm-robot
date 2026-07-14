cmake_minimum_required(VERSION 3.8)
project(six_servo_arm_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(six_servo_arm_msgs REQUIRED)

add_executable(joint_state_to_arm_cmd src/joint_state_to_arm_cmd.cpp)

ament_target_dependencies(joint_state_to_arm_cmd
  rclcpp
  sensor_msgs
  six_servo_arm_msgs
)

install(TARGETS
  joint_state_to_arm_cmd
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()