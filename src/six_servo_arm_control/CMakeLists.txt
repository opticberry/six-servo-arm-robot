cmake_minimum_required(VERSION 3.8)
project(six_servo_arm_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(six_servo_arm_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

add_executable(tool0_pose_target src/tool0_pose_target.cpp)
ament_target_dependencies(tool0_pose_target
  rclcpp
  moveit_ros_planning_interface
  geometry_msgs
)

install(TARGETS
  tool0_pose_target
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

add_executable(joint_target src/joint_target.cpp)
ament_target_dependencies(joint_target
  rclcpp
  moveit_ros_planning_interface
)

install(TARGETS
  joint_target
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(approx_ik_joint_search src/approx_ik_joint_search.cpp)
ament_target_dependencies(approx_ik_joint_search
  rclcpp
  moveit_ros_planning_interface
)

install(TARGETS
  approx_ik_joint_search
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(approx_ik_joint_search_with_direction src/approx_ik_joint_search_with_direction.cpp)
ament_target_dependencies(approx_ik_joint_search_with_direction
  rclcpp
  moveit_ros_planning_interface
)

install(TARGETS
  approx_ik_joint_search_with_direction
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(approx_ik_joint_search_seed src/approx_ik_joint_search_seed.cpp)
ament_target_dependencies(approx_ik_joint_search_seed
  rclcpp
  moveit_ros_planning_interface
)

install(TARGETS
  approx_ik_joint_search_seed
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(approx_ik_joint_search_seed2 src/approx_ik_joint_search_seed2.cpp)
ament_target_dependencies(approx_ik_joint_search_seed2
  rclcpp
  moveit_ros_planning_interface
)

install(TARGETS
  approx_ik_joint_search_seed2
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(approx_ik_with_pregrasp_flow src/approx_ik_with_pregrasp_flow.cpp)
ament_target_dependencies(approx_ik_with_pregrasp_flow
  rclcpp
  moveit_ros_planning_interface
)

install(TARGETS
  approx_ik_with_pregrasp_flow
  DESTINATION lib/${PROJECT_NAME}
)

# log预输出STM32指令
add_executable(approx_ik_with_pregrasp_flow2 src/approx_ik_with_pregrasp_flow2.cpp)
ament_target_dependencies(approx_ik_with_pregrasp_flow2
  rclcpp
  moveit_ros_planning_interface
)

install(TARGETS
  approx_ik_with_pregrasp_flow2
  DESTINATION lib/${PROJECT_NAME}
)

# position IK
add_executable(position_ik src/position_ik.cpp)
ament_target_dependencies(position_ik
  rclcpp
  moveit_ros_planning_interface
)

install(TARGETS
  position_ik
  DESTINATION lib/${PROJECT_NAME}
)

# position IK with close to pregrasp
add_executable(position_ik2 src/position_ik2.cpp)
ament_target_dependencies(position_ik2
  rclcpp
  moveit_ros_planning_interface
)

install(TARGETS
  position_ik2
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(kdl_position_only src/kdl_position_only.cpp)
ament_target_dependencies(kdl_position_only
  rclcpp
  moveit_ros_planning_interface
)

install(TARGETS
  kdl_position_only
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(selected_block_mm_to_moveit src/selected_block_mm_to_moveit.cpp)
ament_target_dependencies(selected_block_mm_to_moveit
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
)

install(TARGETS selected_block_mm_to_moveit
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(selected_block_mm_to_moveit_busy src/selected_block_mm_to_moveit_busy.cpp)
ament_target_dependencies(selected_block_mm_to_moveit_busy
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
)

install(TARGETS selected_block_mm_to_moveit_busy
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(selected_block_mm_to_moveit_busy_with_gripper_postgrasp
  src/selected_block_mm_to_moveit_busy_with_gripper_postgrasp.cpp
)

ament_target_dependencies(selected_block_mm_to_moveit_busy_with_gripper_postgrasp
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
)

install(TARGETS
  selected_block_mm_to_moveit_busy_with_gripper_postgrasp
  DESTINATION lib/${PROJECT_NAME}
)
# 手眼标定
add_executable(selected_block_mm_to_moveit_busy_with_gripper_postgrasp2
  src/selected_block_mm_to_moveit_busy_with_gripper_postgrasp2.cpp
)

ament_target_dependencies(selected_block_mm_to_moveit_busy_with_gripper_postgrasp2
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
)

install(TARGETS
  selected_block_mm_to_moveit_busy_with_gripper_postgrasp2
  DESTINATION lib/${PROJECT_NAME}
)
# rule模式
add_executable(selected_block_mm_to_moveit_busy_with_gripper_postgrasp3
  src/selected_block_mm_to_moveit_busy_with_gripper_postgrasp3.cpp
)

ament_target_dependencies(selected_block_mm_to_moveit_busy_with_gripper_postgrasp3
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
)

install(TARGETS
  selected_block_mm_to_moveit_busy_with_gripper_postgrasp3
  DESTINATION lib/${PROJECT_NAME}
)

# ppo模式
add_executable(selected_block_mm_to_moveit_busy_with_gripper_postgrasp4
  src/selected_block_mm_to_moveit_busy_with_gripper_postgrasp4.cpp
)

ament_target_dependencies(selected_block_mm_to_moveit_busy_with_gripper_postgrasp4
  rclcpp
  geometry_msgs
  six_servo_arm_msgs
  moveit_ros_planning_interface
)

install(TARGETS
  selected_block_mm_to_moveit_busy_with_gripper_postgrasp4
  DESTINATION lib/${PROJECT_NAME}
)

# ppo妯″紡 + MoveIt trajectory retime
add_executable(selected_block_mm_to_moveit_busy_with_gripper_postgrasp5
  src/selected_block_mm_to_moveit_busy_with_gripper_postgrasp5.cpp
)

ament_target_dependencies(selected_block_mm_to_moveit_busy_with_gripper_postgrasp5
  rclcpp
  geometry_msgs
  sensor_msgs
  six_servo_arm_msgs
  moveit_ros_planning_interface
)

install(TARGETS
  selected_block_mm_to_moveit_busy_with_gripper_postgrasp5
  DESTINATION lib/${PROJECT_NAME}
)

# ppo妯″紡 + unified STM32 executor
add_executable(selected_block_mm_to_moveit_busy_with_gripper_postgrasp6
  src/selected_block_mm_to_moveit_busy_with_gripper_postgrasp6.cpp
)

ament_target_dependencies(selected_block_mm_to_moveit_busy_with_gripper_postgrasp6
  rclcpp
  geometry_msgs
  sensor_msgs
  six_servo_arm_msgs
  moveit_ros_planning_interface
)

install(TARGETS
  selected_block_mm_to_moveit_busy_with_gripper_postgrasp6
  DESTINATION lib/${PROJECT_NAME}
)

# ppo local servo through MoveIt plan + external joint_state_to_arm_cmd bridge
add_executable(selected_block_mm_to_moveit_busy_with_gripper_postgrasp7
  src/selected_block_mm_to_moveit_busy_with_gripper_postgrasp7.cpp
)

ament_target_dependencies(selected_block_mm_to_moveit_busy_with_gripper_postgrasp7
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
)

install(TARGETS
  selected_block_mm_to_moveit_busy_with_gripper_postgrasp7
  DESTINATION lib/${PROJECT_NAME}
)

# ppo local servo + MoveIt retimed trajectory executed directly through /arm_joint_cmd
add_executable(selected_block_mm_to_moveit_busy_with_gripper_postgrasp8
  src/selected_block_mm_to_moveit_busy_with_gripper_postgrasp8.cpp
)

ament_target_dependencies(selected_block_mm_to_moveit_busy_with_gripper_postgrasp8
  rclcpp
  geometry_msgs
  six_servo_arm_msgs
  moveit_ros_planning_interface
)

install(TARGETS
  selected_block_mm_to_moveit_busy_with_gripper_postgrasp8
  DESTINATION lib/${PROJECT_NAME}
)
