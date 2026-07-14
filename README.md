# Vision-Guided Six-Servo Robotic Arm

This is a ROS 2 + MoveIt 2 based robotic arm project for real-world block grasping. The system combines camera perception, eye-to-hand calibration, motion planning, STM32 bus-servo control, and PPO-based local visual servo correction.

The main goal of this project is to build a complete robotic grasping pipeline on a real six-servo arm: the camera detects a target block, converts its position into the robot base frame, MoveIt plans the arm motion, and the STM32 controller executes the final joint commands.

## Features

- ROS 2 Humble robotic arm control pipeline.
- MoveIt 2 based motion planning and post-grasp motion.
- YOLO/RKNN object detection on RK3588.
- Eye-to-hand calibration with ArUco board detection.
- Target coordinate conversion from camera frame to `base_link`.
- ArUco marker based gripper-center error estimation.
- PPO local visual servo correction for fine grasp adjustment.
- STM32 + FreeRTOS bus-servo execution layer.

## Repository Structure

```text
six_servo_arm_bridge/        ROS 2 bridge layer for communication.
six_servo_arm_control/       Main grasping, MoveIt control, and servo command nodes.
six_servo_arm_description/   URDF model, RViz config, and robot description.
six_servo_arm_moveit_config/ MoveIt configuration package.
six_servo_arm_msgs/          Custom ROS 2 messages.
six_servo_arm_rl/            PPO local visual-servo training and deployment.
six_servo_arm_vision/        Camera detection, hand-eye calibration, and visual servo error.
```

## System Overview

```text
Camera
  -> YOLO/RKNN detection
  -> eye-to-hand coordinate transform
  -> MoveIt 2 planning
  -> PPO local correction
  -> /arm_joint_cmd
  -> STM32 bus-servo controller
```

## Current Status

The project has been tested on real hardware. The basic grasping pipeline, hand-eye calibration, target coordinate conversion, gripper marker error calculation, and PPO local correction have all been integrated into the system.

This repository is mainly used as an engineering record and project showcase. Some hardware-specific parameters, calibration files, and model files may need to be adjusted before running on a different robotic arm.

## Tech Stack

```text
ROS 2 Humble
MoveIt 2
OpenCV / ArUco
YOLO / RKNN
PPO
STM32 / FreeRTOS
micro-ROS
```

