# Six Servo Arm ROS 2 Visualization

This package is a minimal ROS 2 visualization setup for the six-servo arm URDF.

## What it does

- publishes the URDF through `robot_state_publisher`
- opens `joint_state_publisher_gui` so you can drag the six joints
- opens RViz with a ready-made view

## Package contents

- `urdf/six_servo_arm_template.urdf`: starter arm model
- `launch/display.launch.py`: visualization launch file
- `rviz/display.rviz`: RViz configuration
- `URDF_notes.md`: notes explaining the URDF structure

## How to build

Put this folder inside a ROS 2 workspace `src` directory, then build:

```bash
colcon build --packages-select six_servo_arm_description
```

Source the workspace after building:

```bash
source install/setup.bash
```

## How to run

Launch the visualization:

```bash
ros2 launch six_servo_arm_description display.launch.py
```

## Expected result

- RViz opens with the arm model visible
- a joint slider window opens
- moving sliders updates the arm pose in RViz

## Current limitations

- the arm still uses simple box/cylinder placeholder geometry
- dimensions are only approximate until you replace them with measured values
- no MoveIt config yet

## Next step after this works

1. Correct each joint axis if any direction is wrong.
2. Replace approximate link lengths with your real measurements.
3. Replace placeholder geometry with STL meshes exported from SolidWorks.
4. Generate a MoveIt package from the finished URDF.
