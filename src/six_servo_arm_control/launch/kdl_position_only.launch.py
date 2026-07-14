from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("x", default_value="0.10"),
        DeclareLaunchArgument("y", default_value="0.00"),
        DeclareLaunchArgument("z", default_value="0.20"),

        DeclareLaunchArgument("pregrasp_j0", default_value="0.0"),
        DeclareLaunchArgument("pregrasp_j1", default_value="0.9"),
        DeclareLaunchArgument("pregrasp_j2", default_value="1.57"),
        DeclareLaunchArgument("pregrasp_j3", default_value="-1.57"),
        DeclareLaunchArgument("pregrasp_j4", default_value="0.0"),

        DeclareLaunchArgument("goal_position_tolerance", default_value="0.005"),
        DeclareLaunchArgument("planning_time", default_value="10.0"),
        DeclareLaunchArgument("num_planning_attempts", default_value="20"),

        DeclareLaunchArgument("pregrasp_time_ms", default_value="2000"),
        DeclareLaunchArgument("target_time_ms", default_value="2500"),

        Node(
            package="six_servo_arm_control",
            executable="kdl_position_only",
            name="kdl_position_only",
            output="screen",
            parameters=[{
                "x": LaunchConfiguration("x"),
                "y": LaunchConfiguration("y"),
                "z": LaunchConfiguration("z"),

                "pregrasp_j0": LaunchConfiguration("pregrasp_j0"),
                "pregrasp_j1": LaunchConfiguration("pregrasp_j1"),
                "pregrasp_j2": LaunchConfiguration("pregrasp_j2"),
                "pregrasp_j3": LaunchConfiguration("pregrasp_j3"),
                "pregrasp_j4": LaunchConfiguration("pregrasp_j4"),

                "goal_position_tolerance": LaunchConfiguration("goal_position_tolerance"),
                "planning_time": LaunchConfiguration("planning_time"),
                "num_planning_attempts": LaunchConfiguration("num_planning_attempts"),

                "pregrasp_time_ms": LaunchConfiguration("pregrasp_time_ms"),
                "target_time_ms": LaunchConfiguration("target_time_ms"),
            }],
        )
    ])