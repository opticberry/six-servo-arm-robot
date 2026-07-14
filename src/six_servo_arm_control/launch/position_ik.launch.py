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

        DeclareLaunchArgument("ik_max_iters", default_value="120"),
        DeclareLaunchArgument("ik_position_tolerance", default_value="0.005"),
        DeclareLaunchArgument("ik_damping", default_value="0.05"),
        DeclareLaunchArgument("ik_step_scale", default_value="0.5"),

        DeclareLaunchArgument("pregrasp_time_ms", default_value="2000"),
        DeclareLaunchArgument("target_time_ms", default_value="2500"),

        Node(
            package="six_servo_arm_control",
            executable="position_ik",
            name="position_ik",
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

                "ik_max_iters": LaunchConfiguration("ik_max_iters"),
                "ik_position_tolerance": LaunchConfiguration("ik_position_tolerance"),
                "ik_damping": LaunchConfiguration("ik_damping"),
                "ik_step_scale": LaunchConfiguration("ik_step_scale"),

                "pregrasp_time_ms": LaunchConfiguration("pregrasp_time_ms"),
                "target_time_ms": LaunchConfiguration("target_time_ms"),
            }],
        )
    ])