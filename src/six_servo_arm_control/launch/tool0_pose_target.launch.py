from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("six_servo_arm", package_name="six_servo_arm_moveit_config")
        .to_moveit_configs()
    )

    return LaunchDescription([
        DeclareLaunchArgument("x", default_value="0.0"),
        DeclareLaunchArgument("y", default_value="0.0"),
        DeclareLaunchArgument("z", default_value="0.2"),
        DeclareLaunchArgument("qx", default_value="0.0"),
        DeclareLaunchArgument("qy", default_value="0.0"),
        DeclareLaunchArgument("qz", default_value="0.0"),
        DeclareLaunchArgument("qw", default_value="1.0"),

        Node(
            package="six_servo_arm_control",
            executable="tool0_pose_target",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                {
                    "x": LaunchConfiguration("x"),
                    "y": LaunchConfiguration("y"),
                    "z": LaunchConfiguration("z"),
                    "qx": LaunchConfiguration("qx"),
                    "qy": LaunchConfiguration("qy"),
                    "qz": LaunchConfiguration("qz"),
                    "qw": LaunchConfiguration("qw"),
                },
            ],
        )
    ])