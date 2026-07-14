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
        DeclareLaunchArgument("j0", default_value="0.0"),
        DeclareLaunchArgument("j1", default_value="0.0"),
        DeclareLaunchArgument("j2", default_value="0.0"),
        DeclareLaunchArgument("j3", default_value="0.0"),
        DeclareLaunchArgument("j4", default_value="0.0"),

        Node(
            package="six_servo_arm_control",
            executable="joint_target",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                {
                    "j0": LaunchConfiguration("j0"),
                    "j1": LaunchConfiguration("j1"),
                    "j2": LaunchConfiguration("j2"),
                    "j3": LaunchConfiguration("j3"),
                    "j4": LaunchConfiguration("j4"),
                },
            ],
        )
    ])