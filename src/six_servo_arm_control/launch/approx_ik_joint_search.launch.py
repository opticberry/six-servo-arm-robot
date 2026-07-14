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
        DeclareLaunchArgument("x", default_value="0.00"),
        DeclareLaunchArgument("y", default_value="0.00"),
        DeclareLaunchArgument("z", default_value="0.20"),
        DeclareLaunchArgument("samples", default_value="4000"),
        DeclareLaunchArgument("goal_tolerance", default_value="0.02"),

        Node(
            package="six_servo_arm_control",
            executable="approx_ik_joint_search",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                {
                    "x": LaunchConfiguration("x"),
                    "y": LaunchConfiguration("y"),
                    "z": LaunchConfiguration("z"),
                    "samples": LaunchConfiguration("samples"),
                    "goal_tolerance": LaunchConfiguration("goal_tolerance"),
                },
            ],
        ),
    ])