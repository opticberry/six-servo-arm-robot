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

        DeclareLaunchArgument("dir_x", default_value="0.0"),
        DeclareLaunchArgument("dir_y", default_value="0.0"),
        DeclareLaunchArgument("dir_z", default_value="-1.0"),

        DeclareLaunchArgument("samples", default_value="4000"),
        DeclareLaunchArgument("top_k", default_value="20"),
        DeclareLaunchArgument("goal_tolerance", default_value="0.02"),

        DeclareLaunchArgument("weight_position", default_value="1.0"),
        DeclareLaunchArgument("weight_direction", default_value="0.08"),
        DeclareLaunchArgument("weight_joint_reg", default_value="0.02"),

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

                    "dir_x": LaunchConfiguration("dir_x"),
                    "dir_y": LaunchConfiguration("dir_y"),
                    "dir_z": LaunchConfiguration("dir_z"),

                    "samples": LaunchConfiguration("samples"),
                    "top_k": LaunchConfiguration("top_k"),
                    "goal_tolerance": LaunchConfiguration("goal_tolerance"),

                    "weight_position": LaunchConfiguration("weight_position"),
                    "weight_direction": LaunchConfiguration("weight_direction"),
                    "weight_joint_reg": LaunchConfiguration("weight_joint_reg"),
                },
            ],
        ),
    ])