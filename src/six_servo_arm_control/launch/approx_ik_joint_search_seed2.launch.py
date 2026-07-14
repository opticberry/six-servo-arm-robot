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

        DeclareLaunchArgument("dir_x", default_value="1.0"),
        DeclareLaunchArgument("dir_y", default_value="0.0"),
        DeclareLaunchArgument("dir_z", default_value="-1.0"),

        DeclareLaunchArgument("side_x", default_value="0.0"),
        DeclareLaunchArgument("side_y", default_value="1.0"),
        DeclareLaunchArgument("side_z", default_value="0.0"),

        # Tune these five values to your real pre-grasp posture.
        DeclareLaunchArgument("seed_j0", default_value="0.0"),
        DeclareLaunchArgument("seed_j1", default_value="0.9"),
        DeclareLaunchArgument("seed_j2", default_value="1.57"),
        DeclareLaunchArgument("seed_j3", default_value="-1.57"),
        DeclareLaunchArgument("seed_j4", default_value="0.0"),

        DeclareLaunchArgument("samples", default_value="8000"),
        DeclareLaunchArgument("top_k", default_value="30"),
        DeclareLaunchArgument("accept_position_error", default_value="0.05"),

        DeclareLaunchArgument("weight_position", default_value="1.0"),
        DeclareLaunchArgument("weight_direction", default_value="0.10"),
        DeclareLaunchArgument("weight_side", default_value="0.15"),
        DeclareLaunchArgument("weight_joint_reg", default_value="0.02"),

        DeclareLaunchArgument("side_alignment_min", default_value="0.60"),

        DeclareLaunchArgument("local_ratio", default_value="0.80"),
        DeclareLaunchArgument("local_span", default_value="0.60"),

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

                    "side_x": LaunchConfiguration("side_x"),
                    "side_y": LaunchConfiguration("side_y"),
                    "side_z": LaunchConfiguration("side_z"),

                    "seed_j0": LaunchConfiguration("seed_j0"),
                    "seed_j1": LaunchConfiguration("seed_j1"),
                    "seed_j2": LaunchConfiguration("seed_j2"),
                    "seed_j3": LaunchConfiguration("seed_j3"),
                    "seed_j4": LaunchConfiguration("seed_j4"),

                    "samples": LaunchConfiguration("samples"),
                    "top_k": LaunchConfiguration("top_k"),
                    "accept_position_error": LaunchConfiguration("accept_position_error"),

                    "weight_position": LaunchConfiguration("weight_position"),
                    "weight_direction": LaunchConfiguration("weight_direction"),
                    "weight_side": LaunchConfiguration("weight_side"),
                    "weight_joint_reg": LaunchConfiguration("weight_joint_reg"),

                    "side_alignment_min": LaunchConfiguration("side_alignment_min"),

                    "local_ratio": LaunchConfiguration("local_ratio"),
                    "local_span": LaunchConfiguration("local_span"),
                },
            ],
        ),
    ])
