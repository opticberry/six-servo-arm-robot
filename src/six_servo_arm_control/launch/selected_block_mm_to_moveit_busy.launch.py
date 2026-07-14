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
        DeclareLaunchArgument("a4_origin_x_m", default_value="0.10"),
        DeclareLaunchArgument("a4_origin_y_m", default_value="0.00"),
        DeclareLaunchArgument("a4_x_sign", default_value="1.0"),
        DeclareLaunchArgument("a4_y_sign", default_value="1.0"),
        DeclareLaunchArgument("table_z_m", default_value="0.02"),
        DeclareLaunchArgument("sample_window_size", default_value="6"),
        DeclareLaunchArgument("sample_spread_threshold_m", default_value="0.010"),
        DeclareLaunchArgument("auto_execute", default_value="true"),

        Node(
            package="six_servo_arm_control",
            executable="selected_block_mm_to_moveit_busy",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                {
                    "a4_origin_x_m": LaunchConfiguration("a4_origin_x_m"),
                    "a4_origin_y_m": LaunchConfiguration("a4_origin_y_m"),
                    "a4_x_sign": LaunchConfiguration("a4_x_sign"),
                    "a4_y_sign": LaunchConfiguration("a4_y_sign"),
                    "table_z_m": LaunchConfiguration("table_z_m"),
                    "sample_window_size": LaunchConfiguration("sample_window_size"),
                    "sample_spread_threshold_m": LaunchConfiguration("sample_spread_threshold_m"),
                    "auto_execute": LaunchConfiguration("auto_execute"),
                },
            ],
        ),
    ])