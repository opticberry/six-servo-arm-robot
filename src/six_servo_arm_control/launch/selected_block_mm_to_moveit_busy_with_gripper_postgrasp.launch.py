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
        DeclareLaunchArgument("auto_execute", default_value="true"),

        DeclareLaunchArgument("a4_origin_x_m", default_value="0.10"),
        DeclareLaunchArgument("a4_origin_y_m", default_value="0.00"),
        DeclareLaunchArgument("a4_x_sign", default_value="1.0"),
        DeclareLaunchArgument("a4_y_sign", default_value="1.0"),
        DeclareLaunchArgument("table_z_m", default_value="0.02"),

        DeclareLaunchArgument("sample_window_size", default_value="6"),
        DeclareLaunchArgument("sample_spread_threshold_m", default_value="0.010"),

        DeclareLaunchArgument("move_group_name", default_value="arm"),
        DeclareLaunchArgument("gripper_group_name", default_value="gripper"),
        DeclareLaunchArgument("reference_frame", default_value="base_link"),

        DeclareLaunchArgument("pregrasp_j0", default_value="0.0"),
        DeclareLaunchArgument("pregrasp_j1", default_value="0.9"),
        DeclareLaunchArgument("pregrasp_j2", default_value="1.57"),
        DeclareLaunchArgument("pregrasp_j3", default_value="-1.57"),
        DeclareLaunchArgument("pregrasp_j4", default_value="0.0"),

        DeclareLaunchArgument("postgrasp_j0", default_value="1.57"),
        DeclareLaunchArgument("postgrasp_j1", default_value="0.9"),
        DeclareLaunchArgument("postgrasp_j2", default_value="1.57"),
        DeclareLaunchArgument("postgrasp_j3", default_value="-1.57"),
        DeclareLaunchArgument("postgrasp_j4", default_value="0.0"),

        DeclareLaunchArgument("gripper_close", default_value="0.5"),
        DeclareLaunchArgument("gripper_open", default_value="0.0"),
        DeclareLaunchArgument("gripper_settle_ms", default_value="500"),

        DeclareLaunchArgument("pregrasp_time_ms", default_value="2000"),
        DeclareLaunchArgument("target_time_ms", default_value="2500"),
        DeclareLaunchArgument("postgrasp_time_ms", default_value="2500"),

        DeclareLaunchArgument("ik_max_iters", default_value="120"),
        DeclareLaunchArgument("ik_position_tolerance", default_value="0.005"),
        DeclareLaunchArgument("ik_damping", default_value="0.05"),
        DeclareLaunchArgument("ik_step_scale", default_value="0.5"),

        Node(
            package="six_servo_arm_control",
            executable="selected_block_mm_to_moveit_busy_with_gripper_postgrasp",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                {
                    "auto_execute": LaunchConfiguration("auto_execute"),

                    "a4_origin_x_m": LaunchConfiguration("a4_origin_x_m"),
                    "a4_origin_y_m": LaunchConfiguration("a4_origin_y_m"),
                    "a4_x_sign": LaunchConfiguration("a4_x_sign"),
                    "a4_y_sign": LaunchConfiguration("a4_y_sign"),
                    "table_z_m": LaunchConfiguration("table_z_m"),

                    "sample_window_size": LaunchConfiguration("sample_window_size"),
                    "sample_spread_threshold_m": LaunchConfiguration("sample_spread_threshold_m"),

                    "move_group_name": LaunchConfiguration("move_group_name"),
                    "gripper_group_name": LaunchConfiguration("gripper_group_name"),
                    "reference_frame": LaunchConfiguration("reference_frame"),

                    "pregrasp_j0": LaunchConfiguration("pregrasp_j0"),
                    "pregrasp_j1": LaunchConfiguration("pregrasp_j1"),
                    "pregrasp_j2": LaunchConfiguration("pregrasp_j2"),
                    "pregrasp_j3": LaunchConfiguration("pregrasp_j3"),
                    "pregrasp_j4": LaunchConfiguration("pregrasp_j4"),

                    "postgrasp_j0": LaunchConfiguration("postgrasp_j0"),
                    "postgrasp_j1": LaunchConfiguration("postgrasp_j1"),
                    "postgrasp_j2": LaunchConfiguration("postgrasp_j2"),
                    "postgrasp_j3": LaunchConfiguration("postgrasp_j3"),
                    "postgrasp_j4": LaunchConfiguration("postgrasp_j4"),

                    "gripper_close": LaunchConfiguration("gripper_close"),
                    "gripper_open": LaunchConfiguration("gripper_open"),
                    "gripper_settle_ms": LaunchConfiguration("gripper_settle_ms"),

                    "pregrasp_time_ms": LaunchConfiguration("pregrasp_time_ms"),
                    "target_time_ms": LaunchConfiguration("target_time_ms"),
                    "postgrasp_time_ms": LaunchConfiguration("postgrasp_time_ms"),

                    "ik_max_iters": LaunchConfiguration("ik_max_iters"),
                    "ik_position_tolerance": LaunchConfiguration("ik_position_tolerance"),
                    "ik_damping": LaunchConfiguration("ik_damping"),
                    "ik_step_scale": LaunchConfiguration("ik_step_scale"),
                },
            ],
        ),
    ])