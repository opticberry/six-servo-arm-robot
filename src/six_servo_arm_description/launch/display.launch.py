from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    urdf_path = Path(LaunchConfiguration("urdf_path").perform(context))
    rviz_config = LaunchConfiguration("rviz_config").perform(context)
    use_joint_state_gui = LaunchConfiguration("use_joint_state_gui")
    robot_description = urdf_path.read_text(encoding="utf-8")

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[
                {
                    "robot_description": robot_description,
                }
            ],
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            name="joint_state_publisher_gui",
            output="screen",
            condition=IfCondition(use_joint_state_gui),
            parameters=[
                {
                    "robot_description": robot_description,
                }
            ],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_config],
        ),
    ]


def generate_launch_description():
    package_share = Path(get_package_share_directory("six_servo_arm_description"))
    default_urdf = package_share / "urdf" / "six_servo_arm_template.urdf"
    default_rviz = package_share / "rviz" / "display.rviz"

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_joint_state_gui",
                default_value="true",
                description="Open the joint_state_publisher GUI for interactive joint sliders.",
            ),
            DeclareLaunchArgument(
                "urdf_path",
                default_value=str(default_urdf),
                description="Absolute path to the URDF file to visualize.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=str(default_rviz),
                description="Absolute path to the RViz config file.",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
