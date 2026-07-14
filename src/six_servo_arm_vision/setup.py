from setuptools import setup

package_name = 'six_servo_arm_vision'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='orangepi',
    maintainer_email='orangepi@example.com',
    description='Vision nodes for six_servo_arm, including YOLO detection and plane mapping.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detect_camera_ros2_node = six_servo_arm_vision.detect_camera_ros2_node:main',
            'detect_camera_ros2_node2 = six_servo_arm_vision.detect_camera_ros2_node2:main',
            'detect_camera_ros2_node3 = six_servo_arm_vision.detect_camera_ros2_node3:main',
            'create_plane_mapping = six_servo_arm_vision.create_plane_mapping:main',
            'aruco_gripper_tracker_node = six_servo_arm_vision.aruco_gripper_tracker_node:main',
            'aruco_gripper_base_error_node = six_servo_arm_vision.aruco_gripper_base_error_node:main',
            'handeye_aruco_board_detector_node = six_servo_arm_vision.handeye_aruco_board_detector_node:main',
            'handeye_sample_collector_node = six_servo_arm_vision.handeye_sample_collector_node:main',
            'solve_eye_to_hand_calibration = six_servo_arm_vision.solve_eye_to_hand_calibration:main',
            'yolo_aruco_visual_servo_node = six_servo_arm_vision.yolo_aruco_visual_servo_node:main',
        ],
    },
)
