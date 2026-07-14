#!/usr/bin/env python3
"""Track gripper ArUco markers in 3D and compute error to selected block base point."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import String


def open_capture(source: str) -> cv2.VideoCapture:
    source_value = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(source_value)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    return capture


def load_calibration(calibration_path: str) -> dict:
    if not calibration_path:
        raise RuntimeError("calibration is required for marker 3D pose estimation")

    path = Path(calibration_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "camera_matrix": np.array(data["camera_matrix"], dtype=np.float64),
        "dist_coeffs": np.array(data["dist_coeffs"], dtype=np.float64),
    }


def load_eye_to_hand(eye_to_hand_path: str) -> dict:
    if not eye_to_hand_path:
        raise RuntimeError("eye_to_hand result is required for base coordinate output")

    path = Path(eye_to_hand_path)
    if not path.exists():
        raise FileNotFoundError(f"Eye-to-hand result file not found: {eye_to_hand_path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "T_base_camera": np.array(data["T_base_camera"], dtype=np.float64),
    }


def make_aruco_detector(dictionary_name: str):
    dictionary_id = getattr(cv2.aruco, dictionary_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        return dictionary, cv2.aruco.ArucoDetector(dictionary, parameters)

    parameters = cv2.aruco.DetectorParameters_create()
    return dictionary, parameters


def detect_markers(frame, detector):
    dictionary, detector_or_params = detector
    if hasattr(cv2.aruco, "ArucoDetector"):
        return detector_or_params.detectMarkers(frame)
    return cv2.aruco.detectMarkers(frame, dictionary, parameters=detector_or_params)


def marker_object_points(marker_length_m: float) -> np.ndarray:
    half = marker_length_m * 0.5
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def estimate_marker_center_camera(
    corners,
    marker_length_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray | None:
    object_points = marker_object_points(marker_length_m)
    image_points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    ok, _rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    return tvec.reshape(3)


def transform_point(transform: np.ndarray, point_xyz: np.ndarray) -> np.ndarray:
    point_h = np.array([point_xyz[0], point_xyz[1], point_xyz[2], 1.0], dtype=np.float64)
    return (transform @ point_h)[:3]


def publish_point(node: Node, publisher, frame_id: str, point_xyz: np.ndarray) -> None:
    msg = PointStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = frame_id
    msg.point.x = float(point_xyz[0])
    msg.point.y = float(point_xyz[1])
    msg.point.z = float(point_xyz[2])
    publisher.publish(msg)


class ArucoGripperBaseErrorNode(Node):
    def __init__(self) -> None:
        super().__init__("aruco_gripper_base_error_node")

        self.declare_parameter("source", "/dev/video0")
        self.declare_parameter("calibration", "/home/orangepi/test/calibration/camera_calibration.json")
        self.declare_parameter("eye_to_hand", "/home/orangepi/test/calibration/eye_to_hand_result.json")
        self.declare_parameter("selected_block_topic", "/selected_block_base")
        self.declare_parameter("left_marker_id", 1)
        self.declare_parameter("right_marker_id", 2)
        self.declare_parameter("dictionary", "DICT_4X4_50")
        self.declare_parameter("marker_length_m", 0.020)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("show", False)
        self.declare_parameter("timer_period", 0.03)

        self.source = self.get_parameter("source").get_parameter_value().string_value
        self.calibration_path = self.get_parameter("calibration").get_parameter_value().string_value
        self.eye_to_hand_path = self.get_parameter("eye_to_hand").get_parameter_value().string_value
        self.selected_block_topic = (
            self.get_parameter("selected_block_topic").get_parameter_value().string_value
        )
        self.left_marker_id = self.get_parameter("left_marker_id").get_parameter_value().integer_value
        self.right_marker_id = self.get_parameter("right_marker_id").get_parameter_value().integer_value
        self.dictionary_name = self.get_parameter("dictionary").get_parameter_value().string_value
        self.marker_length_m = self.get_parameter("marker_length_m").get_parameter_value().double_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.show = self.get_parameter("show").get_parameter_value().bool_value
        self.timer_period = self.get_parameter("timer_period").get_parameter_value().double_value

        self.calibration = load_calibration(self.calibration_path)
        self.eye_to_hand = load_eye_to_hand(self.eye_to_hand_path)
        self.capture = open_capture(self.source)
        self.detector = make_aruco_detector(self.dictionary_name)
        self.latest_selected_base: np.ndarray | None = None

        self.left_pub = self.create_publisher(PointStamped, "/gripper_left_marker_base", 10)
        self.right_pub = self.create_publisher(PointStamped, "/gripper_right_marker_base", 10)
        self.center_pub = self.create_publisher(PointStamped, "/gripper_center_base", 10)
        self.error_pub = self.create_publisher(PointStamped, "/visual_servo_error_base", 10)
        self.debug_pub = self.create_publisher(String, "/aruco_gripper_base_error_debug", 10)
        self.create_subscription(
            PointStamped,
            self.selected_block_topic,
            self._selected_block_callback,
            10,
        )
        self.timer = self.create_timer(self.timer_period, self._timer_callback)

        self.get_logger().info(
            "Aruco gripper base error node ready: left_id=%d, right_id=%d, marker=%.3fm"
            % (self.left_marker_id, self.right_marker_id, self.marker_length_m)
        )

    def _selected_block_callback(self, msg: PointStamped) -> None:
        self.latest_selected_base = np.array(
            [msg.point.x, msg.point.y, msg.point.z],
            dtype=np.float64,
        )

    def _timer_callback(self) -> None:
        ok, frame = self.capture.read()
        if not ok:
            self.get_logger().warning("Camera frame read failed.")
            return

        camera_matrix = self.calibration["camera_matrix"]
        dist_coeffs = self.calibration["dist_coeffs"]
        t_base_camera = self.eye_to_hand["T_base_camera"]
        corners_list, ids, _ = detect_markers(frame, self.detector)

        marker_base: dict[int, np.ndarray] = {}
        if ids is not None:
            for marker_id, corners in zip(ids.flatten().tolist(), corners_list):
                if marker_id not in (self.left_marker_id, self.right_marker_id):
                    continue
                center_camera = estimate_marker_center_camera(
                    corners,
                    self.marker_length_m,
                    camera_matrix,
                    dist_coeffs,
                )
                if center_camera is None:
                    continue
                marker_base[int(marker_id)] = transform_point(t_base_camera, center_camera)

        left = marker_base.get(self.left_marker_id)
        right = marker_base.get(self.right_marker_id)
        gripper_center = None
        error = None

        if left is not None:
            publish_point(self, self.left_pub, self.base_frame, left)
        if right is not None:
            publish_point(self, self.right_pub, self.base_frame, right)
        if left is not None and right is not None:
            gripper_center = (left + right) * 0.5
            publish_point(self, self.center_pub, self.base_frame, gripper_center)

        if gripper_center is not None and self.latest_selected_base is not None:
            error = self.latest_selected_base - gripper_center
            publish_point(self, self.error_pub, self.base_frame, error)
            self.get_logger().info(
                "error=(%.1f, %.1f, %.1f)mm norm=%.1fmm"
                % (
                    error[0] * 1000.0,
                    error[1] * 1000.0,
                    error[2] * 1000.0,
                    math.sqrt(float(error @ error)) * 1000.0,
                )
            )

        debug = {
            "left_marker_id": self.left_marker_id,
            "right_marker_id": self.right_marker_id,
            "left_base_m": None if left is None else np.round(left, 6).tolist(),
            "right_base_m": None if right is None else np.round(right, 6).tolist(),
            "gripper_center_base_m": None
            if gripper_center is None
            else np.round(gripper_center, 6).tolist(),
            "selected_block_base_m": None
            if self.latest_selected_base is None
            else np.round(self.latest_selected_base, 6).tolist(),
            "error_base_m": None if error is None else np.round(error, 6).tolist(),
            "error_norm_m": None
            if error is None
            else round(math.sqrt(float(error @ error)), 6),
        }
        msg = String()
        msg.data = json.dumps(debug, ensure_ascii=True)
        self.debug_pub.publish(msg)

        if self.show:
            annotated = frame.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(annotated, corners_list, ids)
            cv2.imshow("Aruco Gripper Base Error", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                rclpy.shutdown()

    def destroy_node(self) -> bool:
        if hasattr(self, "capture") and self.capture is not None:
            self.capture.release()
        if self.show:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = ArucoGripperBaseErrorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
