#!/usr/bin/env python3
"""ROS 2 node: track left/right gripper ArUco markers on the calibrated plane."""

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


def load_calibration(calibration_path: str) -> dict | None:
    if not calibration_path:
        return None

    path = Path(calibration_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "camera_matrix": np.array(data["camera_matrix"], dtype=np.float32),
        "dist_coeffs": np.array(data["dist_coeffs"], dtype=np.float32),
    }


def build_undistort_maps(calibration: dict, frame_width: int, frame_height: int):
    size = (frame_width, frame_height)
    camera_matrix = calibration["camera_matrix"]
    dist_coeffs = calibration["dist_coeffs"]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, size, 1, size
    )
    return cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
        size,
        cv2.CV_16SC2,
    )


def load_plane_mapping(mapping_path: str) -> dict | None:
    if not mapping_path:
        return None

    path = Path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(f"Plane mapping file not found: {mapping_path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "homography_image_to_world": np.array(
            data["homography_image_to_world"], dtype=np.float32
        ),
    }


def pixel_to_world_xy(center_xy: list[float], plane_mapping: dict) -> list[float]:
    point = np.array([[[center_xy[0], center_xy[1]]]], dtype=np.float32)
    world = cv2.perspectiveTransform(point, plane_mapping["homography_image_to_world"])
    x_mm, y_mm = world[0, 0].tolist()
    return [round(float(x_mm), 2), round(float(y_mm), 2)]


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


def marker_center(corners) -> list[float]:
    pts = corners.reshape(-1, 2)
    center = pts.mean(axis=0)
    return [round(float(center[0]), 2), round(float(center[1]), 2)]


class ArucoGripperTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("aruco_gripper_tracker_node")

        self.declare_parameter("source", "0")
        self.declare_parameter("calibration", "")
        self.declare_parameter("plane_mapping", "")
        self.declare_parameter("left_marker_id", 1)
        self.declare_parameter("right_marker_id", 2)
        self.declare_parameter("dictionary", "DICT_4X4_50")
        self.declare_parameter("camera_frame", "camera")
        self.declare_parameter("show", False)
        self.declare_parameter("timer_period", 0.03)

        self.source = self.get_parameter("source").get_parameter_value().string_value
        self.calibration_path = (
            self.get_parameter("calibration").get_parameter_value().string_value
        )
        self.plane_mapping_path = (
            self.get_parameter("plane_mapping").get_parameter_value().string_value
        )
        self.left_marker_id = (
            self.get_parameter("left_marker_id").get_parameter_value().integer_value
        )
        self.right_marker_id = (
            self.get_parameter("right_marker_id").get_parameter_value().integer_value
        )
        self.dictionary_name = (
            self.get_parameter("dictionary").get_parameter_value().string_value
        )
        self.camera_frame = (
            self.get_parameter("camera_frame").get_parameter_value().string_value
        )
        self.show = self.get_parameter("show").get_parameter_value().bool_value
        self.timer_period = (
            self.get_parameter("timer_period").get_parameter_value().double_value
        )

        self.calibration = load_calibration(self.calibration_path)
        self.plane_mapping = load_plane_mapping(self.plane_mapping_path)
        if self.plane_mapping is None:
            raise RuntimeError("plane_mapping is required for gripper mm output")

        self.undistort_maps = None
        self.capture = open_capture(self.source)
        self.detector = make_aruco_detector(self.dictionary_name)

        self.left_pub = self.create_publisher(PointStamped, "/gripper_left_marker_mm", 10)
        self.right_pub = self.create_publisher(
            PointStamped, "/gripper_right_marker_mm", 10
        )
        self.center_pub = self.create_publisher(PointStamped, "/gripper_center_mm", 10)
        self.debug_pub = self.create_publisher(String, "/gripper_aruco_debug", 10)
        self.timer = self.create_timer(self.timer_period, self._timer_callback)

        self.get_logger().info(
            "Aruco gripper tracker ready: left_id=%d, right_id=%d, dictionary=%s"
            % (self.left_marker_id, self.right_marker_id, self.dictionary_name)
        )

    def _publish_point(self, publisher, x_mm: float, y_mm: float) -> None:
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_frame
        msg.point.x = float(x_mm)
        msg.point.y = float(y_mm)
        msg.point.z = 0.0
        publisher.publish(msg)

    def _timer_callback(self) -> None:
        ok, frame = self.capture.read()
        if not ok:
            self.get_logger().warning("Camera frame read failed.")
            return

        if self.calibration is not None:
            frame_height, frame_width = frame.shape[:2]
            if self.undistort_maps is None:
                self.undistort_maps = build_undistort_maps(
                    self.calibration, frame_width, frame_height
                )
            frame = cv2.remap(
                frame, self.undistort_maps[0], self.undistort_maps[1], cv2.INTER_LINEAR
            )

        corners_list, ids, _ = detect_markers(frame, self.detector)
        found: dict[int, dict] = {}
        if ids is not None:
            for marker_id, corners in zip(ids.flatten().tolist(), corners_list):
                center_px = marker_center(corners)
                center_mm = pixel_to_world_xy(center_px, self.plane_mapping)
                found[int(marker_id)] = {
                    "center_px": center_px,
                    "center_mm": center_mm,
                    "corners": corners,
                }

        left = found.get(self.left_marker_id)
        right = found.get(self.right_marker_id)
        debug: dict = {
            "left_marker_id": self.left_marker_id,
            "right_marker_id": self.right_marker_id,
            "markers": {
                str(marker_id): {
                    "center_px": info["center_px"],
                    "center_mm": info["center_mm"],
                }
                for marker_id, info in found.items()
            },
        }

        if left is not None:
            self._publish_point(
                self.left_pub, left["center_mm"][0], left["center_mm"][1]
            )
        if right is not None:
            self._publish_point(
                self.right_pub, right["center_mm"][0], right["center_mm"][1]
            )

        if left is not None and right is not None:
            lx, ly = left["center_mm"]
            rx, ry = right["center_mm"]
            cx = round((lx + rx) * 0.5, 2)
            cy = round((ly + ry) * 0.5, 2)
            yaw_rad = math.atan2(ry - ly, rx - lx)
            debug["gripper_center_mm"] = [cx, cy]
            debug["gripper_yaw_rad"] = round(yaw_rad, 5)
            debug["gripper_yaw_deg"] = round(math.degrees(yaw_rad), 2)
            self._publish_point(self.center_pub, cx, cy)
            self.get_logger().info(
                "gripper center=(%.1f, %.1f) mm yaw=%.1f deg"
                % (cx, cy, math.degrees(yaw_rad))
            )

        msg = String()
        msg.data = json.dumps(debug, ensure_ascii=True)
        self.debug_pub.publish(msg)

        if self.show:
            annotated = frame.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(annotated, corners_list, ids)
            if left is not None and right is not None:
                lpx = [int(v) for v in left["center_px"]]
                rpx = [int(v) for v in right["center_px"]]
                cv2.line(annotated, tuple(lpx), tuple(rpx), (0, 255, 0), 2)
            cv2.imshow("Aruco Gripper Tracker", annotated)
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
    node = ArucoGripperTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
