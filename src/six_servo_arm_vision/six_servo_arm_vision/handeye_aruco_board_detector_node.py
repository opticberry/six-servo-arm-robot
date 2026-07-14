#!/usr/bin/env python3
"""ROS 2 node: detect an ArUco GridBoard and output T_camera_board."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String


def open_capture(source: str) -> cv2.VideoCapture:
    source_value = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(source_value)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera source: {source}")
    return capture


def load_calibration(calibration_path: str) -> dict:
    if not calibration_path:
        raise RuntimeError("calibration is required to estimate T_camera_board")

    path = Path(calibration_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "camera_matrix": np.array(data["camera_matrix"], dtype=np.float64),
        "dist_coeffs": np.array(data["dist_coeffs"], dtype=np.float64),
    }


def make_aruco_detector(dictionary_name: str):
    dictionary_id = getattr(cv2.aruco, dictionary_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)

    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        return dictionary, detector

    parameters = cv2.aruco.DetectorParameters_create()
    return dictionary, parameters


def detect_markers(frame, detector):
    dictionary, detector_or_params = detector
    if hasattr(cv2.aruco, "ArucoDetector"):
        return detector_or_params.detectMarkers(frame)
    return cv2.aruco.detectMarkers(frame, dictionary, parameters=detector_or_params)


def create_grid_board(
    markers_x: int,
    markers_y: int,
    marker_length_m: float,
    marker_separation_m: float,
    dictionary,
):
    if hasattr(cv2.aruco, "GridBoard_create"):
        return cv2.aruco.GridBoard_create(
            markers_x,
            markers_y,
            marker_length_m,
            marker_separation_m,
            dictionary,
        )

    return cv2.aruco.GridBoard(
        (markers_x, markers_y),
        marker_length_m,
        marker_separation_m,
        dictionary,
    )


def _board_points_from_detected_markers(corners, ids, board):
    if hasattr(board, "matchImagePoints"):
        object_points, image_points = board.matchImagePoints(corners, ids)
        return object_points, image_points

    if not hasattr(board, "getObjPoints") or not hasattr(board, "getIds"):
        raise RuntimeError(
            "This OpenCV ArUco build has neither estimatePoseBoard nor "
            "Board.matchImagePoints/getObjPoints. Please upgrade opencv-contrib-python."
        )

    board_ids = board.getIds().flatten().astype(int).tolist()
    board_obj_points = board.getObjPoints()
    detected_ids = ids.flatten().astype(int).tolist()

    object_points = []
    image_points = []
    for marker_id, marker_corners in zip(detected_ids, corners):
        if marker_id not in board_ids:
            continue
        board_index = board_ids.index(marker_id)
        object_points.append(np.asarray(board_obj_points[board_index], dtype=np.float64))
        image_points.append(np.asarray(marker_corners, dtype=np.float64).reshape(4, 2))

    if not object_points:
        return None, None

    return (
        np.concatenate(object_points, axis=0).reshape(-1, 1, 3),
        np.concatenate(image_points, axis=0).reshape(-1, 1, 2),
    )


def estimate_board_pose(corners, ids, board, camera_matrix, dist_coeffs):
    if hasattr(cv2.aruco, "estimatePoseBoard"):
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        used_markers, rvec, tvec = cv2.aruco.estimatePoseBoard(
            corners,
            ids,
            board,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
        )
        return int(used_markers), rvec, tvec

    object_points, image_points = _board_points_from_detected_markers(corners, ids, board)
    if object_points is None or image_points is None:
        return 0, None, None

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0, None, None

    return int(len(image_points) / 4), rvec, tvec

def transform_from_rvec_tvec(rvec, tvec) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = tvec.reshape(3)
    return transform


def quaternion_from_rotation_matrix(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = (trace + 1.0) ** 0.5 * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diag = np.diagonal(rotation)
        axis = int(np.argmax(diag))
        if axis == 0:
            scale = (1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) ** 0.5 * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif axis == 1:
            scale = (1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) ** 0.5 * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = (1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) ** 0.5 * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
    return float(qx), float(qy), float(qz), float(qw)


def rounded_matrix(matrix: np.ndarray, decimals: int = 6) -> list[list[float]]:
    return np.round(matrix.astype(np.float64), decimals=decimals).tolist()


class HandeyeArucoBoardDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("handeye_aruco_board_detector_node")

        self.declare_parameter("source", "/dev/video0")
        self.declare_parameter("calibration", "")
        self.declare_parameter("dictionary", "DICT_4X4_50")
        self.declare_parameter("markers_x", 3)
        self.declare_parameter("markers_y", 3)
        self.declare_parameter("marker_length_m", 0.020)
        self.declare_parameter("marker_separation_m", 0.005)
        self.declare_parameter("min_markers", 4)
        self.declare_parameter("camera_frame", "camera")
        self.declare_parameter("board_frame", "aruco_board")
        self.declare_parameter("show", False)
        self.declare_parameter("timer_period", 0.03)
        self.declare_parameter("log_period", 1.0)

        self.source = self.get_parameter("source").get_parameter_value().string_value
        self.calibration_path = (
            self.get_parameter("calibration").get_parameter_value().string_value
        )
        self.dictionary_name = (
            self.get_parameter("dictionary").get_parameter_value().string_value
        )
        self.markers_x = self.get_parameter("markers_x").get_parameter_value().integer_value
        self.markers_y = self.get_parameter("markers_y").get_parameter_value().integer_value
        self.marker_length_m = (
            self.get_parameter("marker_length_m").get_parameter_value().double_value
        )
        self.marker_separation_m = (
            self.get_parameter("marker_separation_m").get_parameter_value().double_value
        )
        self.min_markers = (
            self.get_parameter("min_markers").get_parameter_value().integer_value
        )
        self.camera_frame = (
            self.get_parameter("camera_frame").get_parameter_value().string_value
        )
        self.board_frame = (
            self.get_parameter("board_frame").get_parameter_value().string_value
        )
        self.show = self.get_parameter("show").get_parameter_value().bool_value
        self.timer_period = (
            self.get_parameter("timer_period").get_parameter_value().double_value
        )
        self.log_period = self.get_parameter("log_period").get_parameter_value().double_value
        self.last_log_time = 0.0

        self.calibration = load_calibration(self.calibration_path)
        self.capture = open_capture(self.source)
        self.dictionary, self.detector_impl = make_aruco_detector(self.dictionary_name)
        self.detector = (self.dictionary, self.detector_impl)
        self.board = create_grid_board(
            self.markers_x,
            self.markers_y,
            self.marker_length_m,
            self.marker_separation_m,
            self.dictionary,
        )

        self.pose_pub = self.create_publisher(PoseStamped, "/handeye_board_pose", 10)
        self.debug_pub = self.create_publisher(String, "/handeye_board_pose_debug", 10)
        self.timer = self.create_timer(self.timer_period, self._timer_callback)

        self.get_logger().info(
            "Handeye ArUco board detector ready: %dx%d, marker=%.3fm, separation=%.3fm"
            % (
                self.markers_x,
                self.markers_y,
                self.marker_length_m,
                self.marker_separation_m,
            )
        )

    def _publish_pose(self, transform: np.ndarray) -> None:
        rotation = transform[:3, :3]
        qx, qy, qz, qw = quaternion_from_rotation_matrix(rotation)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_frame
        msg.pose.position.x = float(transform[0, 3])
        msg.pose.position.y = float(transform[1, 3])
        msg.pose.position.z = float(transform[2, 3])
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.pose_pub.publish(msg)

    def _should_log_now(self) -> bool:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if now_sec - self.last_log_time >= self.log_period:
            self.last_log_time = now_sec
            return True
        return False

    def _timer_callback(self) -> None:
        ok, frame = self.capture.read()
        if not ok:
            self.get_logger().warning("Camera frame read failed.")
            return

        camera_matrix = self.calibration["camera_matrix"]
        dist_coeffs = self.calibration["dist_coeffs"]
        corners, ids, _ = detect_markers(frame, self.detector)
        detected_marker_count = 0 if ids is None else len(ids)

        board_detected = False
        used_markers = 0
        rvec_list = None
        tvec_list = None
        transform_list = None
        transform = None

        if ids is not None and detected_marker_count >= self.min_markers:
            used_markers, rvec, tvec = estimate_board_pose(
                corners,
                ids,
                self.board,
                camera_matrix,
                dist_coeffs,
            )
            board_detected = used_markers >= self.min_markers
            if board_detected:
                transform = transform_from_rvec_tvec(rvec, tvec)
                rvec_list = np.round(rvec.reshape(3), 6).tolist()
                tvec_list = np.round(tvec.reshape(3), 6).tolist()
                transform_list = rounded_matrix(transform)
                self._publish_pose(transform)

        debug = {
            "board_detected": board_detected,
            "detected_marker_count": detected_marker_count,
            "used_marker_count": used_markers,
            "camera_frame": self.camera_frame,
            "board_frame": self.board_frame,
            "rvec": rvec_list,
            "tvec_m": tvec_list,
            "T_camera_board": transform_list,
        }

        msg = String()
        msg.data = json.dumps(debug, ensure_ascii=True)
        self.debug_pub.publish(msg)

        if self._should_log_now():
            if board_detected:
                self.get_logger().info(
                    "board_detected=true, used_markers=%d, tvec_m=%s, T_camera_board=%s"
                    % (used_markers, tvec_list, transform_list)
                )
            else:
                self.get_logger().info(
                    "board_detected=false, detected_markers=%d, need_at_least=%d"
                    % (detected_marker_count, self.min_markers)
                )

        if self.show:
            annotated = frame.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
            if board_detected and transform is not None:
                cv2.drawFrameAxes(
                    annotated,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    self.marker_length_m * 1.5,
                )
            cv2.imshow("Handeye ArUco Board Detector", annotated)
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
    node = HandeyeArucoBoardDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
