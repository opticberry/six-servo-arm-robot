#!/usr/bin/env python3
"""ROS 2 node: run YOLOv8 RKNN and gripper ArUco tracking from one camera stream."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult
from rclpy.node import Node
from std_msgs.msg import String
from rknnlite.api import RKNNLite


DEFAULT_BLOCK_CLASSES = ["red_block", "green_block", "blue_block"]


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
        "image_width": int(data["image_size"]["width"]),
        "image_height": int(data["image_size"]["height"]),
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
        "plane_width_mm": float(data["plane"]["width_mm"]),
        "plane_height_mm": float(data["plane"]["height_mm"]),
    }


def pixel_to_world_xy(
    center_xy: list[float], plane_mapping: dict | None
) -> list[float] | None:
    if plane_mapping is None:
        return None

    point = np.array([[[center_xy[0], center_xy[1]]]], dtype=np.float32)
    world = cv2.perspectiveTransform(point, plane_mapping["homography_image_to_world"])
    x_mm, y_mm = world[0, 0].tolist()
    return [round(float(x_mm), 2), round(float(y_mm), 2)]


def letterbox(image: np.ndarray, new_size: int):
    height, width = image.shape[:2]
    scale = min(new_size / width, new_size / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )

    pad_width = new_size - resized_width
    pad_height = new_size - resized_height
    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, scale, left, top


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)
        order = order[1:][iou <= iou_thres]

    return keep


def parse_yolov8_rknn_output(
    output: np.ndarray,
    classes: list[str],
    conf_thres: float,
    iou_thres: float,
    class_channels: list[int] | None,
    scale: float,
    pad_x: int,
    pad_y: int,
    frame_width: int,
    frame_height: int,
    plane_mapping: dict | None,
    allowed_classes: set[str] | None,
    top1_per_class: bool,
) -> list[dict]:
    pred = np.squeeze(output)
    if pred.ndim != 2:
        raise RuntimeError(f"Unexpected RKNN output shape after squeeze: {pred.shape}")
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    boxes_xywh = pred[:, :4].astype(np.float32)
    if class_channels is None:
        if pred.shape[1] == 4 + len(classes):
            class_channels = list(range(4, 4 + len(classes)))
        else:
            class_channels = list(range(pred.shape[1] - len(classes), pred.shape[1]))

    if min(class_channels) < 4 or max(class_channels) >= pred.shape[1]:
        raise RuntimeError(
            f"Invalid class_channels={class_channels} for output with {pred.shape[1]} channels."
        )

    class_scores = pred[:, class_channels].astype(np.float32)
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = scores >= conf_thres
    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    scores = scores[mask]
    if len(scores) == 0:
        return []

    boxes = xywh_to_xyxy(boxes_xywh)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame_width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame_height - 1)

    detections: list[dict] = []
    for cls_id, class_name in enumerate(classes):
        if allowed_classes is not None and class_name not in allowed_classes:
            continue

        cls_indices = np.where(class_ids == cls_id)[0]
        if len(cls_indices) == 0:
            continue

        keep = nms(boxes[cls_indices], scores[cls_indices], iou_thres)
        for keep_idx in keep:
            idx = cls_indices[keep_idx]
            x1, y1, x2, y2 = boxes[idx].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            center_xy = [round(cx, 2), round(cy, 2)]
            detections.append(
                {
                    "class_id": int(cls_id),
                    "class_name": class_name,
                    "confidence": round(float(scores[idx]), 4),
                    "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "center_xy": center_xy,
                    "center_xy_mm": pixel_to_world_xy(center_xy, plane_mapping),
                }
            )

    if top1_per_class:
        best_by_class: dict[str, dict] = {}
        for det in detections:
            class_name = det["class_name"]
            best = best_by_class.get(class_name)
            if best is None or det["confidence"] > best["confidence"]:
                best_by_class[class_name] = det
        detections = [best_by_class[name] for name in sorted(best_by_class)]

    detections.sort(key=lambda det: (-det["confidence"], det["class_name"]))
    return detections


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


class YoloArucoVisualServoRknnNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_aruco_visual_servo_rknn_node")

        self.declare_parameter("model", "/home/orangepi/test/models/best.rknn")
        self.declare_parameter("source", "0", ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("conf", 0.4)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("imgsz", 416)
        self.declare_parameter("show", False)
        self.declare_parameter("classes", DEFAULT_BLOCK_CLASSES)
        self.declare_parameter("class_channels", [])
        self.declare_parameter("top1_per_class", True)
        self.declare_parameter("calibration", "")
        self.declare_parameter("plane_mapping", "")
        self.declare_parameter("target_color", "red")
        self.declare_parameter("camera_frame", "camera")
        self.declare_parameter("publish_all_debug", True)
        self.declare_parameter("timer_period", 0.05)

        self.declare_parameter("left_marker_id", 1)
        self.declare_parameter("right_marker_id", 2)
        self.declare_parameter("dictionary", "DICT_4X4_50")

        self.model_path = str(self.get_parameter("model").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.show = bool(self.get_parameter("show").value)
        self.top1_per_class = bool(self.get_parameter("top1_per_class").value)
        self.calibration_path = str(self.get_parameter("calibration").value)
        self.plane_mapping_path = str(self.get_parameter("plane_mapping").value)
        self.target_color = str(self.get_parameter("target_color").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.publish_all_debug = bool(self.get_parameter("publish_all_debug").value)
        self.timer_period = float(self.get_parameter("timer_period").value)
        self.left_marker_id = int(self.get_parameter("left_marker_id").value)
        self.right_marker_id = int(self.get_parameter("right_marker_id").value)
        self.dictionary_name = str(self.get_parameter("dictionary").value)

        source_param = self.get_parameter("source").get_parameter_value()
        if source_param.type == ParameterType.PARAMETER_STRING:
            self.source = source_param.string_value
        elif source_param.type == ParameterType.PARAMETER_INTEGER:
            self.source = str(source_param.integer_value)
        else:
            self.source = "0"

        self.classes = [str(name).strip() for name in self.get_parameter("classes").value]
        self.allowed_classes = {name for name in self.classes if name}
        raw_channels = list(self.get_parameter("class_channels").value)
        self.class_channels = [int(v) for v in raw_channels] if raw_channels else None

        self.calibration = load_calibration(self.calibration_path)
        self.plane_mapping = load_plane_mapping(self.plane_mapping_path)
        self.undistort_maps = None

        self.capture = open_capture(self.source)
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")
        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")
        self.aruco_detector = make_aruco_detector(self.dictionary_name)

        self.selected_point_pub = self.create_publisher(
            PointStamped, "/selected_block_mm", 10
        )
        self.selected_color_pub = self.create_publisher(
            String, "/selected_block_color", 10
        )
        self.selected_debug_pub = self.create_publisher(
            String, "/selected_block_debug", 10
        )
        self.all_debug_pub = self.create_publisher(String, "/detected_blocks_debug", 10)

        self.left_pub = self.create_publisher(PointStamped, "/gripper_left_marker_mm", 10)
        self.right_pub = self.create_publisher(
            PointStamped, "/gripper_right_marker_mm", 10
        )
        self.center_pub = self.create_publisher(PointStamped, "/gripper_center_mm", 10)
        self.aruco_debug_pub = self.create_publisher(String, "/gripper_aruco_debug", 10)
        self.error_pub = self.create_publisher(PointStamped, "/visual_servo_error_mm", 10)
        self.combined_debug_pub = self.create_publisher(
            String, "/yolo_aruco_visual_servo_debug", 10
        )

        self.add_on_set_parameters_callback(self._on_set_parameters)
        self.timer = self.create_timer(self.timer_period, self._timer_callback)

        self.get_logger().info(f"RKNN model: {self.model_path}")
        self.get_logger().info(f"Source: {self.source}")
        self.get_logger().info(f"Classes: {self.classes}")
        self.get_logger().info(f"Class channels: {self.class_channels or 'auto'}")
        self.get_logger().info(f"Target color: {self.target_color}")
        self.get_logger().info(
            "Aruco: left_id=%d right_id=%d dictionary=%s"
            % (self.left_marker_id, self.right_marker_id, self.dictionary_name)
        )

    def _on_set_parameters(self, params):
        for param in params:
            if param.name == "target_color":
                self.target_color = str(param.value)
                self.get_logger().info(f"Updated target_color to: {self.target_color}")
        return SetParametersResult(successful=True)

    def _publish_point(self, publisher, x_mm: float, y_mm: float) -> None:
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_frame
        msg.point.x = float(x_mm)
        msg.point.y = float(y_mm)
        msg.point.z = 0.0
        publisher.publish(msg)

    def _select_target_detection(self, detections: list[dict]) -> dict | None:
        target_class = self.target_color.strip().lower()
        if not target_class:
            return None
        if not target_class.endswith("_block"):
            target_class = f"{target_class}_block"

        for det in detections:
            if det["class_name"].lower() == target_class:
                return det
        return None

    def _detect_gripper(self, frame) -> tuple[dict, dict | None]:
        corners_list, ids, _ = detect_markers(frame, self.aruco_detector)
        found: dict[int, dict] = {}
        if ids is not None:
            for marker_id, corners in zip(ids.flatten().tolist(), corners_list):
                center_px = marker_center(corners)
                center_mm = pixel_to_world_xy(center_px, self.plane_mapping)
                if center_mm is None:
                    continue
                found[int(marker_id)] = {
                    "center_px": center_px,
                    "center_mm": center_mm,
                    "corners": corners,
                }

        left = found.get(self.left_marker_id)
        right = found.get(self.right_marker_id)
        gripper = None

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
            gripper = {
                "center_mm": [cx, cy],
                "yaw_rad": yaw_rad,
                "yaw_deg": math.degrees(yaw_rad),
                "corners_list": corners_list,
                "ids": ids,
            }
            self._publish_point(self.center_pub, cx, cy)

        debug = {
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
        if gripper is not None:
            debug["gripper_center_mm"] = gripper["center_mm"]
            debug["gripper_yaw_rad"] = round(gripper["yaw_rad"], 5)
            debug["gripper_yaw_deg"] = round(gripper["yaw_deg"], 2)

        msg = String()
        msg.data = json.dumps(debug, ensure_ascii=True)
        self.aruco_debug_pub.publish(msg)
        return debug, gripper

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

        aruco_debug, gripper = self._detect_gripper(frame)

        frame_height, frame_width = frame.shape[:2]
        resized, scale, pad_x, pad_y = letterbox(frame, self.imgsz)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        model_input = np.expand_dims(rgb, axis=0)

        started = time.perf_counter()
        outputs = self.rknn.inference(inputs=[model_input])
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        detections = parse_yolov8_rknn_output(
            outputs[0],
            classes=self.classes,
            conf_thres=self.conf,
            iou_thres=self.iou,
            class_channels=self.class_channels,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            frame_width=frame_width,
            frame_height=frame_height,
            plane_mapping=self.plane_mapping,
            allowed_classes=self.allowed_classes,
            top1_per_class=self.top1_per_class,
        )

        if self.publish_all_debug:
            debug_msg = String()
            debug_msg.data = json.dumps(
                {
                    "inference_ms": round(elapsed_ms, 2),
                    "detections": detections,
                },
                ensure_ascii=True,
            )
            self.all_debug_pub.publish(debug_msg)

        selected = self._select_target_detection(detections)
        selected_mm = None
        if selected is not None and selected["center_xy_mm"] is not None:
            selected_mm = selected["center_xy_mm"]
            self._publish_point(self.selected_point_pub, selected_mm[0], selected_mm[1])

            color_msg = String()
            color_msg.data = selected["class_name"]
            self.selected_color_pub.publish(color_msg)

            selected_debug_msg = String()
            selected_debug_msg.data = json.dumps(
                {
                    "target_color": self.target_color,
                    "selected": selected,
                    "inference_ms": round(elapsed_ms, 2),
                },
                ensure_ascii=True,
            )
            self.selected_debug_pub.publish(selected_debug_msg)

        error_mm = None
        if selected_mm is not None and gripper is not None:
            error_x = round(selected_mm[0] - gripper["center_mm"][0], 2)
            error_y = round(selected_mm[1] - gripper["center_mm"][1], 2)
            error_norm = round(math.hypot(error_x, error_y), 2)
            error_mm = [error_x, error_y]
            self._publish_point(self.error_pub, error_x, error_y)
            self.get_logger().info(
                "target=(%.1f, %.1f) gripper=(%.1f, %.1f) error=(%.1f, %.1f) norm=%.1fmm"
                % (
                    selected_mm[0],
                    selected_mm[1],
                    gripper["center_mm"][0],
                    gripper["center_mm"][1],
                    error_x,
                    error_y,
                    error_norm,
                )
            )
        else:
            self.get_logger().info(
                "target=%s gripper=%s inference_ms=%.1f"
                % (
                    "ok" if selected_mm is not None else "missing",
                    "ok" if gripper is not None else "missing",
                    elapsed_ms,
                )
            )

        combined_debug_msg = String()
        combined_debug_msg.data = json.dumps(
            {
                "target_color": self.target_color,
                "selected": selected,
                "gripper": None
                if gripper is None
                else {
                    "center_mm": gripper["center_mm"],
                    "yaw_deg": round(gripper["yaw_deg"], 2),
                },
                "error_mm": error_mm,
                "aruco": aruco_debug,
                "inference_ms": round(elapsed_ms, 2),
            },
            ensure_ascii=True,
        )
        self.combined_debug_pub.publish(combined_debug_msg)

        if self.show:
            annotated = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
                cx, cy = [int(round(v)) for v in det["center_xy"]]
                label = f"{det['class_name']} {det['confidence']:.2f}"
                if det["center_xy_mm"] is not None:
                    label += (
                        f" {det['center_xy_mm'][0]:.0f},{det['center_xy_mm'][1]:.0f}mm"
                    )
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.circle(annotated, (cx, cy), 4, (255, 255, 255), -1)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            if gripper is not None:
                ids = gripper["ids"]
                corners_list = gripper["corners_list"]
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(annotated, corners_list, ids)
                cv2.putText(
                    annotated,
                    f"gripper {gripper['center_mm'][0]:.0f},{gripper['center_mm'][1]:.0f}mm",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("RKNN YOLO + ArUco Visual Servo", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self.get_logger().info("Preview closed by user, shutting down.")
                rclpy.shutdown()

    def destroy_node(self) -> bool:
        if hasattr(self, "capture") and self.capture is not None:
            self.capture.release()
        if hasattr(self, "rknn") and self.rknn is not None:
            self.rknn.release()
        if self.show:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = YoloArucoVisualServoRknnNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
