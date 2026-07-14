#!/usr/bin/env python3
"""ROS 2 node: run a YOLOv8 RKNN model and publish one selected color block."""

from __future__ import annotations

import json
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


def pixel_to_world_xy(center_xy: list[float], plane_mapping: dict | None) -> list[float] | None:
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
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

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


class DetectCameraRknnNode(Node):
    def __init__(self) -> None:
        super().__init__("detect_camera_rknn_node")

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

        self.add_on_set_parameters_callback(self._on_set_parameters)
        self.timer = self.create_timer(self.timer_period, self._timer_callback)

        self.get_logger().info(f"RKNN model: {self.model_path}")
        self.get_logger().info(f"Source: {self.source}")
        self.get_logger().info(f"Classes: {self.classes}")
        self.get_logger().info(f"Class channels: {self.class_channels or 'auto'}")
        self.get_logger().info(f"Target color: {self.target_color}")
        if self.calibration is not None:
            self.get_logger().info(f"Calibration: {self.calibration['path']}")
        if self.plane_mapping is not None:
            self.get_logger().info(f"Plane mapping: {self.plane_mapping['path']}")
        if self.show:
            self.get_logger().info("Preview enabled. Press q in the window to quit.")

    def _on_set_parameters(self, params):
        for param in params:
            if param.name == "target_color":
                self.target_color = str(param.value)
                self.get_logger().info(f"Updated target_color to: {self.target_color}")
        return SetParametersResult(successful=True)

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
        if selected is not None and selected["center_xy_mm"] is not None:
            point_msg = PointStamped()
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.header.frame_id = self.camera_frame
            point_msg.point.x = float(selected["center_xy_mm"][0])
            point_msg.point.y = float(selected["center_xy_mm"][1])
            point_msg.point.z = 0.0
            self.selected_point_pub.publish(point_msg)

            color_msg = String()
            color_msg.data = selected["class_name"]
            self.selected_color_pub.publish(color_msg)

            debug_msg = String()
            debug_msg.data = json.dumps(
                {
                    "target_color": self.target_color,
                    "selected": selected,
                    "inference_ms": round(elapsed_ms, 2),
                },
                ensure_ascii=True,
            )
            self.selected_debug_pub.publish(debug_msg)

            self.get_logger().info(
                f"selected={selected['class_name']} "
                f"center_px={selected['center_xy']} "
                f"center_mm={selected['center_xy_mm']} "
                f"inference_ms={elapsed_ms:.1f}"
            )
        else:
            self.get_logger().info(
                f"target_color={self.target_color} not found, detections={len(detections)} "
                f"inference_ms={elapsed_ms:.1f}"
            )

        if self.show:
            annotated = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
                cx, cy = [int(round(v)) for v in det["center_xy"]]
                label = f"{det['class_name']} {det['confidence']:.2f}"
                if det["center_xy_mm"] is not None:
                    label += f" {det['center_xy_mm'][0]:.0f},{det['center_xy_mm'][1]:.0f}mm"
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

            cv2.imshow("RKNN YOLO Camera Preview", annotated)
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
    node = DetectCameraRknnNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
