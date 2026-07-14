#!/usr/bin/env python3
"""Collect paired robot/camera poses for eye-to-hand calibration."""

from __future__ import annotations

import json
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


def transform_msg_to_matrix(transform_msg) -> np.ndarray:
    t = transform_msg.translation
    q = transform_msg.rotation
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)

    norm = (x * x + y * y + z * z + w * w) ** 0.5
    if norm == 0.0:
        raise ValueError("Received zero-length quaternion from TF.")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = [float(t.x), float(t.y), float(t.z)]
    return transform


def matrix_to_list(transform: np.ndarray) -> list[list[float]]:
    return np.round(transform.astype(np.float64), decimals=8).tolist()


class HandeyeSampleCollectorNode(Node):
    def __init__(self) -> None:
        super().__init__("handeye_sample_collector_node")

        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("board_debug_topic", "/handeye_board_pose_debug")
        self.declare_parameter("output", "/home/orangepi/test/calibration/handeye_samples.json")
        self.declare_parameter("min_used_markers", 4)

        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.tool_frame = self.get_parameter("tool_frame").get_parameter_value().string_value
        self.board_debug_topic = (
            self.get_parameter("board_debug_topic").get_parameter_value().string_value
        )
        self.output_path = Path(
            self.get_parameter("output").get_parameter_value().string_value
        )
        self.min_used_markers = (
            self.get_parameter("min_used_markers").get_parameter_value().integer_value
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.latest_board_debug: dict | None = None
        self.samples: list[dict] = []
        self.running = True

        self.create_subscription(
            String,
            self.board_debug_topic,
            self._board_debug_callback,
            10,
        )

        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()

        self.get_logger().info(
            "Collector ready. Press Enter to save a sample, input 'q' then Enter to quit."
        )
        self.get_logger().info(
            "TF pair: %s -> %s, board topic: %s"
            % (self.base_frame, self.tool_frame, self.board_debug_topic)
        )

    def _board_debug_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Received invalid board debug JSON.")
            return
        self.latest_board_debug = data

    def _input_loop(self) -> None:
        while self.running and rclpy.ok():
            line = sys.stdin.readline()
            if line == "":
                continue
            command = line.strip().lower()
            if command in ("q", "quit", "exit"):
                self._save_samples()
                self.running = False
                rclpy.shutdown()
                return
            self._capture_sample()

    def _capture_sample(self) -> None:
        if self.latest_board_debug is None:
            self.get_logger().warning("No board pose received yet.")
            return

        if not self.latest_board_debug.get("board_detected", False):
            self.get_logger().warning("Board is not detected. Sample not saved.")
            return

        used_markers = int(self.latest_board_debug.get("used_marker_count", 0))
        if used_markers < self.min_used_markers:
            self.get_logger().warning(
                "Only %d board markers used. Sample not saved." % used_markers
            )
            return

        t_camera_board = self.latest_board_debug.get("T_camera_board")
        if t_camera_board is None:
            self.get_logger().warning("T_camera_board is missing. Sample not saved.")
            return

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tool_frame,
                rclpy.time.Time(),
            )
        except TransformException as exc:
            self.get_logger().warning("TF lookup failed: %s" % exc)
            return

        t_base_tool0 = transform_msg_to_matrix(tf_msg.transform)
        sample = {
            "index": len(self.samples),
            "stamp": datetime.now().isoformat(timespec="milliseconds"),
            "base_frame": self.base_frame,
            "tool_frame": self.tool_frame,
            "camera_frame": self.latest_board_debug.get("camera_frame", "camera"),
            "board_frame": self.latest_board_debug.get("board_frame", "aruco_board"),
            "used_marker_count": used_markers,
            "T_base_tool0": matrix_to_list(t_base_tool0),
            "T_camera_board": t_camera_board,
        }
        self.samples.append(sample)
        self._save_samples()
        self.get_logger().info(
            "Saved sample %d, used_markers=%d, output=%s"
            % (sample["index"], used_markers, self.output_path)
        )

    def _save_samples(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "description": "Eye-to-hand samples: fixed camera observes ArUco board mounted on tool0.",
            "sample_count": len(self.samples),
            "samples": self.samples,
        }
        self.output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def destroy_node(self) -> bool:
        self.running = False
        self._save_samples()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = HandeyeSampleCollectorNode()
    try:
        while rclpy.ok() and node.running:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
