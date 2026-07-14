#!/usr/bin/env python3
"""ROS 2 PPO inference node for local visual-servo correction in base frame."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from gymnasium import spaces
from rclpy.node import Node
from stable_baselines3 import PPO
from std_msgs.msg import String


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def install_numpy_pickle_compat() -> None:
    """Allow numpy-2-saved SB3 models to load in older numpy environments."""
    try:
        import numpy._core.numeric  # type: ignore  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    try:
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_core_multiarray
        import numpy.core.numeric as numpy_core_numeric
        import numpy.core.umath as numpy_core_umath
    except Exception:
        return

    sys.modules.setdefault("numpy._core", numpy_core)
    sys.modules.setdefault("numpy._core.multiarray", numpy_core_multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy_core_numeric)
    sys.modules.setdefault("numpy._core.umath", numpy_core_umath)

    try:
        import numpy.random._pickle as numpy_random_pickle
    except Exception:
        return

    original_bit_generator_ctor = numpy_random_pickle.__bit_generator_ctor

    def compatible_bit_generator_ctor(bit_generator_name="MT19937"):
        if isinstance(bit_generator_name, type):
            return bit_generator_name()
        return original_bit_generator_ctor(bit_generator_name)

    numpy_random_pickle.__bit_generator_ctor = compatible_bit_generator_ctor


class PpoLocalServoNode(Node):
    def __init__(self) -> None:
        super().__init__("ppo_local_servo_node")

        self.declare_parameter("model", "/home/orangepi/test/models/ppo_local_servo.zip")
        self.declare_parameter("error_topic", "/visual_servo_error_base")
        self.declare_parameter("delta_topic", "/local_servo_delta_base")
        self.declare_parameter("debug_topic", "/ppo_local_servo_debug")
        self.declare_parameter("max_step_m", 0.004)
        self.declare_parameter("stop_error_m", 0.002)
        self.declare_parameter("observation_scale", 1000.0)
        self.declare_parameter("action_scale", 0.001)
        self.declare_parameter("model_initial_error_limit_mm", 25.0)
        self.declare_parameter("model_max_step_mm", 4.0)
        self.declare_parameter("device", "cpu")

        self.model_path = str(self.get_parameter("model").value)
        self.error_topic = str(self.get_parameter("error_topic").value)
        self.delta_topic = str(self.get_parameter("delta_topic").value)
        self.debug_topic = str(self.get_parameter("debug_topic").value)
        self.max_step_m = float(self.get_parameter("max_step_m").value)
        self.stop_error_m = float(self.get_parameter("stop_error_m").value)
        self.observation_scale = float(self.get_parameter("observation_scale").value)
        self.action_scale = float(self.get_parameter("action_scale").value)
        self.model_initial_error_limit_mm = float(
            self.get_parameter("model_initial_error_limit_mm").value
        )
        self.model_max_step_mm = float(self.get_parameter("model_max_step_mm").value)
        self.device = str(self.get_parameter("device").value)

        if self.observation_scale <= 0.0:
            raise ValueError("observation_scale must be > 0")
        if self.action_scale <= 0.0:
            raise ValueError("action_scale must be > 0")

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"PPO model not found: {self.model_path}")
        install_numpy_pickle_compat()
        observation_high = np.array(
            [
                self.model_initial_error_limit_mm,
                self.model_initial_error_limit_mm,
                self.model_max_step_mm,
                self.model_max_step_mm,
            ],
            dtype=np.float32,
        )
        custom_objects = {
            "observation_space": spaces.Box(-observation_high, observation_high, dtype=np.float32),
            "action_space": spaces.Box(
                low=-self.model_max_step_mm,
                high=self.model_max_step_mm,
                shape=(2,),
                dtype=np.float32,
            ),
        }
        self.model = PPO.load(
            self.model_path,
            device=self.device,
            custom_objects=custom_objects,
        )
        self.last_action = np.zeros(2, dtype=np.float32)

        self.create_subscription(
            PointStamped,
            self.error_topic,
            self._on_error,
            10,
        )
        self.delta_pub = self.create_publisher(PointStamped, self.delta_topic, 10)
        self.debug_pub = self.create_publisher(String, self.debug_topic, 10)

        self.get_logger().info(
            "PPO local servo ready: model=%s error_topic=%s delta_topic=%s max_step=%.4fm model_space(error=%.1fmm, action=%.1fmm) scale(obs=%.1f, act=%.4f)"
            % (
                self.model_path,
                self.error_topic,
                self.delta_topic,
                self.max_step_m,
                self.model_initial_error_limit_mm,
                self.model_max_step_mm,
                self.observation_scale,
                self.action_scale,
            )
        )

    def _on_error(self, msg: PointStamped) -> None:
        error_x = float(msg.point.x)
        error_y = float(msg.point.y)
        error_norm = math.hypot(error_x, error_y)

        if error_norm <= self.stop_error_m:
            action = np.zeros(2, dtype=np.float32)
            status = "reached"
        else:
            obs = np.array(
                [
                    error_x * self.observation_scale,
                    error_y * self.observation_scale,
                    self.last_action[0] / self.action_scale,
                    self.last_action[1] / self.action_scale,
                ],
                dtype=np.float32,
            )
            action, _ = self.model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)
            action = action * self.action_scale
            action[0] = clamp(float(action[0]), self.max_step_m)
            action[1] = clamp(float(action[1]), self.max_step_m)
            status = "running"

        self.last_action = action.astype(np.float32)

        delta_msg = PointStamped()
        delta_msg.header.stamp = self.get_clock().now().to_msg()
        delta_msg.header.frame_id = msg.header.frame_id if msg.header.frame_id else "base_link"
        delta_msg.point.x = float(action[0])
        delta_msg.point.y = float(action[1])
        delta_msg.point.z = 0.0
        self.delta_pub.publish(delta_msg)

        debug = {
            "status": status,
            "error_m": [round(error_x, 6), round(error_y, 6)],
            "error_norm_m": round(error_norm, 6),
            "obs_for_model": [round(float(v), 3) for v in obs.tolist()] if error_norm > self.stop_error_m else [0.0, 0.0, round(float(self.last_action[0] / self.action_scale), 3), round(float(self.last_action[1] / self.action_scale), 3)],
            "delta_m": [round(float(action[0]), 6), round(float(action[1]), 6)],
            "model": self.model_path,
        }
        debug_msg = String()
        debug_msg.data = json.dumps(debug, ensure_ascii=True)
        self.debug_pub.publish(debug_msg)


def main() -> None:
    rclpy.init()
    node = PpoLocalServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
