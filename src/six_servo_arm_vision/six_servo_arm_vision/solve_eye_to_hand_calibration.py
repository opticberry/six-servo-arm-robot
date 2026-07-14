#!/usr/bin/env python3
"""Solve eye-to-hand calibration from collected robot/camera pose pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def as_transform(value) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("Expected a 4x4 transform matrix.")
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def rounded_matrix(matrix: np.ndarray, decimals: int = 8) -> list[list[float]]:
    return np.round(matrix.astype(np.float64), decimals=decimals).tolist()


def rotation_error_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    rotation_delta = rotation_a @ rotation_b.T
    cos_angle = (np.trace(rotation_delta) - 1.0) * 0.5
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def load_samples(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples", [])
    if len(samples) < 6:
        raise RuntimeError("At least 6 samples are recommended; 10-20 is better.")
    return samples


def average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    rotations = np.stack([transform[:3, :3] for transform in transforms], axis=0)
    translations = np.stack([transform[:3, 3] for transform in transforms], axis=0)

    rotation_sum = np.sum(rotations, axis=0)
    u, _, vt = np.linalg.svd(rotation_sum)
    rotation_mean = u @ vt
    if np.linalg.det(rotation_mean) < 0.0:
        u[:, -1] *= -1.0
        rotation_mean = u @ vt

    transform_mean = np.eye(4, dtype=np.float64)
    transform_mean[:3, :3] = rotation_mean
    transform_mean[:3, 3] = np.mean(translations, axis=0)
    return transform_mean


def solve(samples: list[dict]) -> dict:
    r_base2tool0 = []
    t_base2tool0 = []
    r_board2camera = []
    t_board2camera = []

    for sample in samples:
        t_base_tool0 = as_transform(sample["T_base_tool0"])
        t_camera_board = as_transform(sample["T_camera_board"])

        # Eye-to-hand trick:
        #   ^bT_c * ^cT_board = ^bT_tool0 * ^tool0T_board
        # Rearranged to OpenCV's H_i * X * K_i = constant form:
        #   ^tool0T_b * ^bT_c * ^cT_board = ^tool0T_board
        # Therefore we input ^tool0T_b as the "gripper2base" argument and
        # OpenCV's returned X is our fixed ^baseT_camera.
        t_tool0_base = invert_transform(t_base_tool0)

        r_base2tool0.append(t_tool0_base[:3, :3])
        t_base2tool0.append(t_tool0_base[:3, 3].reshape(3, 1))
        r_board2camera.append(t_camera_board[:3, :3])
        t_board2camera.append(t_camera_board[:3, 3].reshape(3, 1))

    r_base_camera, t_base_camera = cv2.calibrateHandEye(
        r_base2tool0,
        t_base2tool0,
        r_board2camera,
        t_board2camera,
        method=cv2.CALIB_HAND_EYE_PARK,
    )

    t_base_camera = make_transform(r_base_camera, t_base_camera)
    t_camera_base = invert_transform(t_base_camera)

    t_tool0_board_list = []
    for sample in samples:
        t_base_tool0 = as_transform(sample["T_base_tool0"])
        t_camera_board = as_transform(sample["T_camera_board"])
        t_tool0_board = invert_transform(t_base_tool0) @ t_base_camera @ t_camera_board
        t_tool0_board_list.append(t_tool0_board)

    t_tool0_board_mean = average_transforms(t_tool0_board_list)

    residuals = []
    for sample, t_tool0_board in zip(samples, t_tool0_board_list):
        t_base_tool0 = as_transform(sample["T_base_tool0"])
        t_camera_board_measured = as_transform(sample["T_camera_board"])

        t_camera_board_predicted = t_camera_base @ t_base_tool0 @ t_tool0_board_mean

        translation_error_m = np.linalg.norm(
            t_camera_board_predicted[:3, 3] - t_camera_board_measured[:3, 3]
        )
        residuals.append(
            {
                "index": int(sample.get("index", len(residuals))),
                "translation_error_m": float(translation_error_m),
                "rotation_error_deg": rotation_error_deg(
                    t_camera_board_predicted[:3, :3],
                    t_camera_board_measured[:3, :3],
                ),
            }
        )

    return {
        "sample_count": len(samples),
        "T_base_camera": rounded_matrix(t_base_camera),
        "T_camera_base": rounded_matrix(t_camera_base),
        "T_tool0_board_estimated": rounded_matrix(t_tool0_board_mean),
        "mean_translation_error_m": float(
            np.mean([item["translation_error_m"] for item in residuals])
        ),
        "max_translation_error_m": float(
            np.max([item["translation_error_m"] for item in residuals])
        ),
        "mean_rotation_error_deg": float(
            np.mean([item["rotation_error_deg"] for item in residuals])
        ),
        "max_rotation_error_deg": float(
            np.max([item["rotation_error_deg"] for item in residuals])
        ),
        "residuals": residuals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/orangepi/test/calibration/handeye_samples.json",
        help="Path to the JSON file saved by handeye_sample_collector_node.",
    )
    parser.add_argument(
        "--output",
        default="/home/orangepi/test/calibration/eye_to_hand_result.json",
        help="Path to save calibration result JSON.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    samples = load_samples(input_path)
    result = solve(samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Calibration solved.")
    print("sample_count:", result["sample_count"])
    print("T_base_camera:")
    print(np.array(result["T_base_camera"]))
    print("mean_translation_error_m:", result["mean_translation_error_m"])
    print("max_translation_error_m:", result["max_translation_error_m"])
    print("mean_rotation_error_deg:", result["mean_rotation_error_deg"])
    print("max_rotation_error_deg:", result["max_rotation_error_deg"])
    print("saved:", output_path)


if __name__ == "__main__":
    main()
