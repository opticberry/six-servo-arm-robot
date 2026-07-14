#!/usr/bin/env python3
"""Create a planar pixel-to-mm mapping by clicking 4 points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


CLICK_ORDER = ["top-left", "top-right", "bottom-right", "bottom-left"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze one frame, click 4 plane corners, and save a homography."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, video path, or image path.",
    )
    parser.add_argument(
        "--calibration",
        default="",
        help="Optional path to camera_calibration.json for undistortion.",
    )
    parser.add_argument(
        "--out",
        default="calibration/plane_mapping_a4.json",
        help="Output json path for the plane mapping.",
    )
    parser.add_argument(
        "--width-mm",
        type=float,
        default=210.0,
        help="Real plane width in millimeters.",
    )
    parser.add_argument(
        "--height-mm",
        type=float,
        default=297.0,
        help="Real plane height in millimeters.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Optional capture width when source is a camera.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Optional capture height when source is a camera.",
    )
    return parser.parse_args()


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


def maybe_undistort(frame: np.ndarray, calibration: dict | None, undistort_maps):
    if calibration is None:
        return frame, undistort_maps

    frame_height, frame_width = frame.shape[:2]
    if undistort_maps is None:
        if (
            frame_width != calibration["image_width"]
            or frame_height != calibration["image_height"]
        ):
            print(
                "Warning: calibration image size "
                f"{calibration['image_width']}x{calibration['image_height']} differs from "
                f"current frame size {frame_width}x{frame_height}."
            )
        undistort_maps = build_undistort_maps(calibration, frame_width, frame_height)
    undistorted = cv2.remap(frame, undistort_maps[0], undistort_maps[1], cv2.INTER_LINEAR)
    return undistorted, undistort_maps


def read_source_frame(source: str, width: int, height: int, calibration: dict | None):
    source_path = Path(source)
    if source_path.exists() and source_path.is_file():
        frame = cv2.imread(str(source_path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {source}")
        frame, _ = maybe_undistort(frame, calibration, None)
        return frame

    source_value = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(source_value)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print("Live preview: SPACE freeze frame, q or ESC quit")
    undistort_maps = None
    frozen_frame = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera frame read failed.")

            frame, undistort_maps = maybe_undistort(frame, calibration, undistort_maps)
            preview = frame.copy()
            cv2.putText(
                preview,
                "SPACE: freeze  q/ESC: quit",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Plane Mapping Preview", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                frozen_frame = frame.copy()
                break
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyWindow("Plane Mapping Preview")

    if frozen_frame is None:
        raise RuntimeError("No frame selected.")
    return frozen_frame


def annotate_frame(frame: np.ndarray, clicked_points: list[tuple[int, int]]) -> np.ndarray:
    preview = frame.copy()
    guide = "Click 4 points in order: TL -> TR -> BR -> BL"
    cv2.putText(
        preview,
        guide,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    for idx, (x, y) in enumerate(clicked_points):
        cv2.circle(preview, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(
            preview,
            f"{idx + 1}:{CLICK_ORDER[idx]}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    if len(clicked_points) < 4:
        cv2.putText(
            preview,
            f"Next: {CLICK_ORDER[len(clicked_points)]}",
            (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
    else:
        cv2.polylines(
            preview,
            [np.array(clicked_points, dtype=np.int32)],
            True,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            preview,
            "Press s to save, r to reset, q to quit",
            (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
    return preview


def collect_clicks(frame: np.ndarray) -> list[tuple[int, int]]:
    clicked_points: list[tuple[int, int]] = []
    window_name = "Click Plane Corners"

    def on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
            clicked_points.append((x, y))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        preview = annotate_frame(frame, clicked_points)
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            clicked_points.clear()
        elif key in (ord("q"), 27):
            clicked_points.clear()
            break
        elif key == ord("s") and len(clicked_points) == 4:
            break

    cv2.destroyWindow(window_name)
    if len(clicked_points) != 4:
        raise RuntimeError("Plane point selection was cancelled or incomplete.")
    return clicked_points


def save_mapping(
    out_path: Path,
    clicked_points: list[tuple[int, int]],
    width_mm: float,
    height_mm: float,
    source: str,
    calibration_path: str,
):
    image_points = np.array(clicked_points, dtype=np.float32)
    world_points = np.array(
        [
            [0.0, 0.0],
            [width_mm, 0.0],
            [width_mm, height_mm],
            [0.0, height_mm],
        ],
        dtype=np.float32,
    )

    homography = cv2.getPerspectiveTransform(image_points, world_points)
    homography_inv = cv2.getPerspectiveTransform(world_points, image_points)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "source": source,
        "calibration": calibration_path,
        "plane": {
            "width_mm": width_mm,
            "height_mm": height_mm,
            "corner_order": CLICK_ORDER,
        },
        "image_points": image_points.tolist(),
        "world_points_mm": world_points.tolist(),
        "homography_image_to_world": homography.tolist(),
        "homography_world_to_image": homography_inv.tolist(),
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    args = parse_args()
    calibration = load_calibration(args.calibration)
    frame = read_source_frame(args.source, args.width, args.height, calibration)
    clicked_points = collect_clicks(frame)
    out_path = Path(args.out)
    result = save_mapping(
        out_path,
        clicked_points,
        args.width_mm,
        args.height_mm,
        args.source,
        args.calibration,
    )

    print("Saved plane mapping.")
    print(f"Output: {out_path.resolve()}")
    print("Image points:")
    for idx, point in enumerate(result["image_points"], start=1):
        print(f"  {idx}. {CLICK_ORDER[idx - 1]} -> {point}")
    print("World points (mm):")
    for idx, point in enumerate(result["world_points_mm"], start=1):
        print(f"  {idx}. {CLICK_ORDER[idx - 1]} -> {point}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
