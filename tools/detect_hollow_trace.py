#!/usr/bin/env python3
"""Detect the long pointed-to-rounded hollow trace printed on the gray strip.

The useful visual cue is topology rather than a single line: the printed outline
surrounds a long light cavity. Ink can split that cavity into several components,
so components sharing the same center corridor are joined with a convex hull.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CAVITY_MIN_GRAY = 100
CAVITY_MIN_BRIGHT_FRACTION = 0.55
TRACE_MAX_WIDTH_RATIO = 0.40


@dataclass
class HollowTraceDetection:
    contour: np.ndarray
    parent_contour: np.ndarray
    child_contours: list[np.ndarray]
    threshold: int
    score: float
    center_x: float
    center_y: float
    length_px: float
    width_px: float
    angle_deg: float
    tip_x: float
    tip_y: float
    cavity_area_px: float
    dark_mask: np.ndarray


def _long_axis(rect: tuple) -> tuple[float, float, float]:
    (_, _), (width, height), angle = rect
    if width >= height:
        long_side, short_side, long_angle = width, height, angle
    else:
        long_side, short_side, long_angle = height, width, angle + 90.0
    long_angle = (long_angle + 90.0) % 180.0 - 90.0
    return float(long_side), float(short_side), float(long_angle)


def _contour_center(contour: np.ndarray) -> np.ndarray:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-6:
        return np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
            dtype=float,
        )
    (center_x, center_y), _, _ = cv2.minAreaRect(contour)
    return np.array([center_x, center_y], dtype=float)


def _expand_contour(contour: np.ndarray, image_shape: tuple[int, int], pixels: int = 3) -> np.ndarray:
    mask = np.zeros(image_shape, np.uint8)
    cv2.fillPoly(mask, [contour.astype(np.int32)], 255)
    diameter = pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    expanded = cv2.dilate(mask, kernel)
    contours, _ = cv2.findContours(expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)


def _find_left_tip(
    contour: np.ndarray, angle_deg: float, width_px: float
) -> tuple[float, float]:
    """Return a noise-resistant left endpoint of an elongated contour.

    The extreme projection supplies the longitudinal position, while the median
    of a small apex neighborhood supplies the transverse position. This avoids
    moving the tip because of one jagged threshold pixel.
    """
    angle_rad = math.radians(angle_deg)
    axis = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=float)
    if axis[0] < 0.0:
        axis = -axis
    normal = np.array([-axis[1], axis[0]], dtype=float)

    points = contour.reshape(-1, 2).astype(float)
    along = points @ axis
    across = points @ normal
    min_along = float(np.min(along))
    apex_band = max(2.0, width_px * 0.08)
    apex_across = across[along <= min_along + apex_band]
    if len(apex_across) == 0:
        apex_across = across[np.argmin(along) : np.argmin(along) + 1]
    center_across = float(np.median(apex_across))
    tip = axis * min_along + normal * center_across
    return float(tip[0]), float(tip[1])


def _reconstruct_cavity(
    parent: np.ndarray,
    contours: list[np.ndarray],
    hierarchy: np.ndarray,
    parent_index: int,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, list[np.ndarray], float] | None:
    rect = cv2.minAreaRect(parent)
    parent_center = np.array(rect[0], dtype=float)
    parent_length, parent_width, angle_deg = _long_axis(rect)
    angle_rad = math.radians(angle_deg)
    axis = np.array([math.cos(angle_rad), math.sin(angle_rad)])
    normal = np.array([-axis[1], axis[0]])

    selected: list[np.ndarray] = []
    selected_points: list[np.ndarray] = []
    selected_area = 0.0
    for child_index in np.flatnonzero(hierarchy[0, :, 3] == parent_index):
        child = contours[int(child_index)]
        area = abs(cv2.contourArea(child))
        if area < 40.0:
            continue
        center_delta = _contour_center(child) - parent_center
        along = float(center_delta @ axis)
        across = float(center_delta @ normal)
        if abs(along) > parent_length * 0.55 or abs(across) > parent_width * 0.31:
            continue

        points = child.reshape(-1, 2).astype(float)
        relative = points - parent_center
        point_along = relative @ axis
        point_across = relative @ normal
        keep = (
            (np.abs(point_along) <= parent_length * 0.53)
            & (np.abs(point_across) <= parent_width * 0.34)
        )
        points = points[keep]
        if len(points) < 3:
            continue
        selected.append(child)
        selected_points.append(points.astype(np.int32))
        selected_area += area

    if not selected_points:
        return None
    all_points = np.vstack(selected_points)
    hull = cv2.convexHull(all_points)
    hull_length, hull_width, _ = _long_axis(cv2.minAreaRect(hull))
    if hull_length < parent_length * 0.62 or hull_width < parent_width * 0.22:
        return None

    cavity = _expand_contour(hull, image_shape, pixels=3)
    return cavity, selected, selected_area


def detect_hollow_trace(
    image: np.ndarray,
    thresholds: range = range(90, 171, 10),
) -> HollowTraceDetection:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    best: HollowTraceDetection | None = None

    for threshold in thresholds:
        dark_mask = np.where(gray < threshold, 255, 0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            dark_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is None:
            continue

        for parent_index, parent in enumerate(contours):
            area = abs(cv2.contourArea(parent))
            rect = cv2.minAreaRect(parent)
            (center_x, center_y), _, _ = rect
            length, trace_width, angle = _long_axis(rect)
            aspect = length / max(trace_width, 1.0)

            # These ratios work both on a tight strip crop and on the complete
            # 320x256 / 1920x1080 camera frames. The parent contour is the dark
            # printed outline plus any tick marks connected to it.
            if not width * 0.35 < length < width * 0.66:
                continue
            if not height * 0.05 < trace_width < height * TRACE_MAX_WIDTH_RATIO:
                continue
            if not 5.0 < aspect < 12.0 or abs(angle) > 20.0:
                continue
            if not width * 0.25 < center_x < width * 0.75:
                continue
            if not height * 0.04 < center_y < height * 0.85:
                continue
            if not image.size / 3 * 0.012 < area < image.size / 3 * 0.16:
                continue

            reconstructed = _reconstruct_cavity(
                parent, contours, hierarchy, parent_index, (height, width)
            )
            if reconstructed is None:
                continue
            cavity, children, child_area = reconstructed
            cavity_area = abs(cv2.contourArea(cavity))
            cavity_mask = np.zeros((height, width), np.uint8)
            cv2.fillPoly(cavity_mask, [cavity.astype(np.int32)], 255)
            cavity_pixels = gray[cavity_mask > 0]
            bright_fraction = float(
                np.mean(cavity_pixels > CAVITY_MIN_GRAY)
            )
            # The hollow region is printed on a light gray paper strip.  This
            # rejects elongated holes reconstructed from the near-black table
            # or robot body without constraining the strip's exact position.
            if bright_fraction < CAVITY_MIN_BRIGHT_FRACTION:
                continue
            cavity_length, cavity_width, cavity_angle = _long_axis(
                cv2.minAreaRect(cavity)
            )
            tip_x, tip_y = _find_left_tip(cavity, cavity_angle, cavity_width)
            span_ratio = cavity_length / max(length, 1.0)

            score = (
                100000.0
                - 60.0 * abs(length / width - 0.46) * width
                - 80.0 * abs(trace_width / height - 0.11) * height
                - 0.8 * abs(center_x - width * 0.48)
                - 18.0 * abs(threshold - 120)
                + 0.12 * child_area
                + 8000.0 * min(span_ratio, 1.0)
            )
            detection = HollowTraceDetection(
                contour=cavity,
                parent_contour=parent,
                child_contours=children,
                threshold=threshold,
                score=score,
                center_x=float(center_x),
                center_y=float(center_y),
                length_px=cavity_length,
                width_px=cavity_width,
                angle_deg=cavity_angle,
                tip_x=tip_x,
                tip_y=tip_y,
                cavity_area_px=cavity_area,
                dark_mask=dark_mask,
            )
            if best is None or detection.score > best.score:
                best = detection

    if best is None:
        raise RuntimeError("No elongated hollow trace was found")
    return best


def annotate_hollow_trace(image: np.ndarray, detection: HollowTraceDetection) -> np.ndarray:
    result = image.copy()
    tint = image.copy()
    cv2.fillPoly(tint, [detection.contour], (40, 220, 80))
    result = cv2.addWeighted(tint, 0.20, result, 0.80, 0.0)
    cv2.drawContours(result, [detection.contour], -1, (0, 0, 255), 4, cv2.LINE_AA)

    center = tuple(np.round(_contour_center(detection.contour)).astype(int))
    cv2.circle(result, center, 5, (255, 0, 0), -1, cv2.LINE_AA)
    tip = (int(round(detection.tip_x)), int(round(detection.tip_y)))
    marker_radius = max(6, int(round(min(image.shape[:2]) * 0.009)))
    cv2.circle(result, tip, marker_radius + 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(result, tip, marker_radius, (255, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(
        result,
        f"TIP ({tip[0]}, {tip[1]})",
        (tip[0] + marker_radius + 5, max(18, tip[1] - marker_radius - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        min(0.75, max(0.42, min(image.shape[:2]) / 1000.0)),
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
    label = (
        f"HOLLOW TRACE: length={detection.length_px:.1f}px "
        f"width={detection.width_px:.1f}px angle={detection.angle_deg:.1f}deg"
    )
    scale = min(1.0, max(0.40, min(image.shape[:2]) / 900.0))
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
    )
    cv2.rectangle(
        result,
        (0, 0),
        (min(result.shape[1] - 1, text_width + 10), text_height + 12),
        (255, 255, 255),
        -1,
    )
    cv2.putText(
        result,
        label,
        (5, text_height + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def make_debug_mosaic(
    image: np.ndarray, detection: HollowTraceDetection, annotated: np.ndarray
) -> np.ndarray:
    mask_bgr = cv2.cvtColor(detection.dark_mask, cv2.COLOR_GRAY2BGR)
    components = image.copy()
    cv2.drawContours(components, [detection.parent_contour], -1, (0, 180, 255), 2)
    cv2.drawContours(components, detection.child_contours, -1, (255, 200, 0), 2)
    panels = [image.copy(), mask_bgr, components, annotated.copy()]
    titles = ["original", "dark threshold", "parent + cavity pieces", "reconstructed cavity"]
    for panel, title in zip(panels, titles):
        cv2.rectangle(panel, (0, 0), (240, 24), (255, 255, 255), -1)
        cv2.putText(
            panel,
            title,
            (5, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
    return np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--debug-output", type=Path)
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="Optional crop; by default the complete image is processed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if full_image is None:
        raise SystemExit(f"Could not read image: {args.image}")
    if args.roi:
        roi_x, roi_y, roi_width, roi_height = args.roi
        if (
            roi_x < 0
            or roi_y < 0
            or roi_width <= 0
            or roi_height <= 0
            or roi_x + roi_width > full_image.shape[1]
            or roi_y + roi_height > full_image.shape[0]
        ):
            raise SystemExit(
                f"ROI {args.roi} lies outside image "
                f"{full_image.shape[1]}x{full_image.shape[0]}"
            )
        crop = full_image[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width].copy()
    else:
        roi_x = roi_y = 0
        roi_width, roi_height = full_image.shape[1], full_image.shape[0]
        crop = full_image.copy()

    detection = detect_hollow_trace(crop)
    annotated = annotate_hollow_trace(crop, detection)
    debug = make_debug_mosaic(crop, detection, annotated)
    output = args.output or args.image.with_name(f"{args.image.stem}_hollow_trace.png")
    debug_output = args.debug_output or args.image.with_name(
        f"{args.image.stem}_hollow_trace_debug.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    debug_output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), annotated)
    cv2.imwrite(str(debug_output), debug)

    simplified = cv2.approxPolyDP(detection.contour, 3.0, True).reshape(-1, 2)
    report = {
        "image": str(args.image),
        "roi_xywh": [roi_x, roi_y, roi_width, roi_height],
        "output": str(output),
        "debug_output": str(debug_output),
        "threshold": detection.threshold,
        "score": round(detection.score, 3),
        "center_global": [
            round(detection.center_x + roi_x, 3),
            round(detection.center_y + roi_y, 3),
        ],
        "tip_global": [
            round(detection.tip_x + roi_x, 3),
            round(detection.tip_y + roi_y, 3),
        ],
        "length_px": round(detection.length_px, 3),
        "width_px": round(detection.width_px, 3),
        "angle_deg": round(detection.angle_deg, 3),
        "cavity_area_px": round(detection.cavity_area_px, 3),
        "contour_global": (simplified + np.array([roi_x, roi_y])).tolist(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
