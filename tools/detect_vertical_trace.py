#!/usr/bin/env python3
"""Detect and visualize short dark lines perpendicular to a long horizontal strip.

The target in the user's cropped "figure 2" is not the only vertical-looking line
in the image.  This script therefore detects every plausible perpendicular line
and highlights the candidate nearest an optional horizontal position prior.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass
class Candidate:
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: float
    center_y: float
    length_px: float
    angle_deg: float
    dark_contrast: float
    score: float = 0.0


def _axis_angle_deg(angle_deg: float) -> float:
    """Map an undirected line angle to [-90, 90)."""
    return (angle_deg + 90.0) % 180.0 - 90.0


def _angle_distance_deg(a: float, b: float) -> float:
    """Smallest angle between two undirected lines, in [0, 90]."""
    return abs((a - b + 90.0) % 180.0 - 90.0)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(sorted_values[np.searchsorted(cumulative, cumulative[-1] / 2.0)])


def _sample_gray(gray: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    map_x = xs.astype(np.float32).reshape(-1, 1)
    map_y = ys.astype(np.float32).reshape(-1, 1)
    return cv2.remap(
        gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).reshape(-1).astype(np.float32)


def _line_dark_contrast(
    gray: np.ndarray, line: np.ndarray, strip_axis_deg: float
) -> float:
    x1, y1, x2, y2 = line.astype(float)
    t = np.linspace(0.08, 0.92, 25)
    xs = x1 + (x2 - x1) * t
    ys = y1 + (y2 - y1) * t

    # The two samples beside a perpendicular mark are displaced along the strip.
    axis_rad = math.radians(strip_axis_deg)
    ux, uy = math.cos(axis_rad), math.sin(axis_rad)
    # LSD normally returns an *edge* of a thick black stroke, rather than its
    # center. Search a narrow cross-section for the darkest profile and compare
    # it with profiles sufficiently far away on both sides.
    inner_offset = max(1.5, min(gray.shape[:2]) * 0.018)
    outer_offset = max(4.0, min(gray.shape[:2]) * 0.055)
    inner_profiles = [
        _sample_gray(gray, xs + ux * offset, ys + uy * offset)
        for offset in np.linspace(-inner_offset, inner_offset, 5)
    ]
    darkest_profile = min(float(np.mean(profile)) for profile in inner_profiles)
    side_a = _sample_gray(gray, xs + ux * outer_offset, ys + uy * outer_offset)
    side_b = _sample_gray(gray, xs - ux * outer_offset, ys - uy * outer_offset)
    background = float(np.mean((side_a + side_b) * 0.5))
    return background - darkest_profile


def _merge_candidates(
    raw: list[Candidate], strip_axis_deg: float, image_shape: tuple[int, int]
) -> list[Candidate]:
    """Merge duplicate LSD segments belonging to the two sides of one stroke."""
    height, width = image_shape
    axis_rad = math.radians(strip_axis_deg)
    axis = np.array([math.cos(axis_rad), math.sin(axis_rad)])
    normal = np.array([-axis[1], axis[0]])
    merge_u = max(3.0, width * 0.014)
    max_normal_gap = max(5.0, height * 0.10)

    projected = []
    for candidate in raw:
        p1 = np.array([candidate.x1, candidate.y1], dtype=float)
        p2 = np.array([candidate.x2, candidate.y2], dtype=float)
        u = float(((p1 + p2) * 0.5) @ axis)
        v1, v2 = sorted((float(p1 @ normal), float(p2 @ normal)))
        projected.append((u, v1, v2, candidate))
    projected.sort(key=lambda item: item[0])

    groups: list[list[tuple[float, float, float, Candidate]]] = []
    for item in projected:
        if not groups:
            groups.append([item])
            continue
        group = groups[-1]
        group_u = float(np.mean([entry[0] for entry in group]))
        group_v1 = min(entry[1] for entry in group)
        group_v2 = max(entry[2] for entry in group)
        separated_in_normal = item[1] > group_v2 + max_normal_gap or item[2] < group_v1 - max_normal_gap
        if abs(item[0] - group_u) <= merge_u and not separated_in_normal:
            group.append(item)
        else:
            groups.append([item])

    merged: list[Candidate] = []
    for group in groups:
        weights = np.array([max(entry[3].length_px, 1.0) for entry in group])
        u = float(np.average([entry[0] for entry in group], weights=weights))
        v1 = min(entry[1] for entry in group)
        v2 = max(entry[2] for entry in group)
        p1 = axis * u + normal * v1
        p2 = axis * u + normal * v2
        center = (p1 + p2) * 0.5
        line_angle = _axis_angle_deg(math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])))
        merged.append(
            Candidate(
                x1=int(round(p1[0])),
                y1=int(round(p1[1])),
                x2=int(round(p2[0])),
                y2=int(round(p2[1])),
                center_x=float(center[0]),
                center_y=float(center[1]),
                length_px=float(v2 - v1),
                angle_deg=line_angle,
                dark_contrast=float(max(entry[3].dark_contrast for entry in group)),
            )
        )
    return merged


def detect_candidates(
    image: np.ndarray,
    target_relative_x: float,
    min_contrast: float,
    target_relative_y: float | None = None,
) -> tuple[list[Candidate], int, float, np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    denoised = cv2.GaussianBlur(clahe, (3, 3), 0)
    edges = cv2.Canny(denoised, 45, 130, L2gradient=True)

    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(denoised)[0]
    if detected is None:
        raise RuntimeError("OpenCV did not find any line segments in the image")

    lines = detected.reshape(-1, 4)
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    lengths = np.hypot(dx, dy)
    angles = np.array([_axis_angle_deg(math.degrees(math.atan2(y, x))) for x, y in zip(dx, dy)])

    # Estimate the direction of the long strip from long, nearly horizontal edges.
    long_enough = lengths >= max(18.0, width * 0.14)
    roughly_horizontal = np.abs(angles) <= 35.0
    axis_mask = long_enough & roughly_horizontal
    if np.any(axis_mask):
        strip_axis_deg = _weighted_median(angles[axis_mask], lengths[axis_mask])
    else:
        strip_axis_deg = 0.0

    min_length = max(6.0, height * 0.075)
    raw: list[Candidate] = []
    for line, length, angle in zip(lines, lengths, angles):
        if length < min_length:
            continue
        if _angle_distance_deg(float(angle), strip_axis_deg) < 65.0:
            continue
        contrast = _line_dark_contrast(gray, line, strip_axis_deg)
        if contrast < min_contrast:
            continue
        x1, y1, x2, y2 = line
        raw.append(
            Candidate(
                x1=int(round(x1)),
                y1=int(round(y1)),
                x2=int(round(x2)),
                y2=int(round(y2)),
                center_x=float((x1 + x2) * 0.5),
                center_y=float((y1 + y2) * 0.5),
                length_px=float(length),
                angle_deg=float(angle),
                dark_contrast=contrast,
            )
        )

    candidates = _merge_candidates(raw, strip_axis_deg, (height, width))
    if not candidates:
        raise RuntimeError(
            "No perpendicular dark-line candidates survived filtering; "
            "try --min-contrast 1"
        )

    target_x = target_relative_x * width
    for candidate in candidates:
        position_error = abs(candidate.center_x - target_x) / width
        vertical_error = (
            0.0
            if target_relative_y is None
            else abs(candidate.center_y / height - target_relative_y)
        )
        border_penalty = (
            1.0
            if candidate.center_x < width * 0.04 or candidate.center_x > width * 0.96
            else 0.0
        )
        candidate.score = (
            1.2 * min(candidate.length_px / max(height * 0.25, 1.0), 1.2)
            + 0.5 * min(candidate.dark_contrast / 30.0, 1.5)
            - 8.0 * position_error
            - 4.0 * vertical_error
            - border_penalty
        )
    selected_index = int(np.argmax([candidate.score for candidate in candidates]))
    return candidates, selected_index, strip_axis_deg, clahe, edges


def _draw_dashed_vertical(
    image: np.ndarray, x: int, color: tuple[int, int, int], thickness: int = 1
) -> None:
    for y in range(0, image.shape[0], 12):
        cv2.line(image, (x, y), (x, min(y + 6, image.shape[0] - 1)), color, thickness)


def annotate(
    image: np.ndarray,
    candidates: Iterable[Candidate],
    selected_index: int,
    target_relative_x: float,
    strip_axis_deg: float,
) -> np.ndarray:
    result = image.copy()
    candidates = list(candidates)
    scale = max(0.55, min(image.shape[:2]) / 450.0)
    _draw_dashed_vertical(result, int(round(target_relative_x * image.shape[1])), (255, 80, 0))

    for index, candidate in enumerate(candidates):
        selected = index == selected_index
        color = (0, 0, 255) if selected else (0, 190, 255)
        thickness = 3 if selected else 1
        cv2.line(result, (candidate.x1, candidate.y1), (candidate.x2, candidate.y2), color, thickness, cv2.LINE_AA)
        cv2.circle(
            result,
            (int(round(candidate.center_x)), int(round(candidate.center_y))),
            4 if selected else 2,
            color,
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(
            result,
            f"C{index + 1}",
            (int(candidate.center_x + 4), max(12, int(candidate.center_y - 4))),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            max(1, thickness - 1),
            cv2.LINE_AA,
        )

    selected = candidates[selected_index]
    label = (
        f"SELECTED C{selected_index + 1}: center=({selected.center_x:.1f},"
        f" {selected.center_y:.1f}) angle={selected.angle_deg:.1f} deg; "
        f"strip={strip_axis_deg:.1f} deg"
    )
    label_scale = max(0.38, scale * 0.75)
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 1
    )
    cv2.rectangle(result, (0, 0), (min(result.shape[1] - 1, text_width + 10), text_height + 12), (255, 255, 255), -1)
    cv2.putText(result, label, (5, text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, label_scale, (0, 0, 255), 1, cv2.LINE_AA)
    return result


def _debug_mosaic(
    original: np.ndarray,
    annotated: np.ndarray,
    clahe: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    height, width = original.shape[:2]
    gray_bgr = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    panels = [original, gray_bgr, edges_bgr, annotated]
    names = ["original", "CLAHE gray", "Canny edges", "detected candidates"]
    for panel, name in zip(panels, names):
        cv2.rectangle(panel, (0, 0), (max(110, len(name) * 9), 22), (255, 255, 255), -1)
        cv2.putText(panel, name, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    top = np.hstack(panels[:2])
    bottom = np.hstack(panels[2:])
    return np.vstack((top, bottom)).reshape(height * 2, width * 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to the cropped figure-2 image")
    parser.add_argument("--output", type=Path, help="Annotated output image path")
    parser.add_argument("--debug-output", type=Path, help="Optional four-panel debug image path")
    parser.add_argument(
        "--target-relative-x",
        type=float,
        default=0.19,
        help="Expected x / image-width of the requested mark (default: 0.19)",
    )
    parser.add_argument(
        "--target-relative-y",
        type=float,
        help="Optional expected y / image-height of the requested mark",
    )
    parser.add_argument(
        "--min-contrast",
        type=float,
        default=3.0,
        help="Minimum darkness relative to both sides of a line (default: 3)",
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="Process this pixel ROI and save the annotated crop instead of the full image",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.target_relative_x <= 1.0:
        raise SystemExit("--target-relative-x must be in [0, 1]")
    if args.target_relative_y is not None and not 0.0 <= args.target_relative_y <= 1.0:
        raise SystemExit("--target-relative-y must be in [0, 1]")
    full_image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if full_image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    roi_x = roi_y = 0
    image = full_image
    if args.roi:
        roi_x, roi_y, roi_width, roi_height = args.roi
        full_height, full_width = full_image.shape[:2]
        if roi_width <= 0 or roi_height <= 0:
            raise SystemExit("ROI width and height must be positive")
        if roi_x < 0 or roi_y < 0 or roi_x + roi_width > full_width or roi_y + roi_height > full_height:
            raise SystemExit(
                f"ROI {args.roi} lies outside image size {full_width}x{full_height}"
            )
        image = full_image[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width].copy()

    output = args.output or args.image.with_name(f"{args.image.stem}_vertical_trace.png")
    debug_output = args.debug_output or args.image.with_name(f"{args.image.stem}_vertical_trace_debug.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    debug_output.parent.mkdir(parents=True, exist_ok=True)

    candidates, selected_index, strip_axis_deg, clahe, edges = detect_candidates(
        image,
        target_relative_x=args.target_relative_x,
        min_contrast=args.min_contrast,
        target_relative_y=args.target_relative_y,
    )
    annotated = annotate(
        image,
        candidates,
        selected_index,
        args.target_relative_x,
        strip_axis_deg,
    )
    debug = _debug_mosaic(image.copy(), annotated.copy(), clahe.copy(), edges.copy())
    if not cv2.imwrite(str(output), annotated):
        raise SystemExit(f"Failed to write: {output}")
    if not cv2.imwrite(str(debug_output), debug):
        raise SystemExit(f"Failed to write: {debug_output}")

    selected = candidates[selected_index]
    report = {
        "image": str(args.image),
        "roi_xywh": args.roi,
        "target_relative_xy": [args.target_relative_x, args.target_relative_y],
        "annotated_image": str(output),
        "debug_image": str(debug_output),
        "strip_axis_deg": round(strip_axis_deg, 3),
        "selected": asdict(selected),
        "selected_global_center": [
            round(selected.center_x + roi_x, 3),
            round(selected.center_y + roi_y, 3),
        ],
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
