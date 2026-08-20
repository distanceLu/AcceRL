#!/usr/bin/env python3
"""Measure the image distance from the brush contact point to the trace tip.

The hollow trace is detected in an unobstructed reference frame.  In the target
frame, the low-saturation white brush nib is separated from the static paper by
background difference.  The nib endpoint nearest the trace centerline is used
as the paper contact point.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from detect_hollow_trace import HollowTraceDetection, detect_hollow_trace


# Main OpenCV thresholds.  Keep them together so a new camera/light setup can
# be tuned without searching through the detection implementation.
WHITE_MAX_SATURATION = 40
WHITE_MIN_VALUE = 75
WHITE_MIN_LAB_DIFFERENCE = 18.0
WHITE_MAX_ALONG_SPAN_RATIO = 3.0
INK_MIN_DARKENING = 35
INK_MAX_GRAY = 55


@dataclass
class BrushContactDetection:
    nib_contour: np.ndarray
    candidate_mask: np.ndarray
    contact_x: float
    contact_y: float
    distance_px: float
    axial_distance_px: float
    lateral_offset_px: float
    score: float
    touching_trace: bool
    method: str


def _trace_basis(trace: HollowTraceDetection) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(trace.angle_deg)
    axis = np.array([math.cos(angle), math.sin(angle)], dtype=float)
    if axis[0] < 0.0:
        axis = -axis
    normal = np.array([-axis[1], axis[0]], dtype=float)
    return axis, normal


def _project_pixels(
    shape: tuple[int, int], trace: HollowTraceDetection
) -> tuple[np.ndarray, np.ndarray]:
    axis, normal = _trace_basis(trace)
    yy, xx = np.indices(shape)
    relative_x = xx.astype(np.float32) - trace.tip_x
    relative_y = yy.astype(np.float32) - trace.tip_y
    along = relative_x * axis[0] + relative_y * axis[1]
    across = relative_x * normal[0] + relative_y * normal[1]
    return along, across


def _contact_from_component(
    xs: np.ndarray,
    ys: np.ndarray,
    trace: HollowTraceDetection,
    axis: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, float, float, float, float]:
    points = np.column_stack((xs, ys)).astype(float)
    relative = points - np.array([trace.tip_x, trace.tip_y])
    along = relative @ axis
    across = relative @ normal

    # The brush approaches from above the strip in this camera view.  Therefore
    # the maximum normal projection is the nib end nearest the trace centerline.
    max_across = float(np.max(across))
    apex_band = max(3.0, trace.width_px * 0.08)
    apex_along = along[across >= max_across - apex_band]
    contact_along = float(np.median(apex_along))
    contact = (
        np.array([trace.tip_x, trace.tip_y])
        + axis * contact_along
        + normal * max_across
    )
    return (
        contact,
        contact_along,
        max_across,
        float(np.ptp(along)),
        float(np.ptp(across)),
    )


def _detect_vertical_nib_axis(
    candidate_mask: np.ndarray,
    trace: HollowTraceDetection,
) -> BrushContactDetection | None:
    """Recover a vertical nib even when it is joined to horizontal white paper."""
    axis, normal = _trace_basis(trace)
    along_map, across_map = _project_pixels(candidate_mask.shape, trace)
    kernel_height = max(15, int(round(trace.width_px * 0.5)))
    if kernel_height % 2 == 0:
        kernel_height += 1
    vertical_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, kernel_height)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(vertical_mask)
    best: BrushContactDetection | None = None
    target_across = -trace.width_px * 0.15

    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 40.0:
            continue
        ys, xs = np.nonzero(labels == label)
        along = along_map[ys, xs].astype(float)
        across = across_map[ys, xs].astype(float)
        along_span = float(np.ptp(along))
        across_span = float(np.ptp(across))
        if not trace.width_px * 0.12 < along_span < trace.width_px * 2.0:
            continue
        if not trace.width_px * 0.60 < across_span < trace.width_px * 4.0:
            continue

        slope, intercept = np.polyfit(across, along, 1)
        contact_along = float(slope * target_across + intercept)
        if not trace.length_px * 0.05 < contact_along < trace.length_px * 1.05:
            continue
        contact = (
            np.array([trace.tip_x, trace.tip_y])
            + axis * contact_along
            + normal * target_across
        )
        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour = max(contours, key=cv2.contourArea)
        distance = float(
            np.linalg.norm(contact - np.array([trace.tip_x, trace.tip_y]))
        )
        score = across_span / trace.width_px - 0.2 * along_span / trace.width_px
        detection = BrushContactDetection(
            nib_contour=contour,
            candidate_mask=candidate_mask,
            contact_x=float(contact[0]),
            contact_y=float(contact[1]),
            distance_px=distance,
            axial_distance_px=contact_along,
            lateral_offset_px=target_across,
            score=score,
            touching_trace=True,
            method="white_nib_extrapolated",
        )
        if best is None or detection.score > best.score:
            best = detection
    return best


def detect_brush_contact(
    image: np.ndarray,
    reference: np.ndarray,
    trace: HollowTraceDetection,
) -> BrushContactDetection:
    """Detect the white brush nib and return its paper-facing endpoint."""
    if image.shape != reference.shape:
        raise ValueError(
            f"Image and reference shapes differ: {image.shape} vs {reference.shape}"
        )

    axis, normal = _trace_basis(trace)
    along_map, across_map = _project_pixels(image.shape[:2], trace)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.int16)
    lab_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.int16)
    color_difference = np.linalg.norm(lab_image - lab_reference, axis=2)

    # Only inspect the region where a vertically mounted brush can meet the
    # printed cavity.  This removes the white paper and most printed tick marks.
    corridor = (
        (along_map > trace.length_px * 0.02)
        & (along_map < trace.length_px * 1.12)
        & (across_map > -trace.width_px * 2.8)
        & (across_map < trace.width_px * 0.85)
    )
    white_changed = (
        (hsv[:, :, 1] < WHITE_MAX_SATURATION)
        & (hsv[:, :, 2] > WHITE_MIN_VALUE)
        & (color_difference > WHITE_MIN_LAB_DIFFERENCE)
        & corridor
    )
    candidate_mask = (white_changed.astype(np.uint8) * 255)
    # A mild close reconnects JPEG-fragmented nib pixels.  Do not open first:
    # the pointed end can be only a few pixels wide.
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask)
    best: BrushContactDetection | None = None
    # On this paper the white nib and white paper can be connected by JPEG and
    # illumination differences.  Such a component is deliberately allowed to
    # be large; its paper-facing extreme is still stable.  The normal span and
    # contact-line tests below reject the thin printed outline components.
    min_area = max(80.0, trace.width_px * trace.width_px * 0.18)
    max_area = trace.width_px * trace.width_px * 6.0

    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if not min_area <= area <= max_area:
            continue
        ys, xs = np.nonzero(labels == label)
        if len(xs) < 3:
            continue

        contact, contact_along, contact_across, along_span, across_span = (
            _contact_from_component(xs, ys, trace, axis, normal)
        )
        # A real vertical nib is narrow along the trace axis.  Much wider
        # components are the nib accidentally joined to changed white paper;
        # their endpoint is pulled toward an old position on the strip.
        if not (
            trace.width_px * 0.25
            < along_span
            < trace.width_px * WHITE_MAX_ALONG_SPAN_RATIO
        ):
            continue
        if not trace.width_px * 1.10 < across_span < trace.width_px * 4.0:
            continue
        if not -trace.width_px * 0.55 < contact_across < trace.width_px * 0.25:
            continue
        if not trace.length_px * 0.08 < contact_along < trace.length_px * 1.02:
            continue

        # Prefer a normal-oriented, compact component whose endpoint actually
        # reaches the trace.  Long horizontal paper/outline fragments score low.
        normal_orientation = across_span / max(along_span, 1.0)
        reach_score = 1.0 - min(abs(contact_across) / (trace.width_px * 1.4), 1.0)
        size_score = min(area / max(trace.width_px * trace.width_px, 1.0), 3.0)
        normal_span_score = min(across_span / trace.width_px, 3.0)
        score = (
            4.0 * reach_score
            + normal_span_score
            + size_score
            + min(normal_orientation, 1.0)
        )

        distance = float(
            np.linalg.norm(contact - np.array([trace.tip_x, trace.tip_y]))
        )
        touching = (
            abs(contact_across) <= trace.width_px * 0.75
            and 0.0 <= contact_along <= trace.length_px * 1.05
        )
        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour = max(contours, key=cv2.contourArea)
        detection = BrushContactDetection(
            nib_contour=contour,
            candidate_mask=candidate_mask,
            contact_x=float(contact[0]),
            contact_y=float(contact[1]),
            distance_px=distance,
            axial_distance_px=contact_along,
            lateral_offset_px=contact_across,
            score=score,
            touching_trace=touching,
            method="white_nib",
        )
        if best is None or detection.score > best.score:
            best = detection

    if best is not None:
        return best

    guided = _detect_vertical_nib_axis(candidate_mask, trace)
    if guided is not None:
        try:
            return detect_new_ink_contact(
                image,
                reference,
                trace,
                contact_hint=(guided.contact_x, guided.contact_y),
            )
        except RuntimeError:
            return guided
    return detect_new_ink_contact(image, reference, trace)


def detect_new_ink_contact(
    image: np.ndarray,
    reference: np.ndarray,
    trace: HollowTraceDetection,
    contact_hint: tuple[float, float] | None = None,
) -> BrushContactDetection:
    """Use a new very-dark patch inside the cavity as the contact location."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    cavity = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(cavity, [trace.contour.astype(np.int32)], 255)
    erosion = max(3, int(round(trace.width_px * 0.08)))
    if erosion % 2 == 0:
        erosion += 1
    cavity = cv2.erode(
        cavity,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion, erosion)),
    )
    darkening = gray_reference.astype(np.int16) - gray.astype(np.int16)
    ink_mask = (
        (
            (darkening > INK_MIN_DARKENING)
            & (gray < INK_MAX_GRAY)
            & (cavity > 0)
        ).astype(np.uint8)
        * 255
    )
    ink_mask = cv2.morphologyEx(
        ink_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    hint_point: np.ndarray | None = None
    if contact_hint is not None:
        hint_point = np.asarray(contact_hint, dtype=float)
        axis, _ = _trace_basis(trace)
        hint_delta = hint_point - np.array([trace.tip_x, trace.tip_y])
        hint_along = float(hint_delta @ axis)
        along_map, _ = _project_pixels(gray.shape, trace)
        ink_mask[np.abs(along_map - hint_along) > trace.width_px * 0.75] = 0

    count, labels, stats, centers = cv2.connectedComponentsWithStats(ink_mask)
    area_ratio = 0.003 if hint_point is not None else 0.018
    min_area = max(8.0, trace.width_px * trace.width_px * area_ratio)
    max_area = trace.width_px * trace.width_px * 2.0
    candidates = [
        label
        for label in range(1, count)
        if min_area <= stats[label, cv2.CC_STAT_AREA] <= max_area
    ]
    if not candidates:
        raise RuntimeError("No white brush nib or new ink contact was found")
    if hint_point is None:
        label = max(candidates, key=lambda item: stats[item, cv2.CC_STAT_AREA])
    else:
        label = min(
            candidates,
            key=lambda item: float(np.linalg.norm(centers[item] - hint_point)),
        )
    component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=cv2.contourArea)
    contact = centers[label].astype(float)
    axis, normal = _trace_basis(trace)
    delta = contact - np.array([trace.tip_x, trace.tip_y])
    axial = float(delta @ axis)
    lateral = float(delta @ normal)
    distance = float(np.linalg.norm(delta))
    touching = (
        abs(lateral) <= trace.width_px * 0.75
        and 0.0 <= axial <= trace.length_px * 1.05
    )
    return BrushContactDetection(
        nib_contour=contour,
        candidate_mask=ink_mask,
        contact_x=float(contact[0]),
        contact_y=float(contact[1]),
        distance_px=distance,
        axial_distance_px=axial,
        lateral_offset_px=lateral,
        score=float(stats[label, cv2.CC_STAT_AREA]),
        touching_trace=touching,
        method="white_nib_guided_ink" if hint_point is not None else "new_ink",
    )


def annotate_distance(
    image: np.ndarray,
    trace: HollowTraceDetection,
    brush: BrushContactDetection,
    mm_per_pixel: float | None = None,
    coordinate_offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    result = image.copy()
    cv2.drawContours(result, [trace.contour], -1, (0, 0, 255), 3, cv2.LINE_AA)
    # The selected component can include nearby white paper.  The contact point
    # is reliable, but drawing that whole component would be visually confusing.

    trace_tip = (int(round(trace.tip_x)), int(round(trace.tip_y)))
    brush_tip = (int(round(brush.contact_x)), int(round(brush.contact_y)))
    cv2.line(result, trace_tip, brush_tip, (0, 255, 255), 3, cv2.LINE_AA)

    for point, color in ((trace_tip, (255, 0, 255)), (brush_tip, (0, 255, 255))):
        cv2.circle(result, point, 11, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(result, point, 7, color, -1, cv2.LINE_AA)

    status = "TOUCH" if brush.touching_trace else "ABOVE"
    distance_text = f"distance={brush.distance_px:.1f}px"
    if mm_per_pixel is not None:
        distance_text += f" / {brush.distance_px * mm_per_pixel:.2f}mm"
    label_trace_tip = (
        trace_tip[0] + coordinate_offset[0],
        trace_tip[1] + coordinate_offset[1],
    )
    label_brush_tip = (
        brush_tip[0] + coordinate_offset[0],
        brush_tip[1] + coordinate_offset[1],
    )
    labels = [
        f"TRACE TIP {label_trace_tip}",
        f"BRUSH CONTACT {label_brush_tip} [{status}/{brush.method}]",
        distance_text,
        (
            f"along={brush.axial_distance_px:.1f}px "
            f"lateral={brush.lateral_offset_px:.1f}px"
        ),
    ]
    scale = min(0.85, max(0.48, min(image.shape[:2]) / 1000.0))
    line_height = int(round(27 * scale / 0.6))
    text_sizes = [
        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        for text in labels
    ]
    box_width = max(width for width, _ in text_sizes) + 18
    box_height = line_height * len(labels) + 10
    cv2.rectangle(result, (0, 0), (box_width, box_height), (255, 255, 255), -1)
    for index, text in enumerate(labels):
        cv2.putText(
            result,
            text,
            (8, line_height * (index + 1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
    return result


def make_debug_mosaic(
    image: np.ndarray,
    reference: np.ndarray,
    trace: HollowTraceDetection,
    brush: BrushContactDetection,
    annotated: np.ndarray,
) -> np.ndarray:
    mask = cv2.cvtColor(brush.candidate_mask, cv2.COLOR_GRAY2BGR)
    candidates = image.copy()
    contact = (int(round(brush.contact_x)), int(round(brush.contact_y)))
    cv2.circle(candidates, contact, 10, (0, 220, 255), -1, cv2.LINE_AA)
    panels = [reference.copy(), image.copy(), mask, candidates, annotated.copy()]
    titles = [
        "reference",
        "target",
        "white changed mask",
        "selected contact",
        "distance",
    ]
    target_width = 640
    rendered = []
    for panel, title in zip(panels, titles):
        height = int(round(panel.shape[0] * target_width / panel.shape[1]))
        panel = cv2.resize(panel, (target_width, height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (250, 26), (255, 255, 255), -1)
        cv2.putText(
            panel,
            title,
            (6, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
        rendered.append(panel)
    blank = np.full_like(rendered[0], 245)
    return np.vstack((np.hstack(rendered[:3]), np.hstack(rendered[3:] + [blank])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Frame containing the white brush nib")
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Same static view with the hollow trace visible and brush absent",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--debug-output", type=Path)
    parser.add_argument(
        "--mm-per-pixel",
        type=float,
        help="Optional paper-plane scale for also reporting millimetres",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    reference = cv2.imread(str(args.reference), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")
    if reference is None:
        raise SystemExit(f"Could not read reference image: {args.reference}")

    trace = detect_hollow_trace(reference)
    brush = detect_brush_contact(image, reference, trace)
    annotated = annotate_distance(image, trace, brush, args.mm_per_pixel)
    debug = make_debug_mosaic(image, reference, trace, brush, annotated)

    output = args.output or args.image.with_name(f"{args.image.stem}_brush_distance.png")
    debug_output = args.debug_output or args.image.with_name(
        f"{args.image.stem}_brush_distance_debug.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    debug_output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), annotated)
    cv2.imwrite(str(debug_output), debug)

    report = {
        "image": str(args.image),
        "reference": str(args.reference),
        "output": str(output),
        "debug_output": str(debug_output),
        "trace_tip_global": [round(trace.tip_x, 3), round(trace.tip_y, 3)],
        "brush_contact_global": [round(brush.contact_x, 3), round(brush.contact_y, 3)],
        "touching_trace": brush.touching_trace,
        "method": brush.method,
        "distance_px": round(brush.distance_px, 3),
        "axial_distance_px": round(brush.axial_distance_px, 3),
        "lateral_offset_px": round(brush.lateral_offset_px, 3),
        "distance_mm": (
            round(brush.distance_px * args.mm_per_pixel, 3)
            if args.mm_per_pixel is not None
            else None
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
