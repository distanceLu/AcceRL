#!/usr/bin/env python3
"""Batch-test hollow-trace detection on the requested camera runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from detect_hollow_trace import annotate_hollow_trace, detect_hollow_trace


DEFAULT_STAMPS = [
    "14-09-53",
    "14-11-05",
    "14-13-58",
    "14-15-07",
    "14-16-54",
    "14-17-57",
    "14-19-59",
    "14-20-50",
    "14-23-11",
    "14-24-33",
    "14-26-15",
    "14-27-08",
    "14-30-32",
    "14-32-03",
    "14-34-35",
    "14-35-33",
    "14-38-08",
    "14-38-49",
    "14-40-19",
]


def _sample_indices(count: int, samples: int) -> list[int]:
    if count <= samples:
        return list(range(count))
    return sorted(
        set(int(round(value)) for value in np.linspace(0, count - 1, samples + 2)[1:-1])
    )


def _panel(image: np.ndarray, title: str, width: int = 600) -> np.ndarray:
    image_height = int(round(image.shape[0] * width / image.shape[1]))
    resized = cv2.resize(image, (width, image_height), interpolation=cv2.INTER_AREA)
    result = np.full((image_height + 28, width, 3), 255, np.uint8)
    result[28:] = resized
    cv2.putText(
        result,
        title,
        (7, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    return result


def _contact_sheets(
    rows: list[tuple[str, list[tuple[str, np.ndarray]]]], output_dir: Path, rows_per_page: int
) -> list[str]:
    paths = []
    for page, start in enumerate(range(0, len(rows), rows_per_page), 1):
        rendered_rows = []
        for stamp, entries in rows[start : start + rows_per_page]:
            panels = [_panel(image, f"{stamp} | {label}") for label, image in entries]
            while len(panels) < 3:
                panels.append(np.full_like(panels[0], 245))
            rendered_rows.append(np.hstack(panels[:3]))
        sheet = np.vstack(rendered_rows)
        path = output_dir / f"contact_sheet_{page:02d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
        paths.append(str(path))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/data/lcx2/yanjieworkspace/data_collect/2026-08-05"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/hollow_trace_trials/batch")
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--rows-per-page", type=int, default=5)
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        default=(250, 550, 1500, 450),
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
    )
    parser.add_argument(
        "--full-image",
        action="store_true",
        help="Process complete frames (supports mixed camera resolutions)",
    )
    parser.add_argument("--stamps", nargs="*", default=DEFAULT_STAMPS)
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Auto-discover every child directory containing camera_paper_aruco",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_root = args.output_dir / "annotated"
    annotated_root.mkdir(parents=True, exist_ok=True)
    roi_x, roi_y, roi_width, roi_height = args.roi

    report = {
        "data_root": str(args.data_root),
        "roi_xywh": None if args.full_image else list(args.roi),
        "method": "best visible hollow cavity per run, reused across static run frames",
        "runs": [],
        "failures": [],
    }
    sheet_rows: list[tuple[str, list[tuple[str, np.ndarray]]]] = []

    if args.all_runs:
        stamps = sorted(
            path.parent.name
            for path in args.data_root.glob("*/camera_paper_aruco")
            if path.is_dir()
        )
    else:
        stamps = args.stamps

    for stamp in stamps:
        source_dir = args.data_root / stamp / "camera_paper_aruco"
        images = sorted(source_dir.glob("*.jpg"))
        best_detection = None
        best_source = None
        best_source_index = None

        # Search the run for the frame where the hollow topology is least
        # occluded. The paper and camera are static within each run.
        for source_index, source in enumerate(images):
            full = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if full is None:
                continue
            if args.full_image:
                crop = full
                current_roi_x = current_roi_y = 0
            else:
                crop = full[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
                if crop.shape[:2] != (roi_height, roi_width):
                    continue
                current_roi_x, current_roi_y = roi_x, roi_y
            try:
                detection = detect_hollow_trace(crop)
            except RuntimeError:
                continue
            if best_detection is None or detection.score > best_detection.score:
                best_detection = detection
                best_source = source
                best_source_index = source_index
                best_roi_offset = (current_roi_x, current_roi_y)

        if best_detection is None:
            report["failures"].append(
                {"stamp": stamp, "error": "no valid hollow reference in run"}
            )
            continue

        output_run = annotated_root / stamp
        output_run.mkdir(parents=True, exist_ok=True)
        entries: list[tuple[str, np.ndarray]] = []
        run_report = {
            "stamp": stamp,
            "source_count": len(images),
            "reference_source": str(best_source),
            "reference_frame_one_based": best_source_index + 1,
            "reference_threshold": best_detection.threshold,
            "reference_score": round(best_detection.score, 3),
            "trace": {
                "center_global": [
                    round(best_detection.center_x + best_roi_offset[0], 3),
                    round(best_detection.center_y + best_roi_offset[1], 3),
                ],
                "tip_global": [
                    round(best_detection.tip_x + best_roi_offset[0], 3),
                    round(best_detection.tip_y + best_roi_offset[1], 3),
                ],
                "length_px": round(best_detection.length_px, 3),
                "width_px": round(best_detection.width_px, 3),
                "angle_deg": round(best_detection.angle_deg, 3),
                "cavity_area_px": round(best_detection.cavity_area_px, 3),
            },
            "samples": [],
        }

        for source_index in _sample_indices(len(images), args.samples):
            source = images[source_index]
            full = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if args.full_image:
                crop = full.copy()
            else:
                crop = full[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width].copy()
            marked = annotate_hollow_trace(crop, best_detection)
            output = output_run / f"sample_{source_index + 1:03d}_{source.stem}.jpg"
            cv2.imwrite(str(output), marked, [cv2.IMWRITE_JPEG_QUALITY, 95])
            run_report["samples"].append(
                {
                    "frame_one_based": source_index + 1,
                    "source": str(source),
                    "output": str(output),
                }
            )
            entries.append((f"frame {source_index + 1}/{len(images)}", marked))

        report["runs"].append(run_report)
        sheet_rows.append((stamp, entries))

    report["contact_sheets"] = _contact_sheets(
        sheet_rows, args.output_dir, args.rows_per_page
    )
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "processed_runs": len(report["runs"]),
                "processed_samples": sum(len(run["samples"]) for run in report["runs"]),
                "failure_count": len(report["failures"]),
                "contact_sheets": report["contact_sheets"],
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
