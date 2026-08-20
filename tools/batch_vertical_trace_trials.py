#!/usr/bin/env python3
"""Sample robot-camera runs and build contact sheets of vertical-mark detections."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from detect_vertical_trace import annotate, detect_candidates


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
    # Interior quantiles avoid setup/teardown frames while covering the run.
    return sorted(
        set(int(round(value)) for value in np.linspace(0, count - 1, samples + 2)[1:-1])
    )


def _make_panel(image: np.ndarray, title: str, panel_width: int = 600) -> np.ndarray:
    panel_height = int(round(image.shape[0] * panel_width / image.shape[1]))
    resized = cv2.resize(image, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    title_height = 28
    panel = np.full((panel_height + title_height, panel_width, 3), 255, np.uint8)
    panel[title_height:] = resized
    cv2.putText(
        panel,
        title,
        (7, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    return panel


def _write_contact_sheets(
    rows: list[tuple[str, list[tuple[str, np.ndarray]]]], output_dir: Path, rows_per_sheet: int
) -> list[str]:
    written: list[str] = []
    for page_index, start in enumerate(range(0, len(rows), rows_per_sheet), start=1):
        page_rows = rows[start : start + rows_per_sheet]
        rendered_rows = []
        for stamp, entries in page_rows:
            panels = [_make_panel(image, f"{stamp} | {name}") for name, image in entries]
            while len(panels) < 3:
                panels.append(np.full_like(panels[0], 245))
            rendered_rows.append(np.hstack(panels[:3]))
        sheet = np.vstack(rendered_rows)
        path = output_dir / f"contact_sheet_{page_index:02d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 93])
        written.append(str(path))
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/data/lcx2/yanjieworkspace/data_collect/2026-08-05"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/vertical_trace_trials/batch")
    )
    parser.add_argument("--samples", type=int, default=3, help="Samples per run")
    parser.add_argument("--rows-per-sheet", type=int, default=5)
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        default=(400, 600, 1250, 360),
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
    )
    parser.add_argument("--target-relative-x", type=float, default=0.105)
    parser.add_argument("--target-relative-y", type=float, default=0.62)
    parser.add_argument("--min-contrast", type=float, default=3.0)
    parser.add_argument("--stamps", nargs="*", default=DEFAULT_STAMPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.rows_per_sheet < 1:
        raise SystemExit("--samples and --rows-per-sheet must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_root = args.output_dir / "annotated"
    annotated_root.mkdir(parents=True, exist_ok=True)
    roi_x, roi_y, roi_width, roi_height = args.roi

    report: dict[str, object] = {
        "data_root": str(args.data_root),
        "roi_xywh": list(args.roi),
        "target_relative_x": args.target_relative_x,
        "target_relative_y": args.target_relative_y,
        "samples_per_run": args.samples,
        "runs": [],
        "failures": [],
    }
    sheet_rows: list[tuple[str, list[tuple[str, np.ndarray]]]] = []

    for stamp in args.stamps:
        source_dir = args.data_root / stamp / "camera_paper_aruco"
        images = sorted(source_dir.glob("*.jpg"))
        if not images:
            report["failures"].append({"stamp": stamp, "error": "no JPG images"})
            continue

        output_run = annotated_root / stamp
        output_run.mkdir(parents=True, exist_ok=True)
        row_entries: list[tuple[str, np.ndarray]] = []
        run_report = {"stamp": stamp, "source_count": len(images), "samples": []}

        for source_index in _sample_indices(len(images), args.samples):
            source = images[source_index]
            full_image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if full_image is None:
                report["failures"].append(
                    {"stamp": stamp, "source": str(source), "error": "cv2.imread failed"}
                )
                continue
            full_height, full_width = full_image.shape[:2]
            if roi_x + roi_width > full_width or roi_y + roi_height > full_height:
                report["failures"].append(
                    {
                        "stamp": stamp,
                        "source": str(source),
                        "error": f"ROI outside {full_width}x{full_height} image",
                    }
                )
                continue

            crop = full_image[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width].copy()
            try:
                candidates, selected_index, strip_axis_deg, _, _ = detect_candidates(
                    crop,
                    target_relative_x=args.target_relative_x,
                    min_contrast=args.min_contrast,
                    target_relative_y=args.target_relative_y,
                )
            except RuntimeError as error:
                report["failures"].append(
                    {"stamp": stamp, "source": str(source), "error": str(error)}
                )
                continue

            marked = annotate(
                crop,
                candidates,
                selected_index,
                args.target_relative_x,
                strip_axis_deg,
            )
            output_path = output_run / f"sample_{source_index + 1:03d}_{source.stem}.jpg"
            cv2.imwrite(str(output_path), marked, [cv2.IMWRITE_JPEG_QUALITY, 95])
            selected = candidates[selected_index]
            run_report["samples"].append(
                {
                    "source_index_one_based": source_index + 1,
                    "source": str(source),
                    "output": str(output_path),
                    "candidate_count": len(candidates),
                    "selected": asdict(selected),
                    "selected_global_center": [
                        round(selected.center_x + roi_x, 3),
                        round(selected.center_y + roi_y, 3),
                    ],
                }
            )
            row_entries.append((f"frame {source_index + 1}/{len(images)}", marked))

        report["runs"].append(run_report)
        if row_entries:
            sheet_rows.append((stamp, row_entries))

    contact_sheets = _write_contact_sheets(
        sheet_rows, args.output_dir, rows_per_sheet=args.rows_per_sheet
    )
    report["contact_sheets"] = contact_sheets
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    sample_count = sum(len(run["samples"]) for run in report["runs"])
    print(
        json.dumps(
            {
                "processed_runs": len(sheet_rows),
                "processed_samples": sample_count,
                "failure_count": len(report["failures"]),
                "contact_sheets": contact_sheets,
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
