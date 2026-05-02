#!/usr/bin/env python3
"""
Standalone interactive line-config creator. Writes schema v1.

Usage:
    python tools/setup_lines.py --source 0
    python tools/setup_lines.py --source raw_morning.mp4 --frame 200
    python tools/setup_lines.py --source 0 --output my_lines.json

Workflow:
    1. Click two points per line on the displayed frame.
    2. Press Enter when done drawing.
    3. Terminal prompts for each line: name, description, classes.
    4. Saved as v1 JSON.

For the legacy in-script setup (writes v0), use:
    python run_yolo11_tracking.py --setup --source 0

------------------------------------------------------------------------
v0 → v1 hand-migration cheat sheet
------------------------------------------------------------------------

If you have a v0 line_config.json and want to upgrade it without re-drawing
the lines, edit the file directly. The diff:

    {
   +  "schema_version": 1,
   +  "calibrated_at": {"width": 1280, "height": 720},
      "lines": [
        {
          "name": "line_4",
   +      "description": "",
          "p1": [0.2406, 0.0729],
          "p2": [0.2641, 0.9583],
   +      "classes": ["car", "motorcycle", "bus", "truck"],
          "direction": "both",
   +      "enabled": true,
   +      "webhook_url": ""   // optional, per-line override; falls back
                              // to alerts.webhook_url if empty/missing
        }
      ],
   -  "webhook_url": ""
   +  "alerts": {"webhook_url": ""}   // global default
    }

(Set `calibrated_at` to whatever resolution you originally drew the lines
at — it's only used to warn on a mismatch later.)
"""

import argparse
import json
import logging
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hailo_common import COCO_CLASSES, open_camera

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_CLASSES = ["car", "motorcycle", "bus", "truck"]

LINE_COLORS = [
    (0, 0, 255), (0, 200, 255), (0, 255, 255),
    (255, 0, 255), (255, 200, 0), (0, 255, 0),
]


def grab_frame(args):
    """Grab one frame from a camera or video file."""
    if os.path.isfile(args.source):
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            log.error("Could not open video: %s", args.source)
            sys.exit(1)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        target = args.frame if args.frame is not None else max(0, total // 2)
        if total and target >= total:
            target = total - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            log.error("Could not read frame %d from %s", target, args.source)
            sys.exit(1)
        log.info("Frame %d of %d from %s", target, total, args.source)
        return frame

    cap = open_camera(args.source, 1280, 720)
    if not cap.isOpened():
        log.error("Could not open camera: %s", args.source)
        sys.exit(1)
    ret, frame = False, None
    for _ in range(10):
        ret, frame = cap.read()
    cap.release()
    if not ret:
        log.error("Could not capture frame from camera.")
        sys.exit(1)
    return frame


def draw_lines(frame):
    """Show frame, let user click line endpoints, return list of (p1_norm, p2_norm)."""
    h, w = frame.shape[:2]
    points = []
    raw_lines = []  # list of ((x1,y1), (x2,y2)) in pixels

    def redraw():
        display = frame.copy()
        for i, (a, b) in enumerate(raw_lines):
            color = LINE_COLORS[i % len(LINE_COLORS)]
            cv2.line(display, a, b, color, 2)
            mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            cv2.putText(display, f"line_{i+1}", (mid[0], mid[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if len(points) == 1:
            cv2.circle(display, points[0], 6, (0, 255, 0), -1)
            cv2.putText(display, "Click second point...", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, f"Lines: {len(raw_lines)} | Click=add | Enter=done | u=undo | q=cancel",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow("Setup lines (v1)", display)

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        points.append((x, y))
        if len(points) == 2:
            raw_lines.append((points[0], points[1]))
            log.info("Added line %d", len(raw_lines))
            points.clear()
        redraw()

    cv2.namedWindow("Setup lines (v1)")
    cv2.setMouseCallback("Setup lines (v1)", on_click)
    redraw()

    print("\n=== Draw lines ===")
    print("  Click two points per line. Enter=done, u=undo, Esc=clear in-progress, q=cancel.\n")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 13:  # Enter
            if not raw_lines:
                log.warning("Add at least one line first.")
                continue
            break
        elif key == 27:  # Esc
            points.clear()
            redraw()
        elif key == ord("u") and raw_lines:
            raw_lines.pop()
            redraw()
        elif key == ord("q"):
            log.info("Cancelled.")
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()

    # Normalize to [0, 1]
    return [
        (
            [round(a[0] / w, 4), round(a[1] / h, 4)],
            [round(b[0] / w, 4), round(b[1] / h, 4)],
        )
        for a, b in raw_lines
    ]


def prompt(label, default=None, allow_empty=True):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        val = input(f"  {label}{suffix}: ").strip()
        if val:
            return val
        if default is not None:
            return default
        if allow_empty:
            return ""
        print("    (cannot be empty)")


def prompt_classes(default):
    val = input(f"  classes [{','.join(default)}]: ").strip()
    if not val:
        return list(default)
    classes = [c.strip() for c in val.split(",") if c.strip()]
    unknown = [c for c in classes if c not in COCO_CLASSES]
    if unknown:
        log.warning("Unknown COCO classes (will still be saved): %s", ", ".join(unknown))
    return classes


def prompt_direction(default="both"):
    while True:
        val = input(f"  direction (both/positive/negative) [{default}]: ").strip().lower()
        if not val:
            return default
        if val in ("both", "positive", "negative"):
            return val
        print("    must be one of: both, positive, negative")


def collect_metadata(line_indices):
    """Prompt for name/description/direction/classes for each drawn line."""
    print("\n=== Annotate each line ===\n")
    out = []
    for i in line_indices:
        print(f"Line {i+1}:")
        name = prompt("name", default=f"line_{i+1}", allow_empty=False)
        description = prompt("description (optional)", default="")
        direction = prompt_direction()
        classes = prompt_classes(DEFAULT_CLASSES)
        out.append({"name": name, "description": description,
                    "direction": direction, "classes": classes, "enabled": True})
        print()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0",
                        help="Camera index/path or video file")
    parser.add_argument("--frame", type=int, default=None,
                        help="Frame number to grab (video file only; default: middle)")
    parser.add_argument("--output", default="line_config.json",
                        help="Output config path (default: line_config.json)")
    parser.add_argument("--webhook", default="",
                        help="Initial alerts.webhook_url value (default: empty)")
    args = parser.parse_args()

    if os.path.isfile(args.output):
        ans = input(f"{args.output} exists. Overwrite? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)

    frame = grab_frame(args)
    h, w = frame.shape[:2]
    log.info("Frame: %dx%d", w, h)

    lines_norm = draw_lines(frame)
    metadata = collect_metadata(range(len(lines_norm)))

    config = {
        "schema_version": SCHEMA_VERSION,
        "calibrated_at": {"width": w, "height": h},
        "lines": [
            {
                "name": m["name"],
                "description": m["description"],
                "p1": list(p1),
                "p2": list(p2),
                "direction": m["direction"],
                "classes": m["classes"],
                "enabled": m["enabled"],
            }
            for (p1, p2), m in zip(lines_norm, metadata)
        ],
        "alerts": {"webhook_url": args.webhook},
    }

    with open(args.output, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved {len(config['lines'])} line(s) to {args.output} (schema v1)")


if __name__ == "__main__":
    main()
