#!/usr/bin/env python3
"""
Offline tracker evaluation — run the tracker on a recorded video and compare
the resulting line-crossing counts to a ground-truth (manually counted) value.

Workflow:
    1. Record a clean clip:    python evaluation/record_raw.py --display --duration 60
    2. Manually count vehicles per counting line.
    3. Run this script:        python evaluation/evaluate.py raw_*.mp4 --expected 12

Reads every frame of the file in sequence (no frame dropping), so counts are
deterministic — re-running on the same file with the same config and model
gives the same result.

Examples:
    # Total expected = 12, fail if off by more than 1
    python evaluation/evaluate.py raw_morning.mp4 --expected 12 --tolerance 1

    # Per-line expected (inline)
    python evaluation/evaluate.py raw_morning.mp4 --expected line_1=5,line_2=7

    # Per-line expected (JSON file)
    python evaluation/evaluate.py raw_morning.mp4 --expected expected.json

    # Visual debugging — write an annotated copy showing what the tracker saw
    python evaluation/evaluate.py raw_morning.mp4 --annotate annotated.mp4 --display

Exit code: 0 if every counted line is within --tolerance of expected, 1 otherwise.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time

import cv2

# Allow running as a script from the evaluation/ subfolder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hailo_common import create_session, default_model, load_labels, load_tracker_config
from run_yolo11_tracking import (
    IOUTracker,
    MultiLineVehicleCounter,
    VEHICLE_CLASS_IDS,
    deduplicate_detections,
    draw_tracking_multiline,
    load_config,
)

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
log = logging.getLogger(__name__)

_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True


# ---------------------------------------------------------------------------
# Expected-count parsing
# ---------------------------------------------------------------------------

def parse_expected(spec: str | None):
    """Parse the --expected argument.

    Returns:
        - None              if spec is None
        - int               if spec is a plain integer (total expected)
        - dict[str, int]    if spec is "line=N,line=N..." or path to a .json file
    """
    if spec is None:
        return None
    # JSON file form
    if spec.endswith(".json") or os.path.isfile(spec):
        with open(spec) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            log.error("Expected JSON must be a dict like {\"line_1\": 5}, got %s", type(data).__name__)
            sys.exit(1)
        return {k: int(v) for k, v in data.items()}
    # Inline k=v form
    if "=" in spec:
        out = {}
        for kv in spec.split(","):
            if "=" not in kv:
                log.error("Bad --expected segment: %r (need 'name=N')", kv)
                sys.exit(1)
            k, v = kv.split("=", 1)
            out[k.strip()] = int(v.strip())
        return out
    # Plain integer (total)
    try:
        return int(spec)
    except ValueError:
        log.error("Could not parse --expected %r as integer, k=v list, or .json path", spec)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Codec fallback (same as record_raw.py / run_yolo11_tracking.py)
# ---------------------------------------------------------------------------

def open_writer(path: str, w: int, h: int, fps: float):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".avi":
        candidates = [("MJPG", path), ("XVID", path)]
    else:
        candidates = [
            ("avc1", path),
            ("mp4v", path),
            ("MJPG", os.path.splitext(path)[0] + ".avi"),
        ]
    for codec, p in candidates:
        w_obj = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if w_obj.isOpened():
            return w_obj, p, codec
        w_obj.release()
    return None, path, None


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.recording):
        log.error("Recording not found: %s", args.recording)
        return 2

    config = load_config(args.config)
    expected = parse_expected(args.expected)

    cap = cv2.VideoCapture(args.recording)
    if not cap.isOpened():
        log.error("Could not open recording: %s", args.recording)
        return 2

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    file_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    file_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / file_fps if file_fps else 0.0
    log.info(
        "Recording: %s — %dx%d, %.1f fps, %d frames (~%.1fs)",
        args.recording, file_w, file_h, file_fps, total_frames, duration_s,
    )

    tracker = IOUTracker(
        max_disappeared=args.max_disappeared,
        min_iou=args.min_iou,
        max_distance=args.max_distance,
    )
    counter = MultiLineVehicleCounter(
        config["lines"], buffer_px=args.buffer, min_hits=args.min_hits,
    )
    line_names = [l["name"] for l in config["lines"]]
    log.info("Lines: %s | min_hits=%d, buffer=%dpx", ", ".join(line_names), args.min_hits, args.buffer)

    annotated_writer: cv2.VideoWriter | None = None
    annotated_path: str | None = None

    frame_idx = 0
    eval_start = time.time()
    crossings_log: list[tuple[int, int, str, str]] = []  # (frame, tid, line, dir)

    with create_session(args.model) as session:
        labels = load_labels(args.labels) if args.labels else session.labels
        log.info("Loaded model: %s | classes=%d", args.model, len(labels))

        try:
            while not _shutdown:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = session.detect(
                    frame,
                    conf_threshold=args.confidence,
                    iou_threshold=args.iou,
                    num_classes=len(labels),
                )

                if args.all_classes:
                    veh = detections
                else:
                    veh = [d for d in detections if d[5] in VEHICLE_CLASS_IDS]

                equiv = None if args.all_classes else VEHICLE_CLASS_IDS
                if not args.no_deduplicate:
                    veh = deduplicate_detections(veh, iou_threshold=args.iou, equiv_classes=equiv)

                tracked = tracker.update(veh)
                h, w = frame.shape[:2]
                crossings = counter.update(tracked, w, h, tracker=tracker)
                for tid, name, direction in crossings:
                    crossings_log.append((frame_idx, tid, name, direction))
                    log.info(
                        "Frame %d: track #%d crossed '%s' (%s)",
                        frame_idx, tid, name, direction,
                    )

                if args.annotate or args.display:
                    recent = {tid: name for tid, name, _ in crossings}
                    annotated = draw_tracking_multiline(
                        frame.copy(), tracked, labels, config["lines"], counter, recent,
                    )

                    if args.annotate:
                        if annotated_writer is None:
                            annotated_writer, annotated_path, codec = open_writer(
                                args.annotate, w, h, file_fps,
                            )
                            if annotated_writer is not None:
                                log.info("Annotated output: %s (codec=%s)", annotated_path, codec)
                            else:
                                log.error("Could not open annotated writer for %s", args.annotate)
                        if annotated_writer is not None:
                            annotated_writer.write(annotated)

                    if args.display:
                        cv2.imshow("Evaluation", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            log.info("User stopped early.")
                            break

                frame_idx += 1
                if total_frames and (frame_idx % 60 == 0 or frame_idx == total_frames):
                    pct = 100.0 * frame_idx / total_frames
                    elapsed = time.time() - eval_start
                    proc_fps = frame_idx / max(elapsed, 1e-3)
                    log.info(
                        "Frame %d/%d (%.1f%%) — proc fps=%.1f — total crossings=%d",
                        frame_idx, total_frames, pct, proc_fps, counter.total(),
                    )
        finally:
            cap.release()
            if annotated_writer is not None:
                annotated_writer.release()
                log.info("Saved annotated video: %s", annotated_path)
            if args.display:
                cv2.destroyAllWindows()

    # ----- Report -----
    eval_secs = time.time() - eval_start
    actual_per_line = {name: counter.total(name) for name in line_names}
    actual_total = counter.total()

    print()
    print("=" * 64)
    print(f"Evaluation: {args.recording}")
    print(f"Processed:  {frame_idx}/{total_frames} frames in {eval_secs:.1f}s "
          f"({frame_idx/max(eval_secs,1e-3):.1f} fps)")
    print()

    failed_lines: list[str] = []
    rows: list[tuple[str, int, str, str, str]] = []  # (name, actual, expected, diff, status)

    if isinstance(expected, dict):
        for name in line_names:
            actual = actual_per_line[name]
            exp = expected.get(name)
            if exp is None:
                rows.append((name, actual, "—", "—", "—"))
                continue
            diff = actual - exp
            ok = abs(diff) <= args.tolerance
            if not ok:
                failed_lines.append(name)
            rows.append((name, actual, str(exp), f"{diff:+d}", "OK" if ok else "FAIL"))
        # Total summary if expected has a "_total" key, otherwise sum of present
        rows.append(("TOTAL", actual_total, str(sum(expected.values())),
                     f"{actual_total - sum(expected.values()):+d}", ""))
    elif isinstance(expected, int):
        for name in line_names:
            rows.append((name, actual_per_line[name], "—", "—", "—"))
        diff = actual_total - expected
        ok = abs(diff) <= args.tolerance
        if not ok:
            failed_lines.append("TOTAL")
        rows.append(("TOTAL", actual_total, str(expected), f"{diff:+d}", "OK" if ok else "FAIL"))
    else:
        for name in line_names:
            rows.append((name, actual_per_line[name], "—", "—", "—"))
        rows.append(("TOTAL", actual_total, "—", "—", "—"))

    print(f"  {'Line':<22} {'Actual':>8}  {'Expected':>10}  {'Diff':>6}  {'Status':>6}")
    print(f"  {'-'*22} {'-'*8}  {'-'*10}  {'-'*6}  {'-'*6}")
    for name, actual, exp, diff, status in rows:
        print(f"  {name:<22} {actual:>8}  {exp:>10}  {diff:>6}  {status:>6}")

    if args.json:
        payload = {
            "recording": args.recording,
            "frames_processed": frame_idx,
            "eval_seconds": round(eval_secs, 2),
            "actual_per_line": actual_per_line,
            "actual_total": actual_total,
            "expected": expected,
            "tolerance": args.tolerance,
            "failed_lines": failed_lines,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        log.info("Wrote JSON results to %s", args.json)

    print()
    if expected is None:
        print("(no --expected given; pass --expected N or --expected file.json to enable PASS/FAIL)")
        return 0

    if failed_lines:
        print(f"RESULT: FAIL — {len(failed_lines)} line(s) outside tolerance ±{args.tolerance}: "
              f"{', '.join(failed_lines)}")
        return 1
    print(f"RESULT: PASS — all lines within tolerance ±{args.tolerance}")
    return 0


def main() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _signal_handler)

    parser = argparse.ArgumentParser(
        description="Run the tracker on a recorded video and compare to ground truth."
    )
    parser.add_argument("recording", help="Path to recorded video (e.g. raw_*.mp4)")
    parser.add_argument(
        "--config", default="line_config.json",
        help="Line config JSON (default: ./line_config.json)",
    )
    parser.add_argument(
        "--expected", default=None,
        help="Ground-truth count. Forms: 'N' (total), 'line_1=5,line_2=7' (per-line), "
             "or path to a .json file mapping line names to counts.",
    )
    parser.add_argument(
        "--tolerance", type=int, default=0,
        help="Allowed absolute difference between actual and expected (default: 0)",
    )

    parser.add_argument(
        "--annotate", default=None, metavar="PATH",
        help="Write an annotated copy of the recording (with boxes, lines, counts) for visual debugging",
    )
    parser.add_argument("--display", action="store_true",
                        help="Show preview window while evaluating (slower)")
    parser.add_argument("--json", default=None, metavar="PATH",
                        help="Write results (per-line actual/expected/diff) to JSON file")

    tcfg = load_tracker_config(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tracker_config.json")
    )

    parser.add_argument("--model", default=default_model(),
                        help="Model path: .hef (Hailo) or .pt / .onnx (Ultralytics)")
    parser.add_argument("--labels", default="")
    parser.add_argument("--confidence", type=float, default=tcfg["confidence"])
    parser.add_argument("--iou", type=float, default=tcfg["iou"])

    parser.add_argument("--all-classes", action="store_true")
    parser.add_argument("--no-deduplicate", action="store_true", default=not tcfg["deduplicate"])

    parser.add_argument("--buffer", type=int, default=tcfg["buffer"])
    parser.add_argument("--max-disappeared", type=int, default=tcfg["max_disappeared"])
    parser.add_argument("--min-iou", type=float, default=tcfg["min_iou"])
    parser.add_argument("--max-distance", type=float, default=tcfg["max_distance"])
    parser.add_argument("--min-hits", type=int, default=tcfg["min_hits"])

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    sys.exit(evaluate(args))


if __name__ == "__main__":
    main()
