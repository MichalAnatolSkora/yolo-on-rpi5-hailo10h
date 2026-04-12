#!/usr/bin/env python3
"""
Run YOLO object detection on Hailo-10H NPU or locally (CPU/MPS).

Backend is auto-selected by model extension:
- .hef  -> Hailo-10H NPU (Raspberry Pi)
- .pt   -> Ultralytics (macOS / Linux / Windows)
- .onnx -> Ultralytics (macOS / Linux / Windows)

Usage:
    # Raspberry Pi + Hailo-10H
    python run_yolo11.py --display
    python run_yolo11.py --model ~/hailo_models/yolov11m.hef --display

    # MacBook / laptop
    python run_yolo11.py --model yolo11n.pt --source 0 --display
"""

import argparse
import logging
import signal
import sys

import cv2

from hailo_common import (
    ThreadedCamera, create_session, default_model, default_source,
    draw_detections, load_labels, open_camera,
)

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
log = logging.getLogger(__name__)

_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True


def run(args: argparse.Namespace) -> None:
    with create_session(args.model) as session:
        labels = load_labels(args.labels) if args.labels else session.labels

        cap_w, cap_h = args.input_size
        log.info("Opening camera (source=%s, capture=%dx%d)...", args.source, cap_w, cap_h)
        raw_cap = open_camera(args.source, cap_w, cap_h)
        if not raw_cap.isOpened():
            log.error("Could not open camera. Check connection.")
            sys.exit(1)
        cap = ThreadedCamera(raw_cap)

        log.info("Running YOLO inference. Press 'q' to quit.")
        frame_count = 0

        try:
            while not _shutdown:
                ret, frame = cap.read()
                if not ret:
                    log.warning("Failed to fetch frame — retrying...")
                    continue

                detections = session.detect(
                    frame,
                    conf_threshold=args.confidence,
                    iou_threshold=args.iou,
                    num_classes=len(labels),
                )
                frame = draw_detections(frame, detections, labels)

                frame_count += 1
                if frame_count % 30 == 0:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0
                    log.info("Detections: %d | Camera FPS: %.1f", len(detections), fps)

                if args.display_size:
                    dw, dh = args.display_size
                    show = cv2.resize(frame, (dw, dh)) if (frame.shape[1], frame.shape[0]) != (dw, dh) else frame
                    cv2.imshow("YOLO Inference", show)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if args.display_size:
                cv2.destroyAllWindows()
            log.info("Stopped.")


def main() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    parser = argparse.ArgumentParser(
        description="Run YOLO object detection (Hailo NPU or local CPU/MPS)"
    )
    parser.add_argument(
        "--model", default=default_model(),
        help="Path to model: .hef (Hailo), .pt or .onnx (Ultralytics)",
    )
    parser.add_argument(
        "--labels", default="", help="Path to class labels file (one per line). Defaults to COCO"
    )
    parser.add_argument(
        "--source", default=default_source(),
        help="Camera source: device index (0, 1), V4L2 path (/dev/video0), "
             "or 'picam' for Pi Camera",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)"
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input-small", action="store_const", dest="input_size", const=(640, 480),
        help="Camera capture at 640x480 (default)",
    )
    input_group.add_argument(
        "--input", action="store_const", dest="input_size", const=(1024, 768),
        help="Camera capture at 1024x768",
    )
    input_group.add_argument(
        "--input-large", action="store_const", dest="input_size", const=(1280, 720),
        help="Camera capture at 1280x720",
    )

    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument(
        "--display-small", action="store_const", dest="display_size", const=(640, 480),
        help="Show preview at 640x480",
    )
    display_group.add_argument(
        "--display", action="store_const", dest="display_size", const=(1024, 768),
        help="Show preview at 1024x768 (default size)",
    )
    display_group.add_argument(
        "--display-large", action="store_const", dest="display_size", const=(1280, 720),
        help="Show preview at 1280x720",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    if args.input_size is None:
        args.input_size = (640, 480)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run(args)


if __name__ == "__main__":
    main()
