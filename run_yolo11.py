#!/usr/bin/env python3
"""
Run YOLO object detection via Hailo-10H NPU on a Raspberry Pi 5.

Uses the Hailo Python API (HailoRT) directly instead of GStreamer.
Works with any YOLOv11 .hef model (nano, medium, etc.).

Prerequisites:
- A YOLO .hef model compiled for Hailo-10H.
- OpenCV (pip install opencv-python).
- Hailo RT Python bindings (installed via hailo-all).
- NumPy (pip install numpy).

Usage:
    python run_yolo11.py --display                                        # yolov11n, 640x480
    python run_yolo11.py --model ~/hailo_models/yolov11m.hef --display    # yolov11m
    python run_yolo11.py --display-large --source /dev/video0             # 1280x720, USB cam
"""

import argparse
import logging
import os
import signal
import sys

import cv2

from hailo_common import (
    COCO_CLASSES,
    HailoSession,
    draw_detections,
    load_labels,
    open_camera,
    postprocess_nms,
    postprocess_raw,
    preprocess,
)

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
log = logging.getLogger(__name__)

_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True


def run(args: argparse.Namespace) -> None:
    labels = load_labels(args.labels)

    with HailoSession(args.model) as session:
        # --- Open camera ---
        cap_w, cap_h = args.input_size
        log.info("Opening camera (source=%s, capture=%dx%d)...", args.source, cap_w, cap_h)
        cap = open_camera(args.source, cap_w, cap_h)
        if not cap.isOpened():
            log.error("Could not open camera. Check connection.")
            sys.exit(1)

        log.info("Running YOLO inference. Press 'q' to quit.")
        frame_count = 0

        try:
            while not _shutdown:
                ret, frame = cap.read()
                if not ret:
                    log.warning("Failed to fetch frame — retrying...")
                    continue

                input_data = preprocess(frame, session.input_h, session.input_w)
                output = session.infer(input_data)

                if session.has_nms:
                    detections = postprocess_nms(
                        output.flatten(),
                        frame.shape[0], frame.shape[1],
                        len(labels),
                        args.confidence,
                    )
                else:
                    detections = postprocess_raw(
                        output,
                        frame.shape[0], frame.shape[1],
                        session.input_h, session.input_w,
                        args.confidence, args.iou,
                    )
                frame = draw_detections(frame, detections, labels)

                frame_count += 1
                if frame_count % 30 == 0:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0
                    log.info("Detections: %d | Camera FPS: %.1f", len(detections), fps)

                if args.display_size:
                    dw, dh = args.display_size
                    show = cv2.resize(frame, (dw, dh)) if (frame.shape[1], frame.shape[0]) != (dw, dh) else frame
                    cv2.imshow("YOLO Hailo-10H", show)
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

    default_model = os.path.expanduser("~/hailo_models/yolov11n.hef")

    parser = argparse.ArgumentParser(
        description="Run YOLO object detection on Hailo-10H NPU"
    )
    parser.add_argument(
        "--model", default=default_model,
        help="Path to YOLO .hef model (default: ~/hailo_models/yolov11n.hef)",
    )
    parser.add_argument(
        "--labels", default="", help="Path to class labels file (one per line). Defaults to COCO"
    )
    parser.add_argument(
        "--source", default="/dev/video0",
        help="V4L2 device path for USB camera (e.g. /dev/video0), or 'picam' for Pi Camera",
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
