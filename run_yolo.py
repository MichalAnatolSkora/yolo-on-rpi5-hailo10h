#!/usr/bin/env python3
"""
Run YOLO object detection via Hailo-10H NPU on a Raspberry Pi 5.

Uses a GStreamer pipeline with Hailo plugins for hardware-accelerated inference.

Prerequisites:
- A YOLOv8 .hef model compiled for Hailo-10H (not Hailo-8).
- OpenCV with GStreamer support (pip install opencv-python).
- Hailo GStreamer plugins (sudo apt install hailo-all).
"""

import argparse
import logging
import os
import signal
import sys

import cv2

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
log = logging.getLogger(__name__)

# Default path for the Hailo YOLO post-processing shared library
DEFAULT_POST_PROCESS_SO = (
    "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so"
)

# Set to True on SIGINT/SIGTERM to exit the capture loop cleanly
_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True


def build_pipeline(
    hef_path: str, post_process_so: str, width: int, height: int, source: str
) -> str:
    """Build a GStreamer pipeline string for Hailo YOLO inference."""
    if source == "picam":
        src = "libcamerasrc"
        caps = f"video/x-raw, width={width}, height={height}, format=RGB"
    else:
        # USB camera via Video4Linux2 (e.g. /dev/video0)
        src = f"v4l2src device={source}"
        caps = f"video/x-raw, width={width}, height={height}"

    return (
        f"{src} ! "
        f"{caps} ! "
        f"videoconvert ! "
        f"hailonet hef-path={hef_path} ! "
        f"hailofilter so-path={post_process_so} ! "
        f"hailooverlay ! "
        f"videoconvert ! "
        f"appsink sync=false max-buffers=1 drop=true"
    )


def run(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.model):
        log.error("Model file not found: %s", args.model)
        sys.exit(1)

    if not os.path.isfile(args.post_process_so):
        log.error("Post-process library not found: %s", args.post_process_so)
        sys.exit(1)

    pipeline = build_pipeline(args.model, args.post_process_so, args.width, args.height, args.source)
    log.info("Opening camera with GStreamer pipeline...")
    log.debug("Pipeline: %s", pipeline)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        log.error(
            "Could not open video capture. "
            "Check camera connection and GStreamer plugin installation."
        )
        sys.exit(1)

    log.info("Running inference. Press 'q' to quit.")

    try:
        while not _shutdown:
            ret, frame = cap.read()
            if not ret:
                log.warning("Failed to fetch frame — retrying...")
                continue

            cv2.imshow("Hailo-10H YOLO Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log.info("Stopped.")


def main() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    parser = argparse.ArgumentParser(
        description="Run YOLO object detection on Hailo-10H NPU"
    )
    parser.add_argument(
        "--model", required=True, help="Path to the Hailo-10H YOLO .hef model"
    )
    parser.add_argument(
        "--labels", default="", help="Path to a text file mapping class IDs to names"
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Camera capture width (default: 640)"
    )
    parser.add_argument(
        "--height", type=int, default=640, help="Camera capture height (default: 640)"
    )
    parser.add_argument(
        "--post-process-so",
        default=DEFAULT_POST_PROCESS_SO,
        help="Path to Hailo YOLO post-processing .so library",
    )
    parser.add_argument(
        "--source",
        default="picam",
        help="Camera source: 'picam' for Raspberry Pi Camera, or a V4L2 device path "
        "for USB cameras (e.g. /dev/video0). Default: picam",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run(args)


if __name__ == "__main__":
    main()
