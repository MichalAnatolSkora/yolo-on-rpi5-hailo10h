#!/usr/bin/env python3
"""
Vehicle tracking and counting using YOLO on Hailo-10H NPU.

Extends run_yolo11.py with:
- Centroid-based object tracking across frames.
- A configurable counting line — vehicles that cross it are counted.
- Filters for vehicle classes only (car, motorcycle, bus, truck).

Usage:
    python run_yolo11_tracking.py --display
    python run_yolo11_tracking.py --display --line-y 0.6          # line at 60% height
    python run_yolo11_tracking.py --display --direction both      # count both directions
    python run_yolo11_tracking.py --display --source /dev/video0
"""

import argparse
import logging
import os
import signal
import sys
from collections import OrderedDict

import cv2
import numpy as np

from run_yolo11 import (
    COCO_CLASSES,
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

# COCO class IDs for vehicles
VEHICLE_CLASSES = {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
VEHICLE_CLASS_IDS = set(VEHICLE_CLASSES.values())


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True


class CentroidTracker:
    """
    Simple centroid-based multi-object tracker.

    Associates detections to existing tracks by minimum Euclidean distance.
    """

    def __init__(self, max_disappeared: int = 30, max_distance: float = 80.0):
        self.next_id = 0
        self.objects: OrderedDict[int, np.ndarray] = OrderedDict()  # id -> centroid
        self.bboxes: dict[int, tuple] = {}  # id -> (x1, y1, x2, y2)
        self.disappeared: dict[int, int] = {}  # id -> frames missing
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _register(self, centroid: np.ndarray, bbox: tuple) -> int:
        obj_id = self.next_id
        self.objects[obj_id] = centroid
        self.bboxes[obj_id] = bbox
        self.disappeared[obj_id] = 0
        self.next_id += 1
        return obj_id

    def _deregister(self, obj_id: int):
        del self.objects[obj_id]
        del self.bboxes[obj_id]
        del self.disappeared[obj_id]

    def update(self, detections: list[tuple]) -> dict[int, tuple]:
        """
        Update tracks with new detections.

        Args:
            detections: list of (x1, y1, x2, y2, conf, class_id)

        Returns:
            dict of {track_id: (x1, y1, x2, y2, conf, class_id)}
        """
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return {}

        input_centroids = np.array([
            ((d[0] + d[2]) / 2.0, (d[1] + d[3]) / 2.0) for d in detections
        ])

        if len(self.objects) == 0:
            result = {}
            for i, det in enumerate(detections):
                obj_id = self._register(input_centroids[i], det[:4])
                result[obj_id] = det
            return result

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        # Compute distance matrix
        dists = np.linalg.norm(
            object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2
        )

        # Greedy assignment: sort by distance, assign closest pairs
        rows = dists.min(axis=1).argsort()
        cols = dists.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()
        result = {}

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if dists[row, col] > self.max_distance:
                continue

            obj_id = object_ids[row]
            self.objects[obj_id] = input_centroids[col]
            self.bboxes[obj_id] = detections[col][:4]
            self.disappeared[obj_id] = 0
            result[obj_id] = detections[col]

            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing tracks
        for row in range(len(object_ids)):
            if row not in used_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)

        # Register new detections
        for col in range(len(detections)):
            if col not in used_cols:
                obj_id = self._register(input_centroids[col], detections[col][:4])
                result[obj_id] = detections[col]

        return result


class VehicleCounter:
    """Count vehicles crossing a horizontal line."""

    def __init__(self, line_y_ratio: float = 0.5, direction: str = "down"):
        """
        Args:
            line_y_ratio: Y position of counting line as fraction of frame height (0-1).
            direction: "down", "up", or "both".
        """
        self.line_y_ratio = line_y_ratio
        self.direction = direction
        self.prev_centroids: dict[int, float] = {}  # track_id -> previous cy
        self.counted_ids: set[int] = set()
        self.count_down = 0
        self.count_up = 0

    @property
    def total(self) -> int:
        if self.direction == "both":
            return self.count_down + self.count_up
        elif self.direction == "down":
            return self.count_down
        else:
            return self.count_up

    def update(self, tracked: dict[int, tuple], frame_h: int) -> list[int]:
        """
        Check which tracked vehicles crossed the line this frame.

        Returns list of track_ids that just crossed.
        """
        line_y = int(self.line_y_ratio * frame_h)
        crossed = []

        for track_id, det in tracked.items():
            cy = (det[1] + det[3]) / 2.0

            if track_id in self.prev_centroids and track_id not in self.counted_ids:
                prev_cy = self.prev_centroids[track_id]

                # Crossed downward
                if prev_cy < line_y <= cy:
                    if self.direction in ("down", "both"):
                        self.count_down += 1
                        self.counted_ids.add(track_id)
                        crossed.append(track_id)

                # Crossed upward
                elif prev_cy > line_y >= cy:
                    if self.direction in ("up", "both"):
                        self.count_up += 1
                        self.counted_ids.add(track_id)
                        crossed.append(track_id)

            self.prev_centroids[track_id] = cy

        # Clean up old IDs
        active_ids = set(tracked.keys())
        stale = [tid for tid in self.prev_centroids if tid not in active_ids]
        for tid in stale:
            del self.prev_centroids[tid]

        return crossed


def draw_tracking(
    frame: np.ndarray,
    tracked: dict[int, tuple],
    labels: list[str],
    counter: VehicleCounter,
    crossed_ids: list[int],
) -> np.ndarray:
    """Draw tracked vehicles, counting line, and count overlay."""
    h, w = frame.shape[:2]
    line_y = int(counter.line_y_ratio * h)

    # Draw counting line
    cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 2)

    for track_id, det in tracked.items():
        x1, y1, x2, y2, conf, cls_id = det
        label = labels[cls_id] if cls_id < len(labels) else str(cls_id)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Green for normal, yellow for just-crossed
        color = (0, 255, 255) if track_id in crossed_ids else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"#{track_id} {label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Centroid dot
        cv2.circle(frame, (cx, cy), 4, color, -1)

    # Count overlay
    if counter.direction == "both":
        count_text = f"Down: {counter.count_down}  Up: {counter.count_up}  Total: {counter.total}"
    else:
        count_text = f"Count: {counter.total}"

    cv2.rectangle(frame, (10, 10), (10 + 300, 50), (0, 0, 0), -1)
    cv2.putText(frame, count_text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return frame


def run(args: argparse.Namespace) -> None:
    try:
        from hailo_platform import HEF, VDevice, FormatType, HailoSchedulingAlgorithm
    except ImportError:
        log.error(
            "hailo_platform not found. Ensure hailo-all / hailo-h10-all is installed "
            "and you are using system Python or a venv with --system-site-packages."
        )
        sys.exit(1)

    if not os.path.isfile(args.model):
        log.error("Model file not found: %s", args.model)
        sys.exit(1)

    labels = load_labels(args.labels)

    # --- Load HEF model ---
    log.info("Loading HEF model: %s", args.model)
    hef = HEF(args.model)

    output_vstream_infos = hef.get_output_vstream_infos()
    has_nms = any("nms" in info.name.lower() for info in output_vstream_infos)
    if has_nms:
        log.info("Model has on-chip NMS — using NMS output format")

    input_vstream_infos = hef.get_input_vstream_infos()
    input_shape = input_vstream_infos[0].shape
    input_h, input_w = input_shape[0], input_shape[1]
    log.info("Model input shape: %s", input_shape)

    # --- Init tracker and counter ---
    tracker = CentroidTracker(
        max_disappeared=args.max_disappeared,
        max_distance=args.max_distance,
    )
    counter = VehicleCounter(
        line_y_ratio=args.line_y,
        direction=args.direction,
    )
    log.info(
        "Tracking config: line_y=%.0f%%, direction=%s, max_distance=%.0f, max_disappeared=%d",
        args.line_y * 100, args.direction, args.max_distance, args.max_disappeared,
    )

    # --- Configure Hailo device ---
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    with VDevice(params) as vdevice:
        log.info("Creating InferModel (async API)...")
        infer_model = vdevice.create_infer_model(args.model)
        infer_model.input().set_format_type(FormatType.UINT8)
        infer_model.output().set_format_type(FormatType.FLOAT32)

        output_shape = infer_model.output().shape
        log.info("InferModel output shape: %s", output_shape)

        with infer_model.configure() as configured_model:
            cap_w, cap_h = args.input_size
            log.info("Opening camera (source=%s, capture=%dx%d)...", args.source, cap_w, cap_h)
            cap = open_camera(args.source, cap_w, cap_h)
            if not cap.isOpened():
                log.error("Could not open camera. Check connection.")
                sys.exit(1)

            log.info("Running vehicle tracking. Press 'q' to quit.")
            frame_count = 0

            try:
                while not _shutdown:
                    ret, frame = cap.read()
                    if not ret:
                        log.warning("Failed to fetch frame — retrying...")
                        continue

                    input_data = preprocess(frame, input_h, input_w)

                    bindings = configured_model.create_bindings()
                    bindings.input().set_buffer(np.expand_dims(input_data, axis=0))
                    output_buf = np.empty([1] + list(output_shape), dtype=np.float32)
                    bindings.output().set_buffer(output_buf)

                    configured_model.wait_for_async_ready(timeout_ms=10000)
                    job = configured_model.run_async([bindings])
                    job.wait(timeout_ms=10000)

                    output = output_buf[0]

                    if has_nms:
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
                            input_h, input_w,
                            args.confidence, args.iou,
                        )

                    # Filter to vehicles only (unless --all-classes)
                    if args.all_classes:
                        vehicle_dets = detections
                    else:
                        vehicle_dets = [d for d in detections if d[5] in VEHICLE_CLASS_IDS]

                    # Track
                    tracked = tracker.update(vehicle_dets)

                    # Count
                    crossed = counter.update(tracked, frame.shape[0])

                    # Draw
                    frame = draw_tracking(frame, tracked, labels, counter, crossed)

                    frame_count += 1
                    if frame_count % 30 == 0:
                        fps = cap.get(cv2.CAP_PROP_FPS) or 0
                        log.info(
                            "Tracked: %d | Count: %d | FPS: %.1f",
                            len(tracked), counter.total, fps,
                        )

                    if args.display_size:
                        dw, dh = args.display_size
                        show = cv2.resize(frame, (dw, dh)) if (frame.shape[1], frame.shape[0]) != (dw, dh) else frame
                        cv2.imshow("Vehicle Tracking - Hailo-10H", show)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
            finally:
                cap.release()
                if args.display_size:
                    cv2.destroyAllWindows()
                log.info("Final count: %d vehicles", counter.total)


def main() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    default_model = os.path.expanduser("~/hailo_models/yolov11n.hef")

    parser = argparse.ArgumentParser(
        description="Track and count vehicles using YOLO on Hailo-10H NPU"
    )
    parser.add_argument("--model", default=default_model, help="Path to YOLO .hef model")
    parser.add_argument("--labels", default="", help="Path to class labels file")
    parser.add_argument(
        "--source", default="/dev/video0",
        help="V4L2 device or 'picam' for Pi Camera",
    )
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold (default: 0.3, lower than detection for stable tracking)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument(
        "--all-classes", action="store_true",
        help="Track all detected objects, not just vehicles (useful for testing)",
    )

    # Tracking parameters
    parser.add_argument(
        "--line-y", type=float, default=0.5,
        help="Counting line Y position as fraction of frame height (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--direction", choices=["down", "up", "both"], default="down",
        help="Count vehicles going down, up, or both (default: down)",
    )
    parser.add_argument(
        "--max-disappeared", type=int, default=50,
        help="Frames before a lost track is removed (default: 50)",
    )
    parser.add_argument(
        "--max-distance", type=float, default=200.0,
        help="Max pixel distance for centroid matching (default: 200)",
    )

    # Input resolution
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--input-small", action="store_const", dest="input_size", const=(640, 480))
    input_group.add_argument("--input", action="store_const", dest="input_size", const=(1024, 768))
    input_group.add_argument("--input-large", action="store_const", dest="input_size", const=(1280, 720))

    # Display resolution
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument("--display-small", action="store_const", dest="display_size", const=(640, 480))
    display_group.add_argument("--display", action="store_const", dest="display_size", const=(1024, 768))
    display_group.add_argument("--display-large", action="store_const", dest="display_size", const=(1280, 720))

    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.input_size is None:
        args.input_size = (640, 480)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run(args)


if __name__ == "__main__":
    main()
