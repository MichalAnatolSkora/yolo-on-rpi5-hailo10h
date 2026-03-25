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


class IOUTracker:
    """
    IoU-based tracker with linear velocity prediction (simplified SORT).

    Matches detections to existing tracks by IoU between the predicted
    bounding box (shifted by velocity) and the new detection. Much more
    stable than centroid distance for fast-moving objects like vehicles.
    """

    def __init__(self, max_disappeared: int = 50, min_iou: float = 0.15, max_distance: float = 200.0):
        self.next_id = 0
        self.tracks: OrderedDict[int, dict] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.min_iou = min_iou
        self.max_distance = max_distance

    def _register(self, det: tuple) -> int:
        obj_id = self.next_id
        self.tracks[obj_id] = {
            "bbox": det[:4],         # (x1, y1, x2, y2)
            "det": det,              # full detection tuple
            "vx": 0.0, "vy": 0.0,   # velocity of centroid
            "disappeared": 0,
        }
        self.next_id += 1
        return obj_id

    def _predict_bbox(self, track: dict) -> tuple:
        """Shift bbox by velocity to predict next position."""
        x1, y1, x2, y2 = track["bbox"]
        vx, vy = track["vx"], track["vy"]
        return (x1 + vx, y1 + vy, x2 + vx, y2 + vy)

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        xa = max(a[0], b[0])
        ya = max(a[1], b[1])
        xb = min(a[2], b[2])
        yb = min(a[3], b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections: list[tuple]) -> dict[int, tuple]:
        """
        Update tracks with new detections.

        Args:
            detections: list of (x1, y1, x2, y2, conf, class_id)

        Returns:
            dict of {track_id: (x1, y1, x2, y2, conf, class_id)}
        """
        # No detections — age all tracks
        if len(detections) == 0:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]
            return {}

        # No existing tracks — register all
        if len(self.tracks) == 0:
            result = {}
            for det in detections:
                tid = self._register(det)
                result[tid] = det
            return result

        track_ids = list(self.tracks.keys())
        predicted = [self._predict_bbox(self.tracks[tid]) for tid in track_ids]

        # Compute IoU matrix: tracks x detections
        det_boxes = [d[:4] for d in detections]
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        for r, pred_box in enumerate(predicted):
            for c, det_box in enumerate(det_boxes):
                iou_matrix[r, c] = self._iou(pred_box, det_box)

        # Greedy matching by highest IoU
        matched_tracks = set()
        matched_dets = set()
        result = {}

        while True:
            if iou_matrix.size == 0:
                break
            max_iou = iou_matrix.max()
            if max_iou < self.min_iou:
                break
            r, c = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)

            tid = track_ids[r]
            det = detections[c]
            old_bbox = self.tracks[tid]["bbox"]

            # Update velocity (smoothed)
            old_cx = (old_bbox[0] + old_bbox[2]) / 2
            old_cy = (old_bbox[1] + old_bbox[3]) / 2
            new_cx = (det[0] + det[2]) / 2
            new_cy = (det[1] + det[3]) / 2
            alpha = 0.4  # smoothing factor
            self.tracks[tid]["vx"] = alpha * (new_cx - old_cx) + (1 - alpha) * self.tracks[tid]["vx"]
            self.tracks[tid]["vy"] = alpha * (new_cy - old_cy) + (1 - alpha) * self.tracks[tid]["vy"]

            self.tracks[tid]["bbox"] = det[:4]
            self.tracks[tid]["det"] = det
            self.tracks[tid]["disappeared"] = 0
            result[tid] = det

            matched_tracks.add(r)
            matched_dets.add(c)

            # Zero out matched row and column
            iou_matrix[r, :] = 0
            iou_matrix[:, c] = 0

        # Fallback: match remaining tracks/detections by centroid distance
        # This catches fast-moving objects whose bboxes don't overlap between frames
        unmatched_track_idxs = [r for r in range(len(track_ids)) if r not in matched_tracks]
        unmatched_det_idxs = [c for c in range(len(detections)) if c not in matched_dets]

        if unmatched_track_idxs and unmatched_det_idxs:
            pred_centroids = np.array([
                ((predicted[r][0] + predicted[r][2]) / 2, (predicted[r][1] + predicted[r][3]) / 2)
                for r in unmatched_track_idxs
            ])
            det_centroids = np.array([
                ((detections[c][0] + detections[c][2]) / 2, (detections[c][1] + detections[c][3]) / 2)
                for c in unmatched_det_idxs
            ])
            dist_matrix = np.linalg.norm(
                pred_centroids[:, np.newaxis] - det_centroids[np.newaxis, :], axis=2
            )

            while True:
                if dist_matrix.size == 0:
                    break
                min_dist = dist_matrix.min()
                if min_dist > self.max_distance:
                    break
                ri, ci = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)

                r = unmatched_track_idxs[ri]
                c = unmatched_det_idxs[ci]
                tid = track_ids[r]
                det = detections[c]
                old_bbox = self.tracks[tid]["bbox"]

                old_cx = (old_bbox[0] + old_bbox[2]) / 2
                old_cy = (old_bbox[1] + old_bbox[3]) / 2
                new_cx = (det[0] + det[2]) / 2
                new_cy = (det[1] + det[3]) / 2
                alpha = 0.4
                self.tracks[tid]["vx"] = alpha * (new_cx - old_cx) + (1 - alpha) * self.tracks[tid]["vx"]
                self.tracks[tid]["vy"] = alpha * (new_cy - old_cy) + (1 - alpha) * self.tracks[tid]["vy"]

                self.tracks[tid]["bbox"] = det[:4]
                self.tracks[tid]["det"] = det
                self.tracks[tid]["disappeared"] = 0
                result[tid] = det

                matched_tracks.add(r)
                matched_dets.add(c)

                dist_matrix[ri, :] = np.inf
                dist_matrix[:, ci] = np.inf

        # Age unmatched tracks
        for r, tid in enumerate(track_ids):
            if r not in matched_tracks:
                self.tracks[tid]["disappeared"] += 1
                # Drift predicted bbox so next frame prediction stays reasonable
                self.tracks[tid]["bbox"] = self._predict_bbox(self.tracks[tid])
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]

        # Register unmatched detections
        for c, det in enumerate(detections):
            if c not in matched_dets:
                tid = self._register(det)
                result[tid] = det

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


def _compute_iou(box_a: tuple, box_b: tuple) -> float:
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def deduplicate_detections(detections: list[tuple], iou_threshold: float = 0.45) -> list[tuple]:
    """
    Remove overlapping detections of the same class, keeping the highest confidence.
    Prevents the tracker from seeing multiple IDs for a single object.
    """
    if len(detections) <= 1:
        return detections
    # Sort by confidence descending
    dets = sorted(detections, key=lambda d: d[4], reverse=True)
    keep = []
    for det in dets:
        suppressed = False
        for kept in keep:
            if det[5] == kept[5] and _compute_iou(det[:4], kept[:4]) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(det)
    return keep


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
    tracker = IOUTracker(
        max_disappeared=args.max_disappeared,
        min_iou=args.min_iou,
        max_distance=args.max_distance,
    )
    counter = VehicleCounter(
        line_y_ratio=args.line_y,
        direction=args.direction,
    )
    log.info(
        "Tracking config: line_y=%.0f%%, direction=%s, min_iou=%.2f, max_dist=%.0f, max_disappeared=%d",
        args.line_y * 100, args.direction, args.min_iou, args.max_distance, args.max_disappeared,
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

                    # Deduplicate overlapping boxes (keeps highest confidence per overlap group)
                    if args.deduplicate:
                        vehicle_dets = deduplicate_detections(vehicle_dets, iou_threshold=args.iou)

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
    parser.add_argument(
        "--deduplicate", action="store_true",
        help="Remove overlapping detections before tracking (helps at low confidence)",
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
        "--min-iou", type=float, default=0.15,
        help="Minimum IoU to match detection to track (default: 0.15)",
    )
    parser.add_argument(
        "--max-distance", type=float, default=200.0,
        help="Max centroid distance fallback when IoU fails (default: 200)",
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
