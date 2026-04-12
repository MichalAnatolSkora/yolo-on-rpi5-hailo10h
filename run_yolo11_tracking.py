#!/usr/bin/env python3
"""
Vehicle tracking and counting using YOLO (Hailo NPU or local CPU/MPS).

Extends run_yolo11.py with:
- IoU-based object tracking across frames.
- A configurable counting line — vehicles that cross it are counted.
- Filters for vehicle classes only (car, motorcycle, bus, truck).

Usage:
    # Raspberry Pi + Hailo-10H
    python run_yolo11_tracking.py --display
    python run_yolo11_tracking.py --display --source /dev/video0

    # MacBook / laptop
    python run_yolo11_tracking.py --model yolo11n.pt --source 0 --display
"""

import argparse
import logging
import signal
import sys
from collections import OrderedDict

import cv2
import numpy as np

from hailo_common import (
    ThreadedCamera, create_session, default_model, default_source,
    load_labels, open_camera,
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
    """Count vehicles entering a horizontal counting zone."""

    def __init__(self, line_y_ratio: float = 0.5, direction: str = "down", margin: int = 40):
        self.line_y_ratio = line_y_ratio
        self.direction = direction
        self.margin = margin
        self.prev_centroids: dict[int, float] = {}
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
        line_y = int(self.line_y_ratio * frame_h)
        zone_top = line_y - self.margin
        zone_bot = line_y + self.margin
        crossed = []

        for track_id, det in tracked.items():
            cy = (det[1] + det[3]) / 2.0

            if track_id in self.prev_centroids and track_id not in self.counted_ids:
                prev_cy = self.prev_centroids[track_id]

                if prev_cy < zone_top and cy >= zone_top:
                    if self.direction in ("down", "both"):
                        self.count_down += 1
                        self.counted_ids.add(track_id)
                        crossed.append(track_id)

                elif prev_cy > zone_bot and cy <= zone_bot:
                    if self.direction in ("up", "both"):
                        self.count_up += 1
                        self.counted_ids.add(track_id)
                        crossed.append(track_id)

            self.prev_centroids[track_id] = cy

        active_ids = set(tracked.keys())
        stale = [tid for tid in self.prev_centroids if tid not in active_ids]
        for tid in stale:
            del self.prev_centroids[tid]

        return crossed


def deduplicate_detections(detections: list[tuple], iou_threshold: float = 0.45) -> list[tuple]:
    """Remove overlapping detections of the same class, keeping highest confidence."""
    if len(detections) <= 1:
        return detections
    dets = sorted(detections, key=lambda d: d[4], reverse=True)
    keep = []
    for det in dets:
        suppressed = False
        for kept in keep:
            if det[5] == kept[5] and IOUTracker._iou(det[:4], kept[:4]) > iou_threshold:
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
    """Draw tracked vehicles, counting zone, and count overlay."""
    h, w = frame.shape[:2]
    line_y = int(counter.line_y_ratio * h)
    zone_top = line_y - counter.margin
    zone_bot = line_y + counter.margin

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, zone_top), (w, zone_bot), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 2)

    for track_id, det in tracked.items():
        x1, y1, x2, y2, conf, cls_id = det
        label = labels[cls_id] if cls_id < len(labels) else str(cls_id)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        color = (0, 255, 255) if track_id in crossed_ids else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"#{track_id} {label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.circle(frame, (cx, cy), 4, color, -1)

    if counter.direction == "both":
        count_text = f"Down: {counter.count_down}  Up: {counter.count_up}  Total: {counter.total}"
    else:
        count_text = f"Count: {counter.total}"

    cv2.rectangle(frame, (10, 10), (10 + 300, 50), (0, 0, 0), -1)
    cv2.putText(frame, count_text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return frame


def run(args: argparse.Namespace) -> None:
    tracker = IOUTracker(
        max_disappeared=args.max_disappeared,
        min_iou=args.min_iou,
        max_distance=args.max_distance,
    )
    counter = VehicleCounter(
        line_y_ratio=args.line_y,
        direction=args.direction,
        margin=args.line_margin,
    )
    log.info(
        "Tracking config: line_y=%.0f%%, direction=%s, min_iou=%.2f, max_dist=%.0f, max_disappeared=%d",
        args.line_y * 100, args.direction, args.min_iou, args.max_distance, args.max_disappeared,
    )

    with create_session(args.model) as session:
        labels = load_labels(args.labels) if args.labels else session.labels

        cap_w, cap_h = args.input_size
        log.info("Opening camera (source=%s, capture=%dx%d)...", args.source, cap_w, cap_h)
        raw_cap = open_camera(args.source, cap_w, cap_h)
        if not raw_cap.isOpened():
            log.error("Could not open camera. Check connection.")
            sys.exit(1)
        cap = ThreadedCamera(raw_cap)

        log.info("Running vehicle tracking. Press 'q' to quit.")
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

                # Filter to vehicles only (unless --all-classes)
                if args.all_classes:
                    vehicle_dets = detections
                else:
                    vehicle_dets = [d for d in detections if d[5] in VEHICLE_CLASS_IDS]

                if args.deduplicate:
                    vehicle_dets = deduplicate_detections(vehicle_dets, iou_threshold=args.iou)

                tracked = tracker.update(vehicle_dets)
                crossed = counter.update(tracked, frame.shape[0])
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
                    cv2.imshow("Vehicle Tracking", show)
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

    parser = argparse.ArgumentParser(
        description="Track and count vehicles using YOLO (Hailo NPU or local CPU/MPS)"
    )
    parser.add_argument("--model", default=default_model(),
                        help="Path to model: .hef (Hailo), .pt or .onnx (Ultralytics)")
    parser.add_argument("--labels", default="", help="Path to class labels file")
    parser.add_argument(
        "--source", default=default_source(),
        help="Camera source: device index (0, 1), V4L2 path, or 'picam'",
    )
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Confidence threshold (default: 0.3)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--all-classes", action="store_true",
                        help="Track all objects, not just vehicles")
    parser.add_argument("--deduplicate", action="store_true",
                        help="Remove overlapping detections before tracking")

    parser.add_argument("--line-y", type=float, default=0.5,
                        help="Counting line Y position (0.0-1.0, default: 0.5)")
    parser.add_argument("--line-margin", type=int, default=40,
                        help="Half-height of counting zone in pixels (default: 40)")
    parser.add_argument("--direction", choices=["down", "up", "both"], default="down",
                        help="Count direction (default: down)")
    parser.add_argument("--max-disappeared", type=int, default=50,
                        help="Frames before a lost track is removed (default: 50)")
    parser.add_argument("--min-iou", type=float, default=0.15,
                        help="Minimum IoU to match detection to track (default: 0.15)")
    parser.add_argument("--max-distance", type=float, default=200.0,
                        help="Max centroid distance fallback (default: 200)")

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--input-small", action="store_const", dest="input_size", const=(640, 480))
    input_group.add_argument("--input", action="store_const", dest="input_size", const=(1024, 768))
    input_group.add_argument("--input-large", action="store_const", dest="input_size", const=(1280, 720))

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
