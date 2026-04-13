#!/usr/bin/env python3
"""
Security camera: person tracking with line-crossing alerts.

Tracks persons using YOLO on Hailo-10H NPU or locally (CPU/MPS).
When a person crosses a configurable line, a REST API call is triggered
(mock by default).

Usage:
    # Raspberry Pi + Hailo
    python person_line_alert.py --display
    python person_line_alert.py --display --line-y 0.6
    python person_line_alert.py --display --webhook-url http://my-server/api/alert

    # MacBook / laptop
    python person_line_alert.py --model yolo11n.pt --source 0 --display
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from collections import OrderedDict
from urllib.request import Request, urlopen
from urllib.error import URLError

import cv2
import numpy as np

# Add parent dir so we can import hailo_common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hailo_common import (
    ThreadedCamera,
    create_session,
    default_model,
    default_source,
    load_labels,
    open_camera,
)

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
log = logging.getLogger(__name__)

_shutdown = False

PERSON_CLASS_ID = 0  # COCO class 0 = "person"


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True


# ---------------------------------------------------------------------------
# Mock / real webhook
# ---------------------------------------------------------------------------

def mock_webhook(event: dict) -> None:
    """Simulate a REST API call by logging the payload."""
    log.info(
        "MOCK ALERT >>> POST /api/alert  payload=%s",
        json.dumps(event, indent=2),
    )


def send_webhook(url: str, event: dict) -> None:
    """Fire a real HTTP POST with JSON payload (non-blocking)."""
    def _post():
        try:
            data = json.dumps(event).encode("utf-8")
            req = Request(url, data=data, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=5) as resp:
                log.info("Webhook response: %d %s", resp.status, resp.reason)
        except URLError as exc:
            log.error("Webhook failed: %s", exc)

    threading.Thread(target=_post, daemon=True).start()


def fire_alert(
    track_id: int,
    direction: str,
    bbox: tuple,
    webhook_url: str | None,
) -> None:
    """Build event payload and dispatch via mock or real webhook."""
    event = {
        "event": "person_line_crossed",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "track_id": track_id,
        "direction": direction,
        "bbox": {"x1": int(bbox[0]), "y1": int(bbox[1]), "x2": int(bbox[2]), "y2": int(bbox[3])},
    }

    if webhook_url:
        send_webhook(webhook_url, event)
    else:
        mock_webhook(event)


# ---------------------------------------------------------------------------
# Tracker (reused IoU-based approach from run_yolo11_tracking)
# ---------------------------------------------------------------------------

class PersonTracker:
    """IoU-based tracker tuned for persons."""

    def __init__(self, max_disappeared: int = 60, min_iou: float = 0.20, max_distance: float = 150.0):
        self.next_id = 0
        self.tracks: OrderedDict[int, dict] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.min_iou = min_iou
        self.max_distance = max_distance

    def _register(self, det: tuple) -> int:
        obj_id = self.next_id
        self.tracks[obj_id] = {
            "bbox": det[:4],
            "det": det,
            "vx": 0.0, "vy": 0.0,
            "disappeared": 0,
        }
        self.next_id += 1
        return obj_id

    def _predict_bbox(self, track: dict) -> tuple:
        x1, y1, x2, y2 = track["bbox"]
        vx, vy = track["vx"], track["vy"]
        return (x1 + vx, y1 + vy, x2 + vx, y2 + vy)

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        xa, ya = max(a[0], b[0]), max(a[1], b[1])
        xb, yb = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections: list[tuple]) -> dict[int, tuple]:
        if not detections:
            for tid in list(self.tracks):
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]
            return {}

        if not self.tracks:
            return {self._register(d): d for d in detections}

        track_ids = list(self.tracks)
        predicted = [self._predict_bbox(self.tracks[tid]) for tid in track_ids]

        # IoU matrix
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        for r, pred_box in enumerate(predicted):
            for c, det in enumerate(detections):
                iou_matrix[r, c] = self._iou(pred_box, det[:4])

        matched_tracks, matched_dets, result = set(), set(), {}

        # Greedy IoU matching
        while True:
            if iou_matrix.size == 0:
                break
            max_val = iou_matrix.max()
            if max_val < self.min_iou:
                break
            r, c = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            self._assign(track_ids[r], detections[c], result)
            matched_tracks.add(r)
            matched_dets.add(c)
            iou_matrix[r, :] = 0
            iou_matrix[:, c] = 0

        # Centroid distance fallback
        um_t = [r for r in range(len(track_ids)) if r not in matched_tracks]
        um_d = [c for c in range(len(detections)) if c not in matched_dets]

        if um_t and um_d:
            pc = np.array([((predicted[r][0]+predicted[r][2])/2, (predicted[r][1]+predicted[r][3])/2) for r in um_t])
            dc = np.array([((detections[c][0]+detections[c][2])/2, (detections[c][1]+detections[c][3])/2) for c in um_d])
            dist = np.linalg.norm(pc[:, None] - dc[None, :], axis=2)
            while True:
                if dist.size == 0:
                    break
                min_d = dist.min()
                if min_d > self.max_distance:
                    break
                ri, ci = np.unravel_index(dist.argmin(), dist.shape)
                self._assign(track_ids[um_t[ri]], detections[um_d[ci]], result)
                matched_tracks.add(um_t[ri])
                matched_dets.add(um_d[ci])
                dist[ri, :] = np.inf
                dist[:, ci] = np.inf

        # Age unmatched tracks
        for r, tid in enumerate(track_ids):
            if r not in matched_tracks:
                self.tracks[tid]["disappeared"] += 1
                self.tracks[tid]["bbox"] = self._predict_bbox(self.tracks[tid])
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]

        # Register new detections
        for c, det in enumerate(detections):
            if c not in matched_dets:
                tid = self._register(det)
                result[tid] = det

        return result

    def _assign(self, tid: int, det: tuple, result: dict) -> None:
        old = self.tracks[tid]["bbox"]
        alpha = 0.4
        self.tracks[tid]["vx"] = alpha * (((det[0]+det[2])-(old[0]+old[2]))/2) + (1-alpha) * self.tracks[tid]["vx"]
        self.tracks[tid]["vy"] = alpha * (((det[1]+det[3])-(old[1]+old[3]))/2) + (1-alpha) * self.tracks[tid]["vy"]
        self.tracks[tid]["bbox"] = det[:4]
        self.tracks[tid]["det"] = det
        self.tracks[tid]["disappeared"] = 0
        result[tid] = det


# ---------------------------------------------------------------------------
# Line-crossing detector
# ---------------------------------------------------------------------------

class LineCrossingDetector:
    """Detect when tracked persons cross a horizontal line."""

    def __init__(self, line_y_ratio: float = 0.5, direction: str = "both", margin: int = 30):
        self.line_y_ratio = line_y_ratio
        self.direction = direction
        self.margin = margin
        self.prev_cy: dict[int, float] = {}
        self.alerted_ids: set[int] = set()
        self.count_down = 0
        self.count_up = 0

    @property
    def total(self) -> int:
        if self.direction == "both":
            return self.count_down + self.count_up
        return self.count_down if self.direction == "down" else self.count_up

    def update(self, tracked: dict[int, tuple], frame_h: int) -> list[tuple[int, str]]:
        """Returns list of (track_id, "up"|"down") for persons that just crossed."""
        line_y = int(self.line_y_ratio * frame_h)
        zone_top = line_y - self.margin
        zone_bot = line_y + self.margin
        crossed = []

        for tid, det in tracked.items():
            cy = (det[1] + det[3]) / 2.0

            if tid in self.prev_cy and tid not in self.alerted_ids:
                prev = self.prev_cy[tid]

                if prev < zone_top and cy >= zone_top and self.direction in ("down", "both"):
                    self.count_down += 1
                    self.alerted_ids.add(tid)
                    crossed.append((tid, "down"))
                elif prev > zone_bot and cy <= zone_bot and self.direction in ("up", "both"):
                    self.count_up += 1
                    self.alerted_ids.add(tid)
                    crossed.append((tid, "up"))

            self.prev_cy[tid] = cy

        # Cleanup stale
        active = set(tracked)
        for tid in [t for t in self.prev_cy if t not in active]:
            del self.prev_cy[tid]

        return crossed


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_overlay(
    frame: np.ndarray,
    tracked: dict[int, tuple],
    labels: list[str],
    detector: LineCrossingDetector,
    crossed_ids: set[int],
) -> np.ndarray:
    h, w = frame.shape[:2]
    line_y = int(detector.line_y_ratio * h)
    zone_top = line_y - detector.margin
    zone_bot = line_y + detector.margin

    # Draw trip-wire zone
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, zone_top), (w, zone_bot), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 2)

    for tid, det in tracked.items():
        x1, y1, x2, y2, conf, cls_id = det
        color = (0, 255, 255) if tid in crossed_ids else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"#{tid} person {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 4, color, -1)

    # Status bar
    if detector.direction == "both":
        txt = f"Down: {detector.count_down}  Up: {detector.count_up}  Total: {detector.total}"
    else:
        txt = f"Alerts: {detector.total}"
    cv2.rectangle(frame, (10, 10), (350, 50), (0, 0, 0), -1)
    cv2.putText(frame, txt, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return frame


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    tracker = PersonTracker(
        max_disappeared=args.max_disappeared,
        min_iou=args.min_iou,
        max_distance=args.max_distance,
    )
    detector = LineCrossingDetector(
        line_y_ratio=args.line_y,
        direction=args.direction,
        margin=args.line_margin,
    )

    log.info(
        "Person tracking: line_y=%.0f%%, direction=%s, webhook=%s",
        args.line_y * 100,
        args.direction,
        args.webhook_url or "MOCK",
    )

    with create_session(args.model) as session:
        labels = load_labels(args.labels) if args.labels else session.labels

        cap_w, cap_h = args.input_size
        log.info("Opening camera (source=%s, capture=%dx%d)...", args.source, cap_w, cap_h)
        raw_cap = open_camera(args.source, cap_w, cap_h)
        if not raw_cap.isOpened():
            log.error("Could not open camera.")
            sys.exit(1)
        cap = ThreadedCamera(raw_cap)

        log.info("Running person line-crossing detection. Press 'q' to quit.")
        frame_count = 0
        recently_crossed: set[int] = set()

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

                # Filter to persons only
                person_dets = [d for d in detections if d[5] == PERSON_CLASS_ID]

                tracked = tracker.update(person_dets)
                crossed = detector.update(tracked, frame.shape[0])

                # Fire alerts for crossings
                for tid, direction in crossed:
                    det = tracked[tid]
                    fire_alert(tid, direction, det[:4], args.webhook_url)
                    recently_crossed.add(tid)

                frame = draw_overlay(frame, tracked, labels, detector, recently_crossed)

                # Clear highlight after a few frames
                frame_count += 1
                if frame_count % 15 == 0:
                    recently_crossed.clear()

                if frame_count % 30 == 0:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0
                    log.info("Tracked: %d | Alerts: %d | FPS: %.1f", len(tracked), detector.total, fps)

                if args.display_size:
                    dw, dh = args.display_size
                    show = cv2.resize(frame, (dw, dh)) if (frame.shape[1], frame.shape[0]) != (dw, dh) else frame
                    cv2.imshow("Person Line-Crossing Alert", show)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if args.display_size:
                cv2.destroyAllWindows()
            log.info("Final alert count: %d", detector.total)


def main() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    parser = argparse.ArgumentParser(
        description="Person tracking with line-crossing alerts (Hailo NPU or local CPU/MPS)"
    )
    parser.add_argument("--model", default=default_model(),
                        help="Path to model: .hef (Hailo), .pt or .onnx (Ultralytics)")
    parser.add_argument("--labels", default="", help="Path to class labels file")
    parser.add_argument("--source", default=default_source(),
                        help="Camera source: device index (0, 1), V4L2 path, or 'picam'")
    parser.add_argument("--confidence", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")

    # Line-crossing
    parser.add_argument("--line-y", type=float, default=0.5, help="Line Y position (0.0-1.0)")
    parser.add_argument("--line-margin", type=int, default=30, help="Half-height of crossing zone (px)")
    parser.add_argument("--direction", choices=["down", "up", "both"], default="both", help="Alert direction")

    # Tracker tuning
    parser.add_argument("--max-disappeared", type=int, default=60)
    parser.add_argument("--min-iou", type=float, default=0.20)
    parser.add_argument("--max-distance", type=float, default=150.0)

    # Webhook
    parser.add_argument(
        "--webhook-url", default=None,
        help="REST endpoint to POST alerts to. If omitted, alerts are logged (mock mode).",
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

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.input_size is None:
        args.input_size = (640, 480)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run(args)


if __name__ == "__main__":
    main()
