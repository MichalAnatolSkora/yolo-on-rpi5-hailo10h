#!/usr/bin/env python3
"""
Security camera v2: person tracking with multi-line crossing alerts.

Tracks persons using YOLO (Hailo NPU or local CPU/MPS). When a person
crosses any of the configured trip-wire lines, a REST API alert is fired.

Lines can be at any angle and are drawn interactively via --setup mode.
Configuration is stored in a JSON file so it persists between runs.

Setup:
    python person_line_alert_v2.py --setup --source 0
    python person_line_alert_v2.py --setup --source 0 --config my_lines.json

Run:
    python person_line_alert_v2.py --config line_config.json --display
    python person_line_alert_v2.py --config line_config.json --model yolo11n.pt --source 0 --display
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
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np

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
PERSON_CLASS_ID = 0

# Palette for drawing multiple lines (BGR)
LINE_COLORS = [
    (0, 0, 255),    # red
    (0, 200, 255),  # orange
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 200, 0),  # cyan-ish
    (0, 255, 0),    # green
]


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CURRENT_SCHEMA_VERSION = 1  # bump when the line config gains required fields


def load_config(path: str) -> dict:
    """Load and validate a line config JSON file (v0 or v1)."""
    if not os.path.isfile(path):
        log.error("Config file not found: %s", path)
        sys.exit(1)
    with open(path) as f:
        config = json.load(f)
    version = config.get("schema_version")
    if version is None:
        log.warning(
            "%s has no schema_version — treating as legacy (v0). "
            "Add \"schema_version\": 0 to silence this warning, or hand-edit "
            "to v1 (see tools/setup_lines.py docstring for the diff).", path,
        )
        version = 0
    if version > CURRENT_SCHEMA_VERSION:
        log.error(
            "%s has schema_version=%d but this script only understands up to v%d. "
            "Upgrade the script.", path, version, CURRENT_SCHEMA_VERSION,
        )
        sys.exit(1)
    lines = config.get("lines", [])
    if not lines:
        log.error("Config has no lines defined. Run --setup first.")
        sys.exit(1)

    active = [ln for ln in lines if ln.get("enabled", True)]
    skipped = len(lines) - len(active)
    if skipped:
        log.info("Skipped %d disabled line(s).", skipped)
    if not active:
        log.error("All lines are disabled in %s.", path)
        sys.exit(1)
    config["lines"] = active

    for i, line in enumerate(active):
        for key in ("name", "p1", "p2"):
            if key not in line:
                log.error("Line %d missing '%s' field.", i, key)
                sys.exit(1)
        line.setdefault("direction", "both")

    # v1 puts webhook_url under "alerts"; v0 has it at root. Normalize so the
    # rest of the script can read config["webhook_url"] regardless.
    if version >= 1 and "alerts" in config:
        config.setdefault("webhook_url", config["alerts"].get("webhook_url", ""))

    return config


def save_config(path: str, config: dict) -> None:
    """Save line config to JSON."""
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    log.info("Config saved to: %s", path)


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------

def mock_webhook(event: dict) -> None:
    log.info("MOCK ALERT >>> POST /api/alert  payload=%s", json.dumps(event, indent=2))


def send_webhook(url: str, event: dict) -> None:
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
    track_id: int, direction: str, bbox: tuple,
    line_name: str, webhook_url: str | None,
) -> None:
    event = {
        "event": "person_line_crossed",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "track_id": track_id,
        "line_name": line_name,
        "direction": direction,
        "bbox": {"x1": int(bbox[0]), "y1": int(bbox[1]),
                 "x2": int(bbox[2]), "y2": int(bbox[3])},
    }
    if webhook_url:
        send_webhook(webhook_url, event)
    else:
        mock_webhook(event)


# ---------------------------------------------------------------------------
# Person tracker (same as v1)
# ---------------------------------------------------------------------------

class PersonTracker:
    def __init__(self, max_disappeared=60, min_iou=0.20, max_distance=150.0):
        self.next_id = 0
        self.tracks: OrderedDict[int, dict] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.min_iou = min_iou
        self.max_distance = max_distance

    def _register(self, det):
        oid = self.next_id
        self.tracks[oid] = {"bbox": det[:4], "det": det, "vx": 0.0, "vy": 0.0, "disappeared": 0}
        self.next_id += 1
        return oid

    def _predict_bbox(self, t):
        x1, y1, x2, y2 = t["bbox"]
        return (x1+t["vx"], y1+t["vy"], x2+t["vx"], y2+t["vy"])

    @staticmethod
    def _iou(a, b):
        xa, ya = max(a[0], b[0]), max(a[1], b[1])
        xb, yb = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xb-xa) * max(0, yb-ya)
        aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
        ab = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
        u = aa + ab - inter
        return inter / u if u > 0 else 0.0

    def _assign(self, tid, det, result):
        old = self.tracks[tid]["bbox"]
        a = 0.4
        self.tracks[tid]["vx"] = a*(((det[0]+det[2])-(old[0]+old[2]))/2) + (1-a)*self.tracks[tid]["vx"]
        self.tracks[tid]["vy"] = a*(((det[1]+det[3])-(old[1]+old[3]))/2) + (1-a)*self.tracks[tid]["vy"]
        self.tracks[tid]["bbox"] = det[:4]
        self.tracks[tid]["det"] = det
        self.tracks[tid]["disappeared"] = 0
        result[tid] = det

    def update(self, detections):
        if not detections:
            for tid in list(self.tracks):
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]
            return {}
        if not self.tracks:
            return {self._register(d): d for d in detections}

        tids = list(self.tracks)
        pred = [self._predict_bbox(self.tracks[t]) for t in tids]
        iou_m = np.zeros((len(tids), len(detections)))
        for r, pb in enumerate(pred):
            for c, d in enumerate(detections):
                iou_m[r, c] = self._iou(pb, d[:4])

        mt, md, res = set(), set(), {}
        while True:
            if iou_m.size == 0 or iou_m.max() < self.min_iou:
                break
            r, c = np.unravel_index(iou_m.argmax(), iou_m.shape)
            self._assign(tids[r], detections[c], res)
            mt.add(r); md.add(c)
            iou_m[r, :] = 0; iou_m[:, c] = 0

        um_t = [r for r in range(len(tids)) if r not in mt]
        um_d = [c for c in range(len(detections)) if c not in md]
        if um_t and um_d:
            pc = np.array([((pred[r][0]+pred[r][2])/2, (pred[r][1]+pred[r][3])/2) for r in um_t])
            dc = np.array([((detections[c][0]+detections[c][2])/2, (detections[c][1]+detections[c][3])/2) for c in um_d])
            dist = np.linalg.norm(pc[:, None] - dc[None, :], axis=2)
            while True:
                if dist.size == 0 or dist.min() > self.max_distance:
                    break
                ri, ci = np.unravel_index(dist.argmin(), dist.shape)
                self._assign(tids[um_t[ri]], detections[um_d[ci]], res)
                mt.add(um_t[ri]); md.add(um_d[ci])
                dist[ri, :] = np.inf; dist[:, ci] = np.inf

        for r, tid in enumerate(tids):
            if r not in mt:
                self.tracks[tid]["disappeared"] += 1
                self.tracks[tid]["bbox"] = self._predict_bbox(self.tracks[tid])
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]
        for c, d in enumerate(detections):
            if c not in md:
                res[self._register(d)] = d
        return res


# ---------------------------------------------------------------------------
# Multi-line crossing detector (cross-product based)
# ---------------------------------------------------------------------------

class MultiLineCrossingDetector:
    """Detect when tracked persons cross any of the configured trip-wire lines.

    Lines are defined as two normalized points (0.0-1.0). Crossing is detected
    via cross-product sign change -- works at any angle.
    """

    def __init__(self, lines_config: list[dict], buffer_px: int = 0):
        self.lines = lines_config
        self.buffer_px = buffer_px
        self.prev_centroids: dict[int, tuple[float, float]] = {}
        self.alerted: dict[int, set[str]] = {}  # track_id -> set of line names
        self.counts: dict[str, dict[str, int]] = {
            line["name"]: {"positive": 0, "negative": 0} for line in lines_config
        }

    @staticmethod
    def _cross(p1, p2, point):
        """Sign of (p2-p1) x (point-p1). Positive = left side, negative = right."""
        return (p2[0]-p1[0]) * (point[1]-p1[1]) - (p2[1]-p1[1]) * (point[0]-p1[0])

    def total(self, line_name: str | None = None) -> int:
        if line_name:
            c = self.counts.get(line_name, {})
            return c.get("positive", 0) + c.get("negative", 0)
        return sum(c["positive"] + c["negative"] for c in self.counts.values())

    def update(
        self, tracked: dict[int, tuple], frame_w: int, frame_h: int
    ) -> list[tuple[int, str, str]]:
        """Returns list of (track_id, line_name, "positive"|"negative")."""
        crossings = []

        for tid, det in tracked.items():
            cx = (det[0] + det[2]) / 2.0
            cy = (det[1] + det[3]) / 2.0
            curr = (cx, cy)

            if tid in self.prev_centroids:
                prev = self.prev_centroids[tid]
                for line in self.lines:
                    name = line["name"]
                    if name in self.alerted.get(tid, set()):
                        continue

                    # Convert normalized coords to pixels
                    p1 = (line["p1"][0] * frame_w, line["p1"][1] * frame_h)
                    p2 = (line["p2"][0] * frame_w, line["p2"][1] * frame_h)

                    s_prev = self._cross(p1, p2, prev)
                    s_curr = self._cross(p1, p2, curr)

                    crossed = False
                    if s_prev * s_curr < 0:  # exact sign change = crossing
                        crossed = True
                    elif self.buffer_px > 0:
                        # Perpendicular distance to line = |cross| / line_length
                        line_len = max(1.0, ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) ** 0.5)
                        dist_curr = abs(s_curr) / line_len
                        dist_prev = abs(s_prev) / line_len
                        if dist_curr < self.buffer_px and dist_prev >= self.buffer_px:
                            crossed = True

                    if crossed:
                        # Direction = side they came from → side they're heading to
                        direction = "positive" if s_prev < 0 else "negative"
                        want = line.get("direction", "both")
                        if want == "both" or want == direction:
                            crossings.append((tid, name, direction))
                            self.alerted.setdefault(tid, set()).add(name)
                            self.counts[name][direction] += 1

            self.prev_centroids[tid] = curr

        # Cleanup stale tracks
        active = set(tracked)
        for tid in [t for t in self.prev_centroids if t not in active]:
            del self.prev_centroids[tid]
            self.alerted.pop(tid, None)

        return crossings


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_buffer_zone(frame: np.ndarray, p1: tuple, p2: tuple, buffer_px: int, color: tuple) -> None:
    """Draw a semi-transparent buffer strip around a line."""
    if buffer_px <= 0:
        return
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    length = max(1, (dx*dx + dy*dy) ** 0.5)
    nx, ny = -dy/length * buffer_px, dx/length * buffer_px
    pts = np.array([
        [int(p1[0]+nx), int(p1[1]+ny)],
        [int(p2[0]+nx), int(p2[1]+ny)],
        [int(p2[0]-nx), int(p2[1]-ny)],
        [int(p1[0]-nx), int(p1[1]-ny)],
    ], dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)


def draw_overlay(
    frame: np.ndarray,
    tracked: dict[int, tuple],
    lines_config: list[dict],
    detector: MultiLineCrossingDetector,
    recently_crossed: dict[int, str],
) -> np.ndarray:
    h, w = frame.shape[:2]

    # Draw trip-wire lines
    for i, line in enumerate(lines_config):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        p1 = (int(line["p1"][0] * w), int(line["p1"][1] * h))
        p2 = (int(line["p2"][0] * w), int(line["p2"][1] * h))

        _draw_buffer_zone(frame, p1, p2, detector.buffer_px, color)
        cv2.line(frame, p1, p2, color, 2)

        # Label at midpoint
        mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
        count = detector.total(line["name"])
        label = f"{line['name']} ({count})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (mid[0]-2, mid[1]-th-4), (mid[0]+tw+2, mid[1]+4), (0, 0, 0), -1)
        cv2.putText(frame, label, (mid[0], mid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Direction arrow (perpendicular to line, pointing "positive" side)
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        length = max(1, (dx*dx + dy*dy) ** 0.5)
        nx, ny = -dy/length * 25, dx/length * 25  # normal, 25px long
        arrow_start = mid
        arrow_end = (int(mid[0]+nx), int(mid[1]+ny))
        cv2.arrowedLine(frame, arrow_start, arrow_end, color, 2, tipLength=0.4)

    # Draw tracked persons
    for tid, det in tracked.items():
        x1, y1, x2, y2, conf, _ = det
        crossed_line = recently_crossed.get(tid)
        if crossed_line:
            # Find color of the line that was crossed
            color = (0, 255, 255)
            for i, line in enumerate(lines_config):
                if line["name"] == crossed_line:
                    color = LINE_COLORS[i % len(LINE_COLORS)]
                    break
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"#{tid} person {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1-th-6), (x1+tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        cv2.circle(frame, (cx, cy), 4, color, -1)

    # Status bar
    cv2.rectangle(frame, (10, 10), (350, 50), (0, 0, 0), -1)
    cv2.putText(
        frame, f"Total alerts: {detector.total()}", (15, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
    )

    return frame


# ---------------------------------------------------------------------------
# Interactive setup mode
# ---------------------------------------------------------------------------

def run_setup(args: argparse.Namespace) -> None:
    config_path = args.config or "line_config.json"

    # Load existing config if present
    config: dict = {"schema_version": CURRENT_SCHEMA_VERSION, "lines": [], "webhook_url": ""}
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
        config.setdefault("schema_version", CURRENT_SCHEMA_VERSION)
        log.info("Loaded existing config with %d lines from %s", len(config.get("lines", [])), config_path)

    # Grab a frame
    cap_w, cap_h = args.input_size
    cap = open_camera(args.source, cap_w, cap_h)
    if not cap.isOpened():
        log.error("Could not open camera.")
        sys.exit(1)

    # Read a few frames to let the camera warm up
    for _ in range(10):
        ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        log.error("Could not capture a frame from camera.")
        sys.exit(1)

    base_frame = frame.copy()
    h, w = frame.shape[:2]
    points: list[tuple[int, int]] = []
    line_count = len(config.get("lines", []))

    def redraw():
        display = base_frame.copy()
        # Draw existing lines
        for i, line in enumerate(config["lines"]):
            color = LINE_COLORS[i % len(LINE_COLORS)]
            p1 = (int(line["p1"][0] * w), int(line["p1"][1] * h))
            p2 = (int(line["p2"][0] * w), int(line["p2"][1] * h))
            _draw_buffer_zone(display, p1, p2, args.buffer, color)
            cv2.line(display, p1, p2, color, 2)
            mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
            cv2.putText(display, line["name"], (mid[0], mid[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Direction arrow (perpendicular, pointing "positive" side)
            dx, dy = p2[0]-p1[0], p2[1]-p1[1]
            length = max(1, (dx*dx + dy*dy) ** 0.5)
            nx, ny = -dy/length * 25, dx/length * 25
            arrow_end = (int(mid[0]+nx), int(mid[1]+ny))
            cv2.arrowedLine(display, mid, arrow_end, color, 2, tipLength=0.4)
        # Draw in-progress point
        if len(points) == 1:
            cv2.circle(display, points[0], 6, (0, 255, 0), -1)
            cv2.putText(display, "Click second point...", (10, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Instructions
        n = len(config["lines"])
        cv2.putText(display, f"Lines: {n} | Click to add | Enter=save | u=undo | q=quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow("Line Setup", display)

    def mouse_callback(event, x, y, flags, param):
        nonlocal points, line_count
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        points.append((x, y))
        if len(points) == 2:
            line_count += 1
            name = f"line_{line_count}"
            config["lines"].append({
                "name": name,
                "p1": [round(points[0][0] / w, 4), round(points[0][1] / h, 4)],
                "p2": [round(points[1][0] / w, 4), round(points[1][1] / h, 4)],
                "direction": "both",
            })
            log.info("Added '%s': (%.3f, %.3f) -> (%.3f, %.3f)",
                     name, *config["lines"][-1]["p1"], *config["lines"][-1]["p2"])
            points.clear()
        redraw()

    cv2.namedWindow("Line Setup")
    cv2.setMouseCallback("Line Setup", mouse_callback)
    redraw()

    print()
    print("=== Line Setup Mode ===")
    print("  Click two points on the image to define a trip-wire line.")
    print("  Each line is auto-named (edit names in the JSON later).")
    print("  Keys: Enter=save  u=undo last line  Esc=clear current  q=quit without saving")
    print()

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 13:  # Enter
            if not config["lines"]:
                log.warning("No lines defined. Add at least one line before saving.")
                continue
            save_config(config_path, config)
            print(f"\nSaved {len(config['lines'])} line(s) to {config_path}")
            print(f"Run detection with:  python {__file__} --config {config_path} --display")
            break
        elif key == 27:  # Esc -- clear in-progress
            points.clear()
            redraw()
        elif key == ord("u"):
            if config["lines"]:
                removed = config["lines"].pop()
                log.info("Undid line '%s'", removed["name"])
                redraw()
        elif key == ord("q"):
            log.info("Quit without saving.")
            break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main detection loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    global_webhook = args.webhook_url or config.get("webhook_url") or None

    # Per-line webhook lookup: --webhook-url > line.webhook_url > global > MOCK.
    # CLI flag is a hard override (used for testing), so it wins over per-line.
    line_webhooks: dict[str, str | None] = {}
    for ln in config["lines"]:
        if args.webhook_url:
            line_webhooks[ln["name"]] = args.webhook_url
        else:
            line_webhooks[ln["name"]] = ln.get("webhook_url") or global_webhook

    tracker = PersonTracker(
        max_disappeared=args.max_disappeared,
        min_iou=args.min_iou,
        max_distance=args.max_distance,
    )
    detector = MultiLineCrossingDetector(config["lines"], buffer_px=args.buffer)

    line_names = [l["name"] for l in config["lines"]]
    log.info("Loaded %d trip-wire line(s): %s (buffer=%dpx)", len(config["lines"]), ", ".join(line_names), args.buffer)
    distinct = {url or "MOCK" for url in line_webhooks.values()}
    if len(distinct) == 1:
        log.info("Webhook: %s", next(iter(distinct)))
    else:
        log.info("Webhooks (per line): %s",
                 ", ".join(f"{n}={url or 'MOCK'}" for n, url in line_webhooks.items()))

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
        recently_crossed: dict[int, str] = {}  # track_id -> line_name

        try:
            while not _shutdown:
                ret, frame = cap.read()
                if not ret:
                    continue

                detections = session.detect(
                    frame,
                    conf_threshold=args.confidence,
                    iou_threshold=args.iou,
                    num_classes=len(labels),
                )
                person_dets = [d for d in detections if d[5] == PERSON_CLASS_ID]

                tracked = tracker.update(person_dets)
                h, w = frame.shape[:2]
                crossings = detector.update(tracked, w, h)

                for tid, line_name, direction in crossings:
                    det = tracked[tid]
                    fire_alert(tid, direction, det[:4], line_name, line_webhooks.get(line_name))
                    recently_crossed[tid] = line_name

                frame = draw_overlay(frame, tracked, config["lines"], detector, recently_crossed)

                frame_count += 1
                if frame_count % 15 == 0:
                    recently_crossed.clear()
                if frame_count % 30 == 0:
                    log.info("Tracked: %d | Alerts: %d", len(tracked), detector.total())

                if args.display_size:
                    dw, dh = args.display_size
                    show = cv2.resize(frame, (dw, dh)) if (frame.shape[1], frame.shape[0]) != (dw, dh) else frame
                    cv2.imshow("Person Line-Crossing Alert v2", show)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if args.display_size:
                cv2.destroyAllWindows()
            log.info("Final alert count: %d", detector.total())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    parser = argparse.ArgumentParser(
        description="Person tracking with multi-line crossing alerts (v2)"
    )

    # Mode
    parser.add_argument("--setup", action="store_true",
                        help="Interactive mode: draw trip-wire lines on camera frame")
    parser.add_argument("--config", default=None,
                        help="Path to line config JSON (created by --setup, required for detection)")

    # Model / camera
    parser.add_argument("--model", default=default_model())
    parser.add_argument("--labels", default="")
    parser.add_argument("--source", default=default_source())
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.45)

    # Tracker
    parser.add_argument("--max-disappeared", type=int, default=60)
    parser.add_argument("--min-iou", type=float, default=0.20)
    parser.add_argument("--max-distance", type=float, default=150.0)

    # Buffer zone (pixels) — person counted when entering this distance from line
    parser.add_argument("--buffer", type=int, default=0,
                        help="Buffer zone in pixels around each line (0 = exact crossing)")

    # Webhook (overrides config value)
    parser.add_argument("--webhook-url", default=None)

    # Resolution
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--input-small", action="store_const", dest="input_size", const=(640, 480))
    input_group.add_argument("--input", action="store_const", dest="input_size", const=(1024, 768))
    input_group.add_argument("--input-large", action="store_const", dest="input_size", const=(1280, 720))

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

    if args.setup:
        run_setup(args)
    else:
        if not args.config:
            parser.error("--config is required for detection mode. Run --setup first to create one.")
        run(args)


if __name__ == "__main__":
    main()
