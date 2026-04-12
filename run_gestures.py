#!/usr/bin/env python3
"""
Real-time hand gesture recognition with action triggers.

Works on Hailo-10H NPU (.hef) or locally via Ultralytics (.pt / .onnx).

Detects hand gestures and maps them to configurable actions (shell commands,
on-screen messages). Supports gesture hold detection, cooldowns, gesture
history, and multi-hand tracking.

Usage:
    # Raspberry Pi + Hailo-10H
    python run_gestures.py --model ~/hailo_models/yolov12n_gestures.hef

    # MacBook / laptop
    python run_gestures.py --model yolov12n_gestures.pt --source 0 --display

Prerequisites:
    - OpenCV, NumPy, PyYAML
    - For .pt/.onnx: pip install ultralytics
    - For .hef: Hailo RT Python bindings (via hailo-all)
"""

import argparse
import csv
import logging
import os
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml

from hailo_common import (
    ThreadedCamera, create_session, default_source,
    load_labels as _load_labels, open_camera,
)

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
log = logging.getLogger(__name__)

_shutdown = False

# HaGRID 18-class gesture set
GESTURE_CLASSES = [
    "call", "dislike", "fist", "four", "like", "mute", "ok", "one",
    "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted",
    "three", "three2", "two_up", "two_up_inverted",
]

# Colors for each gesture class (BGR)
GESTURE_COLORS = {
    "like": (0, 200, 0), "dislike": (0, 0, 200), "fist": (200, 100, 0),
    "palm": (0, 200, 200), "peace": (200, 200, 0), "ok": (0, 255, 128),
    "call": (255, 150, 0), "one": (128, 0, 255), "rock": (0, 128, 255),
    "mute": (128, 128, 128), "stop": (0, 0, 255), "four": (200, 0, 200),
    "three": (100, 200, 100), "two_up": (200, 100, 200),
}
DEFAULT_COLOR = (200, 200, 200)


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True


def load_labels(path: str) -> list[str]:
    """Load class labels, falling back to HaGRID defaults."""
    return _load_labels(path, default=GESTURE_CLASSES)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GestureAction:
    message: str = ""
    command: str = ""
    cooldown: float = 2.0
    hold_time: float = 0.0


@dataclass
class GestureState:
    first_seen: float = 0.0
    last_seen: float = 0.0
    triggered: bool = False
    last_triggered: float = 0.0


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str


@dataclass
class OverlayMessage:
    text: str
    color: tuple
    expire: float


@dataclass
class GestureTracker:
    states: dict = field(default_factory=dict)
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    messages: list = field(default_factory=list)
    fps_samples: deque = field(default_factory=lambda: deque(maxlen=60))

    def get_state(self, gesture_name: str) -> GestureState:
        if gesture_name not in self.states:
            self.states[gesture_name] = GestureState()
        return self.states[gesture_name]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_actions(path: str) -> dict[str, GestureAction]:
    """Load gesture-to-action mappings from a YAML file."""
    if not path or not os.path.isfile(path):
        return {}

    with open(path) as f:
        data = yaml.safe_load(f)

    actions = {}
    for name, cfg in (data.get("gestures") or {}).items():
        if not isinstance(cfg, dict):
            continue
        actions[name] = GestureAction(
            message=cfg.get("message", ""),
            command=cfg.get("command", ""),
            cooldown=float(cfg.get("cooldown", 2.0)),
            hold_time=float(cfg.get("hold_time", 0.0)),
        )
    log.info("Loaded %d gesture actions from %s", len(actions), path)
    return actions


def _tuples_to_detections(tuples: list[tuple], labels: list[str]) -> list[Detection]:
    """Convert (x1, y1, x2, y2, conf, class_id) tuples to Detection objects."""
    return [
        Detection(
            x1=t[0], y1=t[1], x2=t[2], y2=t[3],
            confidence=t[4], class_id=t[5],
            class_name=labels[t[5]] if t[5] < len(labels) else str(t[5]),
        )
        for t in tuples
    ]


# ---------------------------------------------------------------------------
# Action engine
# ---------------------------------------------------------------------------

def process_gestures(
    detections: list[Detection],
    actions: dict[str, GestureAction],
    tracker: GestureTracker,
    csv_writer=None,
) -> None:
    now = time.monotonic()
    seen_gestures = set()

    for det in detections:
        gesture = det.class_name
        seen_gestures.add(gesture)
        state = tracker.get_state(gesture)

        if now - state.last_seen > 0.5:
            state.first_seen = now
            state.triggered = False
        state.last_seen = now

        action = actions.get(gesture)
        if not action:
            continue

        held_for = now - state.first_seen
        cooled_down = (now - state.last_triggered) >= action.cooldown

        if held_for >= action.hold_time and cooled_down and not state.triggered:
            state.triggered = True
            state.last_triggered = now

            tracker.history.append({
                "time": time.strftime("%H:%M:%S"),
                "gesture": gesture,
                "confidence": det.confidence,
                "held_for": round(held_for, 2),
            })

            if action.message:
                color = GESTURE_COLORS.get(gesture, DEFAULT_COLOR)
                tracker.messages.append(OverlayMessage(
                    text=action.message, color=color, expire=now + 2.0,
                ))
                log.info("Gesture: %s (conf=%.2f, held=%.1fs) -> %s",
                         gesture, det.confidence, held_for, action.message)

            if action.command:
                try:
                    subprocess.Popen(
                        action.command, shell=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                except OSError as e:
                    log.warning("Failed to run command for '%s': %s", gesture, e)

            if csv_writer:
                csv_writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    gesture, f"{det.confidence:.3f}",
                    f"{held_for:.2f}",
                    action.message,
                ])

    for gesture, state in tracker.states.items():
        if gesture not in seen_gestures and now - state.last_seen > 0.5:
            state.triggered = False


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_detections(frame: np.ndarray, detections: list[Detection]) -> None:
    for det in detections:
        color = GESTURE_COLORS.get(det.class_name, DEFAULT_COLOR)
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)

        label = f"{det.class_name} {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (det.x1, det.y1 - th - 8), (det.x1 + tw + 4, det.y1), color, -1)
        cv2.putText(frame, label, (det.x1 + 2, det.y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def draw_hud(
    frame: np.ndarray, tracker: GestureTracker, detections: list[Detection], fps: float
) -> None:
    h, w = frame.shape[:2]
    now = time.monotonic()

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Hands: {len(detections)}", (w - 140, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    tracker.messages = [m for m in tracker.messages if m.expire > now]
    y_msg = h - 20
    for msg in reversed(tracker.messages[-5:]):
        alpha = min(1.0, (msg.expire - now) / 0.5)
        color = tuple(int(c * alpha) for c in msg.color)
        cv2.putText(frame, msg.text, (10, y_msg),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y_msg -= 36

    if tracker.history:
        y_hist = 60
        cv2.putText(frame, "Recent:", (w - 200, y_hist),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        y_hist += 22
        for entry in list(tracker.history)[-5:]:
            text = f"{entry['time']} {entry['gesture']}"
            cv2.putText(frame, text, (w - 200, y_hist),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            y_hist += 18

    y_bar = 50
    for det in detections:
        state = tracker.states.get(det.class_name)
        if state:
            hold_time = getattr(state, '_hold_time', 0)
            if hold_time > 0:
                held = now - state.first_seen
                progress = min(1.0, held / hold_time)
                bar_w = 150
                color = GESTURE_COLORS.get(det.class_name, DEFAULT_COLOR)
                cv2.rectangle(frame, (10, y_bar), (10 + bar_w, y_bar + 14), (60, 60, 60), -1)
                cv2.rectangle(frame, (10, y_bar), (10 + int(bar_w * progress), y_bar + 14), color, -1)
                cv2.putText(frame, det.class_name, (bar_w + 18, y_bar + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_bar += 22


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    actions = load_actions(args.actions)
    tracker = GestureTracker()

    for gesture_name, action in actions.items():
        state = tracker.get_state(gesture_name)
        state._hold_time = action.hold_time

    csv_file = None
    csv_writer = None
    if args.log_csv:
        csv_file = open(args.log_csv, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if csv_file.tell() == 0:
            csv_writer.writerow(["timestamp", "gesture", "confidence", "hold_time", "action"])
        log.info("Logging gestures to: %s", args.log_csv)

    with create_session(args.model) as session:
        # Use model's own labels if available, otherwise fall back
        labels = load_labels(args.labels) if args.labels else session.labels
        if not args.labels and labels is session.labels:
            # For gesture models, prefer GESTURE_CLASSES over COCO
            labels = load_labels("")
        log.info("Model classes: %d", len(labels))

        log.info("Opening camera (source=%s)...", args.source)
        raw_cap = open_camera(args.source, args.capture_width, args.capture_height)
        if not raw_cap.isOpened():
            log.error("Could not open camera.")
            sys.exit(1)
        cap = ThreadedCamera(raw_cap)

        log.info("Running gesture recognition. Press 'q' to quit.")
        prev_time = time.monotonic()

        try:
            while not _shutdown:
                ret, frame = cap.read()
                if not ret:
                    log.warning("Frame capture failed — retrying...")
                    continue

                raw_dets = session.detect(
                    frame,
                    conf_threshold=args.confidence,
                    iou_threshold=args.iou,
                    num_classes=len(labels),
                )
                detections = _tuples_to_detections(raw_dets, labels)

                process_gestures(detections, actions, tracker, csv_writer)

                now = time.monotonic()
                dt = now - prev_time
                prev_time = now
                tracker.fps_samples.append(1.0 / dt if dt > 0 else 0)
                fps = sum(tracker.fps_samples) / len(tracker.fps_samples)

                if args.display:
                    draw_detections(frame, detections)
                    draw_hud(frame, tracker, detections, fps)
                    cv2.imshow("Gesture Recognition", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    if int(now) % 5 == 0 and dt < 0.3:
                        active = [d.class_name for d in detections]
                        if active:
                            log.info("FPS: %.1f | Active: %s", fps, ", ".join(active))

        finally:
            cap.release()
            if args.display:
                cv2.destroyAllWindows()
            if csv_file:
                csv_file.close()
            log.info("Stopped. Total gestures triggered: %d", len(tracker.history))


def main() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    parser = argparse.ArgumentParser(
        description="Hand gesture recognition with action triggers (Hailo NPU or local CPU/MPS)"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to model: .hef (Hailo), .pt or .onnx (Ultralytics)",
    )
    parser.add_argument("--actions", default="gesture_actions.yaml",
                        help="Path to gesture-to-action YAML config")
    parser.add_argument("--labels", default="",
                        help="Path to class labels file. Defaults to HaGRID 18 classes")
    parser.add_argument("--source", default=default_source(),
                        help="Camera source: index (0, 1), V4L2 path, or 'picam'")
    parser.add_argument("--capture-width", type=int, default=640)
    parser.add_argument("--capture-height", type=int, default=480)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--display", action="store_true",
                        help="Show live preview window")
    parser.add_argument("--log-csv", default="",
                        help="Path to CSV file for logging triggered gestures")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run(args)


if __name__ == "__main__":
    main()
