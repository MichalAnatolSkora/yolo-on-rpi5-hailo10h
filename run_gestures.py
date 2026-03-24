#!/usr/bin/env python3
"""
Real-time hand gesture recognition on Hailo-10H NPU with action triggers.

Detects hand gestures via a YOLOv12 model compiled for Hailo-10H and maps them
to configurable actions (shell commands, on-screen messages). Supports gesture
hold detection, cooldowns, gesture history, and multi-hand tracking.

Usage:
    python run_gestures.py --model ~/hailo_models/yolov12n_gestures.hef
    python run_gestures.py --model ~/hailo_models/yolov12n_gestures.hef --source /dev/video0
    python run_gestures.py --model ~/hailo_models/yolov12n_gestures.hef --actions gesture_actions.yaml
    python run_gestures.py --model ~/hailo_models/yolov12n_gestures.hef --log-csv gestures.csv

Prerequisites:
    - YOLOv12 gesture .hef model compiled for Hailo-10H
    - OpenCV, NumPy, PyYAML
    - Hailo RT Python bindings (via hailo-all)
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


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GestureAction:
    """Action to execute when a gesture is detected."""
    message: str = ""
    command: str = ""
    cooldown: float = 2.0
    hold_time: float = 0.0


@dataclass
class GestureState:
    """Tracks the state of a single gesture for hold detection."""
    first_seen: float = 0.0
    last_seen: float = 0.0
    triggered: bool = False
    last_triggered: float = 0.0


@dataclass
class Detection:
    """A single detection result."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str


@dataclass
class OverlayMessage:
    """A temporary message to display on the overlay."""
    text: str
    color: tuple
    expire: float


@dataclass
class GestureTracker:
    """Tracks gesture states, history, and overlay messages."""
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


def load_labels(path: str) -> list[str]:
    """Load class labels from a text file, falling back to HaGRID defaults."""
    if not path or not os.path.isfile(path):
        return GESTURE_CLASSES
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def preprocess(frame: np.ndarray, input_h: int, input_w: int) -> np.ndarray:
    """Resize and convert a BGR frame for YOLO input."""
    resized = cv2.resize(frame, (input_w, input_h))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.uint8)


def postprocess_nms(
    output: np.ndarray,
    frame_h: int, frame_w: int,
    num_classes: int,
    conf_threshold: float,
    labels: list[str],
) -> list[Detection]:
    """Parse Hailo on-chip NMS output into Detection objects.

    Hailo NMS output is a flat array organized per class:
      [num_det_class0, y_min, x_min, y_max, x_max, score, ..., num_det_class1, ...]
    Coordinates are normalized [0, 1].
    """
    detections = []
    offset = 0
    for class_id in range(num_classes):
        if offset >= len(output):
            break
        num_dets = int(output[offset])
        offset += 1
        for _ in range(num_dets):
            if offset + 5 > len(output):
                break
            y_min, x_min, y_max, x_max, score = output[offset:offset + 5]
            offset += 5
            if score >= conf_threshold:
                detections.append(Detection(
                    x1=int(x_min * frame_w), y1=int(y_min * frame_h),
                    x2=int(x_max * frame_w), y2=int(y_max * frame_h),
                    confidence=float(score),
                    class_id=class_id,
                    class_name=labels[class_id] if class_id < len(labels) else str(class_id),
                ))
    return detections


def postprocess_raw(
    output: np.ndarray,
    frame_h: int, frame_w: int,
    input_h: int, input_w: int,
    conf_threshold: float, iou_threshold: float,
    labels: list[str],
) -> list[Detection]:
    """Parse raw YOLO output tensor (no on-chip NMS) into Detection objects."""
    if output.ndim == 3:
        output = output[0]
    if output.shape[0] < output.shape[1]:
        output = output.T

    boxes = output[:, :4]
    scores = output[:, 4:]

    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(class_ids)), class_ids]

    mask = confidences >= conf_threshold
    boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]

    if len(boxes) == 0:
        return []

    # Center-format to corners, scaled to frame dimensions
    x1 = (boxes[:, 0] - boxes[:, 2] / 2) * frame_w / input_w
    y1 = (boxes[:, 1] - boxes[:, 3] / 2) * frame_h / input_h
    x2 = (boxes[:, 0] + boxes[:, 2] / 2) * frame_w / input_w
    y2 = (boxes[:, 1] + boxes[:, 3] / 2) * frame_h / input_h

    rects = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(rects, confidences.tolist(), conf_threshold, iou_threshold)

    detections = []
    for i in indices:
        idx = i if isinstance(i, int) else i[0]
        cid = int(class_ids[idx])
        detections.append(Detection(
            x1=int(x1[idx]), y1=int(y1[idx]),
            x2=int(x2[idx]), y2=int(y2[idx]),
            confidence=float(confidences[idx]),
            class_id=cid,
            class_name=labels[cid] if cid < len(labels) else str(cid),
        ))
    return detections


# ---------------------------------------------------------------------------
# Action engine
# ---------------------------------------------------------------------------

def process_gestures(
    detections: list[Detection],
    actions: dict[str, GestureAction],
    tracker: GestureTracker,
    csv_writer=None,
) -> None:
    """
    Process detected gestures: update hold timers, fire actions when
    hold_time is met and cooldown has elapsed, log to CSV.
    """
    now = time.monotonic()
    seen_gestures = set()

    for det in detections:
        gesture = det.class_name
        seen_gestures.add(gesture)
        state = tracker.get_state(gesture)

        # Update timing
        if now - state.last_seen > 0.5:
            # Gap in detection — reset hold timer
            state.first_seen = now
            state.triggered = False
        state.last_seen = now

        action = actions.get(gesture)
        if not action:
            continue

        held_for = now - state.first_seen
        cooled_down = (now - state.last_triggered) >= action.cooldown

        if held_for >= action.hold_time and cooled_down and not state.triggered:
            # Fire the action
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

    # Reset gestures that are no longer visible
    for gesture, state in tracker.states.items():
        if gesture not in seen_gestures and now - state.last_seen > 0.5:
            state.triggered = False


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_detections(frame: np.ndarray, detections: list[Detection]) -> None:
    """Draw bounding boxes and gesture labels."""
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
    """Draw the heads-up display: FPS, active gestures, action messages, history."""
    h, w = frame.shape[:2]
    now = time.monotonic()

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Active gesture count
    cv2.putText(frame, f"Hands: {len(detections)}", (w - 140, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Action messages (bottom-left, stacked upward)
    tracker.messages = [m for m in tracker.messages if m.expire > now]
    y_msg = h - 20
    for msg in reversed(tracker.messages[-5:]):
        alpha = min(1.0, (msg.expire - now) / 0.5)  # fade out
        color = tuple(int(c * alpha) for c in msg.color)
        cv2.putText(frame, msg.text, (10, y_msg),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y_msg -= 36

    # Gesture history sidebar (right side, last 5)
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

    # Hold progress bars for active gestures
    y_bar = 50
    for det in detections:
        action = None
        # Look up if there's a hold requirement
        state = tracker.states.get(det.class_name)
        if state:
            # Find the action config from somewhere accessible
            hold_time = 0
            for gesture, gs in tracker.states.items():
                if gesture == det.class_name:
                    hold_time = getattr(gs, '_hold_time', 0)
                    break

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
# Camera
# ---------------------------------------------------------------------------

def open_camera(source: str, width: int, height: int) -> cv2.VideoCapture:
    """Open Pi Camera via libcamera or USB camera via V4L2."""
    if source == "picam":
        pipeline = (
            f"libcamerasrc ! "
            f"video/x-raw, width={width}, height={height}, format=RGB ! "
            f"videoconvert ! "
            f"appsink sync=false max-buffers=1 drop=true"
        )
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    # Force V4L2 backend to avoid GStreamer issues with USB cameras
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    try:
        from hailo_platform import (
            HEF, VDevice, FormatType,
            InferVStreams, InputVStreamParams, OutputVStreamParams,
        )
    except ImportError:
        log.error(
            "hailo_platform not found. Ensure hailo-all is installed and you are "
            "using system Python or a venv with --system-site-packages."
        )
        sys.exit(1)

    if not os.path.isfile(args.model):
        log.error("Model file not found: %s", args.model)
        sys.exit(1)

    labels = load_labels(args.labels)
    actions = load_actions(args.actions)
    tracker = GestureTracker()

    # Attach hold_time to states for HUD progress bars
    for gesture_name, action in actions.items():
        state = tracker.get_state(gesture_name)
        state._hold_time = action.hold_time

    # CSV logging
    csv_file = None
    csv_writer = None
    if args.log_csv:
        csv_file = open(args.log_csv, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if csv_file.tell() == 0:
            csv_writer.writerow(["timestamp", "gesture", "confidence", "hold_time", "action"])
        log.info("Logging gestures to: %s", args.log_csv)

    # Load HEF
    log.info("Loading model: %s", args.model)
    hef = HEF(args.model)

    # Check if model has on-chip NMS
    output_vstream_infos = hef.get_output_vstream_infos()
    has_nms = any("nms" in info.name.lower() for info in output_vstream_infos)
    if has_nms:
        log.info("Model has on-chip NMS — using NMS output format")

    # Get input info for shape
    input_vstream_infos = hef.get_input_vstream_infos()
    input_shape = input_vstream_infos[0].shape
    input_h, input_w = input_shape[1], input_shape[2]
    input_name = input_vstream_infos[0].name

    with VDevice() as vdevice:
        network_group = vdevice.configure(hef)[0]

        input_vstream_params = InputVStreamParams.make(
            network_group, format_type=FormatType.UINT8
        )
        output_vstream_params = OutputVStreamParams.make(
            network_group, format_type=FormatType.FLOAT32
        )

        log.info("Model input: %s | Classes: %d | NMS: %s", input_shape, len(labels), has_nms)

        # Open camera
        log.info("Opening camera (source=%s)...", args.source)
        cap = open_camera(args.source, args.capture_width, args.capture_height)
        if not cap.isOpened():
            log.error("Could not open camera.")
            sys.exit(1)

        log.info("Running gesture recognition. Press 'q' to quit.")
        prev_time = time.monotonic()

        with InferVStreams(network_group, input_vstream_params, output_vstream_params) as pipeline:
            try:
                while not _shutdown:
                    ret, frame = cap.read()
                    if not ret:
                        log.warning("Frame capture failed — retrying...")
                        continue

                    # Inference
                    input_data = preprocess(frame, input_h, input_w)
                    input_dict = {input_name: np.expand_dims(input_data, axis=0)}
                    results = pipeline.infer(input_dict)

                    # Get output (first output layer)
                    output_name = list(results.keys())[0]
                    output = results[output_name][0]  # remove batch dim

                    # Postprocess
                    if has_nms:
                        detections = postprocess_nms(
                            output.flatten(), frame.shape[0], frame.shape[1],
                            len(labels), args.confidence, labels,
                        )
                    else:
                        detections = postprocess_raw(
                            output, frame.shape[0], frame.shape[1],
                            input_h, input_w,
                            args.confidence, args.iou, labels,
                        )

                    # Process gesture actions
                    process_gestures(detections, actions, tracker, csv_writer)

                    # FPS
                    now = time.monotonic()
                    dt = now - prev_time
                    prev_time = now
                    tracker.fps_samples.append(1.0 / dt if dt > 0 else 0)
                    fps = sum(tracker.fps_samples) / len(tracker.fps_samples)

                    if args.display:
                        draw_detections(frame, detections)
                        draw_hud(frame, tracker, detections, fps)
                        cv2.imshow("Gesture Recognition - Hailo-10H", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    else:
                        # Headless: periodic status log
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
        description="Hand gesture recognition on Hailo-10H with action triggers"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to gesture recognition .hef model compiled for Hailo-10H",
    )
    parser.add_argument(
        "--actions", default="gesture_actions.yaml",
        help="Path to gesture-to-action YAML config (default: gesture_actions.yaml)",
    )
    parser.add_argument(
        "--labels", default="",
        help="Path to class labels file (one per line). Defaults to HaGRID 18 classes",
    )
    parser.add_argument(
        "--source", default="picam",
        help="'picam' for Pi Camera, or V4L2 device path (e.g. /dev/video0)",
    )
    parser.add_argument(
        "--capture-width", type=int, default=640,
        help="Camera capture width (default: 640)",
    )
    parser.add_argument(
        "--capture-height", type=int, default=480,
        help="Camera capture height (default: 480)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="NMS IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Show live preview window (requires a display/monitor)",
    )
    parser.add_argument(
        "--log-csv", default="",
        help="Path to CSV file for logging triggered gestures",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run(args)


if __name__ == "__main__":
    main()
