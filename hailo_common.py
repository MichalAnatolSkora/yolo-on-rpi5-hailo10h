"""
Shared inference helpers for YOLO on Raspberry Pi 5 + Hailo-10H or local (CPU/MPS).

Provides camera access, drawing, and two inference backends:
- HailoSession:        Hailo-10H NPU via HailoRT  (.hef models)
- UltralyticsSession:  CPU / MPS via Ultralytics   (.pt / .onnx models)

Use ``create_session(model_path)`` to auto-select based on file extension.
"""

import json
import logging
import os
import platform
import sys
import threading

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tracker config
# ---------------------------------------------------------------------------

TRACKER_CONFIG_DEFAULTS = {
    "confidence": 0.3,
    "iou": 0.45,
    "min_iou": 0.15,
    "max_distance": 200.0,
    "max_disappeared": 50,
    "min_hits": 3,
    "buffer": 0,
    "deduplicate": True,
}


def load_tracker_config(path: str = "tracker_config.json") -> dict:
    """Load tunable tracker params from JSON, falling back to defaults.

    Missing file or missing keys → defaults are used. The tune-tracker agent
    edits this file; CLI flags still override the loaded values.
    """
    cfg = dict(TRACKER_CONFIG_DEFAULTS)
    if not os.path.isfile(path):
        return cfg
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not parse %s (%s) — using defaults", path, exc)
        return cfg
    for k, v in data.items():
        if k.startswith("_"):
            continue
        if k in cfg:
            cfg[k] = v
        else:
            log.warning("Unknown key in %s: %r — ignored", path, k)
    return cfg

# COCO class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def open_camera(source: str, width: int, height: int) -> cv2.VideoCapture:
    """Open a camera source, cross-platform.

    - ``"picam"``       -> Pi Camera via libcamera GStreamer pipeline
    - ``"/dev/videoN"`` -> V4L2 device (Linux)
    - ``"0"``, ``"1"``  -> system camera by index (macOS / Windows / Linux)
    """
    if source == "picam":
        pipeline = (
            f"libcamerasrc ! "
            f"video/x-raw, width={width}, height={height}, format=RGB ! "
            f"videoconvert ! "
            f"appsink sync=false max-buffers=1 drop=true"
        )
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    # Numeric string -> camera index (works on all platforms)
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap

    # Device path (e.g. /dev/video0) -> V4L2 on Linux
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


class ThreadedCamera:
    """Wraps a cv2.VideoCapture to read frames in a background thread.

    ``read()`` always returns the most recent frame immediately without
    blocking on the camera.  This decouples camera FPS from inference FPS
    so the model can run as fast as it can.

    Drop-in replacement for cv2.VideoCapture in the main loop.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._ret = False
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            ret, frame = self._cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame

    def isOpened(self) -> bool:
        return self._cap.isOpened()

    def read(self):
        with self._lock:
            return self._ret, self._frame

    def get(self, prop):
        return self._cap.get(prop)

    def release(self):
        self._running = False
        self._thread.join(timeout=2)
        self._cap.release()


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def load_labels(path: str, default: list[str] | None = None) -> list[str]:
    """Load class labels from a text file (one label per line)."""
    if not path or not os.path.isfile(path):
        return default if default is not None else COCO_CLASSES
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_detections(
    frame: np.ndarray, detections: list[tuple], labels: list[str]
) -> np.ndarray:
    """Draw bounding boxes and labels on the frame."""
    for x1, y1, x2, y2, conf, cls_id in detections:
        label = labels[cls_id] if cls_id < len(labels) else str(cls_id)
        text = f"{label} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(frame, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


# ---------------------------------------------------------------------------
# Hailo inference session (.hef models, Raspberry Pi + Hailo NPU)
# ---------------------------------------------------------------------------

def _preprocess(frame: np.ndarray, input_h: int, input_w: int) -> np.ndarray:
    """Resize and convert a BGR frame for YOLO input (RGB uint8)."""
    resized = cv2.resize(frame, (input_w, input_h))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.uint8)


def _postprocess_nms(
    output: np.ndarray,
    frame_h: int, frame_w: int,
    num_classes: int,
    conf_threshold: float,
) -> list[tuple]:
    """Parse Hailo on-chip NMS output into detection tuples."""
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
                detections.append((
                    int(x_min * frame_w), int(y_min * frame_h),
                    int(x_max * frame_w), int(y_max * frame_h),
                    float(score), class_id,
                ))
    return detections


def _postprocess_raw(
    output: np.ndarray,
    frame_h: int, frame_w: int,
    input_h: int, input_w: int,
    conf_threshold: float,
    iou_threshold: float,
) -> list[tuple]:
    """Parse raw YOLO output tensor (no on-chip NMS) into detection tuples."""
    if output.ndim == 3:
        output = output[0]
    if output.shape[0] < output.shape[1]:
        output = output.T

    boxes = output[:, :4]
    scores = output[:, 4:]

    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(class_ids)), class_ids]

    mask = confidences >= conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    x1 = (boxes[:, 0] - boxes[:, 2] / 2) * frame_w / input_w
    y1 = (boxes[:, 1] - boxes[:, 3] / 2) * frame_h / input_h
    x2 = (boxes[:, 0] + boxes[:, 2] / 2) * frame_w / input_w
    y2 = (boxes[:, 1] + boxes[:, 3] / 2) * frame_h / input_h

    rects = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(rects, confidences.tolist(), conf_threshold, iou_threshold)

    detections = []
    for i in indices:
        idx = i if isinstance(i, int) else i[0]
        detections.append((
            int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx]),
            float(confidences[idx]), int(class_ids[idx]),
        ))
    return detections


class HailoSession:
    """
    Hailo-10H NPU inference backend for .hef models.

    Usage::

        with HailoSession("model.hef") as session:
            detections = session.detect(frame, conf=0.5, iou=0.45, num_classes=80)
    """

    def __init__(self, model_path: str):
        try:
            from hailo_platform import HEF, VDevice, FormatType, HailoSchedulingAlgorithm
        except ImportError:
            log.error(
                "hailo_platform not found. Ensure hailo-all / hailo-h10-all is installed "
                "and you are using system Python or a venv with --system-site-packages."
            )
            sys.exit(1)

        if not os.path.isfile(model_path):
            log.error("Model file not found: %s", model_path)
            sys.exit(1)

        self._model_path = model_path
        self._FormatType = FormatType
        self._HailoSchedulingAlgorithm = HailoSchedulingAlgorithm
        self._VDevice = VDevice
        self._HEF = HEF

        self.has_nms: bool = False
        self.input_h: int = 0
        self.input_w: int = 0
        self.output_shape: tuple = ()
        self.labels: list[str] = COCO_CLASSES

        self._vdevice = None
        self._configured_model = None

    def __enter__(self):
        log.info("Loading HEF model: %s", self._model_path)
        hef = self._HEF(self._model_path)

        output_vstream_infos = hef.get_output_vstream_infos()
        self.has_nms = any("nms" in info.name.lower() for info in output_vstream_infos)
        if self.has_nms:
            log.info("Model has on-chip NMS -- using NMS output format")

        input_vstream_infos = hef.get_input_vstream_infos()
        input_shape = input_vstream_infos[0].shape
        self.input_h, self.input_w = input_shape[0], input_shape[1]
        log.info("Model input shape: %s", input_shape)

        params = self._VDevice.create_params()
        params.scheduling_algorithm = self._HailoSchedulingAlgorithm.ROUND_ROBIN
        self._vdevice = self._VDevice(params)
        self._vdevice.__enter__()

        log.info("Creating InferModel (async API)...")
        infer_model = self._vdevice.create_infer_model(self._model_path)
        infer_model.input().set_format_type(self._FormatType.UINT8)
        infer_model.output().set_format_type(self._FormatType.FLOAT32)

        self.output_shape = infer_model.output().shape
        log.info("InferModel output shape: %s", self.output_shape)

        self._configured_model = infer_model.configure()
        self._configured_model.__enter__()

        return self

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        num_classes: int = 80,
    ) -> list[tuple]:
        """Run inference and return list of (x1, y1, x2, y2, conf, class_id)."""
        input_data = _preprocess(frame, self.input_h, self.input_w)

        bindings = self._configured_model.create_bindings()
        bindings.input().set_buffer(np.expand_dims(input_data, axis=0))
        output_buf = np.empty([1] + list(self.output_shape), dtype=np.float32)
        bindings.output().set_buffer(output_buf)

        self._configured_model.wait_for_async_ready(timeout_ms=10000)
        job = self._configured_model.run_async([bindings])
        job.wait(timeout_ms=10000)

        output = output_buf[0]

        if self.has_nms:
            return _postprocess_nms(
                output.flatten(),
                frame.shape[0], frame.shape[1],
                num_classes, conf_threshold,
            )
        return _postprocess_raw(
            output,
            frame.shape[0], frame.shape[1],
            self.input_h, self.input_w,
            conf_threshold, iou_threshold,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._configured_model is not None:
            self._configured_model.__exit__(exc_type, exc_val, exc_tb)
        if self._vdevice is not None:
            self._vdevice.__exit__(exc_type, exc_val, exc_tb)
        return False


# ---------------------------------------------------------------------------
# Ultralytics inference session (.pt / .onnx models, CPU / MPS / CUDA)
# ---------------------------------------------------------------------------

class UltralyticsSession:
    """
    Local inference backend using Ultralytics YOLO.

    Works on any platform (macOS, Linux, Windows).
    Automatically uses MPS on Apple Silicon, CUDA if available, else CPU.

    Usage::

        with UltralyticsSession("yolo11n.pt") as session:
            detections = session.detect(frame, conf=0.5, iou=0.45)
    """

    def __init__(self, model_path: str):
        try:
            from ultralytics import YOLO
        except ImportError:
            log.error("ultralytics not found. Install with: pip install ultralytics")
            sys.exit(1)

        self._YOLO = YOLO
        self._model_path = model_path
        self._model = None
        self.labels: list[str] = COCO_CLASSES

    def __enter__(self):
        log.info("Loading model: %s (Ultralytics backend)", self._model_path)
        self._model = self._YOLO(self._model_path)

        # Use model's own class names if available
        if hasattr(self._model, "names") and self._model.names:
            self.labels = [self._model.names.get(i, str(i)) for i in range(len(self._model.names))]
            log.info("Model classes: %d", len(self.labels))

        # Log device selection
        device = "cpu"
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            try:
                import torch
                if torch.backends.mps.is_available():
                    device = "mps"
            except Exception:
                pass
        log.info("Device: %s", device)
        self._device = device

        return self

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        num_classes: int = 80,
    ) -> list[tuple]:
        """Run inference and return list of (x1, y1, x2, y2, conf, class_id)."""
        results = self._model.predict(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self._device,
            verbose=False,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append((
                    int(x1), int(y1), int(x2), int(y2),
                    float(box.conf[0]),
                    int(box.cls[0]),
                ))
        return detections

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_session(model_path: str):
    """Auto-select inference backend based on model file extension.

    - ``.hef``          -> HailoSession  (requires Hailo hardware + HailoRT)
    - ``.pt`` / ``.onnx`` -> UltralyticsSession (CPU / MPS / CUDA)
    """
    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".hef":
        return HailoSession(model_path)
    if ext in (".pt", ".onnx"):
        return UltralyticsSession(model_path)
    log.error("Unsupported model format '%s'. Use .hef, .pt, or .onnx", ext)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Platform defaults
# ---------------------------------------------------------------------------

def default_source() -> str:
    """Return a sensible default camera source for the current platform."""
    if platform.machine() == "aarch64" and os.path.isfile("/etc/rpi-issue"):
        return "/dev/video0"
    return "0"


def default_model() -> str:
    """Return a sensible default model path for the current platform."""
    if platform.machine() == "aarch64" and os.path.isfile("/etc/rpi-issue"):
        return os.path.expanduser("~/hailo_models/yolov11n.hef")
    return "yolo11n.pt"
