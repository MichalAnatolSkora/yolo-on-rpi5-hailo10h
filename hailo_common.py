"""
Shared Hailo inference helpers for YOLO on Raspberry Pi 5 + Hailo-10H.

Provides camera access, pre/post-processing, and a context manager
that wraps the HailoRT VDevice/InferModel boilerplate.
"""

import logging
import os
import sys

import cv2
import numpy as np

log = logging.getLogger(__name__)

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
    """Open a camera source -- Pi Camera via libcamera or USB via V4L2."""
    if source == "picam":
        pipeline = (
            f"libcamerasrc ! "
            f"video/x-raw, width={width}, height={height}, format=RGB ! "
            f"videoconvert ! "
            f"appsink sync=false max-buffers=1 drop=true"
        )
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    # USB camera -- force V4L2 backend to avoid GStreamer issues
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


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
# Pre/post-processing
# ---------------------------------------------------------------------------

def preprocess(frame: np.ndarray, input_h: int, input_w: int) -> np.ndarray:
    """Resize and convert a BGR frame for YOLO input (RGB uint8)."""
    resized = cv2.resize(frame, (input_w, input_h))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.uint8)


def postprocess_nms(
    output: np.ndarray,
    frame_h: int,
    frame_w: int,
    num_classes: int,
    conf_threshold: float,
) -> list[tuple]:
    """
    Parse Hailo on-chip NMS output and return detections.

    Hailo NMS output is a flat array organized per class:
      [num_det_class0, y_min, x_min, y_max, x_max, score, ..., num_det_class1, ...]
    Coordinates are normalized [0, 1].

    Returns list of (x1, y1, x2, y2, confidence, class_id).
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
                detections.append((
                    int(x_min * frame_w), int(y_min * frame_h),
                    int(x_max * frame_w), int(y_max * frame_h),
                    float(score), class_id,
                ))
    return detections


def postprocess_raw(
    output: np.ndarray,
    frame_h: int,
    frame_w: int,
    input_h: int,
    input_w: int,
    conf_threshold: float,
    iou_threshold: float,
) -> list[tuple]:
    """
    Parse raw YOLO output tensor (no on-chip NMS) and return detections.

    Handles the standard YOLO output format: (1, num_detections, 4 + num_classes)
    where the 4 values are [x_center, y_center, width, height].

    Returns list of (x1, y1, x2, y2, confidence, class_id).
    """
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
# Hailo inference session
# ---------------------------------------------------------------------------

class HailoSession:
    """
    Context manager wrapping Hailo VDevice / InferModel setup.

    Usage::

        with HailoSession(model_path) as session:
            for frame in frames:
                input_data = preprocess(frame, session.input_h, session.input_w)
                output = session.infer(input_data)
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

        # Populated in __enter__
        self.has_nms: bool = False
        self.input_h: int = 0
        self.input_w: int = 0
        self.output_shape: tuple = ()

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

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on preprocessed uint8 RGB input. Returns the raw output array."""
        bindings = self._configured_model.create_bindings()
        bindings.input().set_buffer(np.expand_dims(input_data, axis=0))
        output_buf = np.empty([1] + list(self.output_shape), dtype=np.float32)
        bindings.output().set_buffer(output_buf)

        self._configured_model.wait_for_async_ready(timeout_ms=10000)
        job = self._configured_model.run_async([bindings])
        job.wait(timeout_ms=10000)

        return output_buf[0]

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._configured_model is not None:
            self._configured_model.__exit__(exc_type, exc_val, exc_tb)
        if self._vdevice is not None:
            self._vdevice.__exit__(exc_type, exc_val, exc_tb)
        return False
