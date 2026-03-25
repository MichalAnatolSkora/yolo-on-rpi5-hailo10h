#!/usr/bin/env python3
"""
Run YOLOv12 object detection via Hailo-10H NPU on a Raspberry Pi 5.

This script uses the Hailo Python API (HailoRT) directly instead of GStreamer,
giving more control over pre/post-processing — useful for newer YOLO architectures
like YOLOv12 where GStreamer post-process .so files may not yet be available.

Prerequisites:
- A YOLOv12 .hef model compiled for Hailo-10H.
- OpenCV (pip install opencv-python).
- Hailo RT Python bindings (installed via hailo-all).
- NumPy (pip install numpy).

Usage:
    python run_yolo11.py --model ~/hailo_models/yolov12n.hef
    python run_yolo11.py --model ~/hailo_models/yolov12n.hef --source /dev/video0
    python run_yolo11.py --model ~/hailo_models/yolov12n.hef --confidence 0.4
"""

import argparse
import logging
import os
import signal
import sys

import cv2
import numpy as np

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
log = logging.getLogger(__name__)

_shutdown = False

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


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True


def load_labels(path: str) -> list[str]:
    """Load class labels from a text file (one label per line)."""
    if not path or not os.path.isfile(path):
        return COCO_CLASSES
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def preprocess(frame: np.ndarray, input_h: int, input_w: int) -> np.ndarray:
    """Resize and normalize a BGR frame for YOLO input."""
    resized = cv2.resize(frame, (input_w, input_h))
    # Convert BGR to RGB, normalize to [0, 1], NHWC uint8 for Hailo
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8)


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
    # Squeeze batch dimension
    if output.ndim == 3:
        output = output[0]

    # If shape is (features, detections), transpose to (detections, features)
    if output.shape[0] < output.shape[1]:
        output = output.T

    boxes = output[:, :4]
    scores = output[:, 4:]

    # Get best class per detection
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(class_ids)), class_ids]

    # Filter by confidence
    mask = confidences >= conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    # Convert center format to corner format
    x1 = (boxes[:, 0] - boxes[:, 2] / 2) * frame_w / input_w
    y1 = (boxes[:, 1] - boxes[:, 3] / 2) * frame_h / input_h
    x2 = (boxes[:, 0] + boxes[:, 2] / 2) * frame_w / input_w
    y2 = (boxes[:, 1] + boxes[:, 3] / 2) * frame_h / input_h

    # NMS
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


def open_camera(source: str, width: int, height: int) -> cv2.VideoCapture:
    """Open a camera source — Pi Camera via libcamera or USB via V4L2."""
    if source == "picam":
        pipeline = (
            f"libcamerasrc ! "
            f"video/x-raw, width={width}, height={height}, format=RGB ! "
            f"videoconvert ! "
            f"appsink sync=false max-buffers=1 drop=true"
        )
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    # USB camera — force V4L2 backend to avoid GStreamer issues
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


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

    # --- Load HEF model ---
    log.info("Loading HEF model: %s", args.model)
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
    log.info("Model input shape: %s", input_shape)

    # --- Configure Hailo device ---
    with VDevice() as vdevice:
        try:
            network_group = vdevice.configure(hef)[0]
        except Exception as e:
            if "HAILO_NOT_IMPLEMENTED" in str(e) or "error: 7" in str(e):
                log.error(
                    "HEF model is not compatible with this Hailo device. "
                    "This usually means the .hef was compiled for a different architecture "
                    "(e.g. hailo8 vs hailo10h). Download the correct HEF for your device."
                )
                sys.exit(1)
            raise

        input_vstream_params = InputVStreamParams.make(
            network_group, format_type=FormatType.UINT8
        )
        output_vstream_params = OutputVStreamParams.make(
            network_group, format_type=FormatType.FLOAT32
        )

        # --- Open camera ---
        log.info("Opening camera (source=%s)...", args.source)
        cap = open_camera(args.source, args.capture_width, args.capture_height)
        if not cap.isOpened():
            log.error("Could not open camera. Check connection.")
            sys.exit(1)

        log.info("Running YOLOv12 inference. Press 'q' to quit.")
        frame_count = 0

        with InferVStreams(network_group, input_vstream_params, output_vstream_params) as pipeline:
            try:
                while not _shutdown:
                    ret, frame = cap.read()
                    if not ret:
                        log.warning("Failed to fetch frame — retrying...")
                        continue

                    # Preprocess
                    input_data = preprocess(frame, input_h, input_w)

                    # Run inference on Hailo-10H
                    input_dict = {input_name: np.expand_dims(input_data, axis=0)}
                    results = pipeline.infer(input_dict)

                    # Get output (first output layer)
                    output_name = list(results.keys())[0]
                    output = results[output_name][0]  # remove batch dim

                    # Postprocess and draw
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
                    frame = draw_detections(frame, detections, labels)

                    # Show FPS every 30 frames
                    frame_count += 1
                    if frame_count % 30 == 0:
                        fps = cap.get(cv2.CAP_PROP_FPS) or 0
                        log.info("Detections: %d | Camera FPS: %.1f", len(detections), fps)

                    if args.display:
                        cv2.imshow("YOLOv12 Hailo-10H", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
            finally:
                cap.release()
                if args.display:
                    cv2.destroyAllWindows()
                log.info("Stopped.")


def main() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    parser = argparse.ArgumentParser(
        description="Run YOLOv12 object detection on Hailo-10H NPU"
    )
    parser.add_argument(
        "--model", required=True, help="Path to YOLOv12 .hef model compiled for Hailo-10H"
    )
    parser.add_argument(
        "--labels", default="", help="Path to class labels file (one per line). Defaults to COCO"
    )
    parser.add_argument(
        "--source", default="picam",
        help="'picam' for Pi Camera, or V4L2 device path for USB (e.g. /dev/video0)",
    )
    parser.add_argument(
        "--capture-width", type=int, default=640, help="Camera capture width (default: 640)"
    )
    parser.add_argument(
        "--capture-height", type=int, default=480, help="Camera capture height (default: 480)"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)"
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Show live preview window (requires a display/monitor)",
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
