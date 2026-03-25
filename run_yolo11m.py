#!/usr/bin/env python3
"""
Run YOLOv11m object detection via Hailo-10H NPU on a Raspberry Pi 5.

Thin wrapper around run_yolo11.py that defaults to the YOLOv11m model.

Usage:
    python run_yolo11m.py
    python run_yolo11m.py --source /dev/video0
    python run_yolo11m.py --model ~/hailo_models/yolov11m.hef --confidence 0.4
"""

import os
import sys

# Inject default --model if not provided by the user
DEFAULT_MODEL = os.path.expanduser("~/hailo_models/yolov11m.hef")
if "--model" not in sys.argv:
    sys.argv.extend(["--model", DEFAULT_MODEL])

from run_yolo11 import main  # noqa: E402

if __name__ == "__main__":
    main()
