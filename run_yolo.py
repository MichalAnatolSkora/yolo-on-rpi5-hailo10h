#!/usr/bin/env python3
"""
Sample script to run YOLO object detection via Hailo-10H NPU on a Raspberry Pi 5.
This uses the GStreamer pipeline natively supported by Hailo software on the Pi.

Prerequisites:
- A compatible YOLOv8 .hef model compiled specifically for Hailo-10H.
- OpenCV installed (pip install opencv-python).
- Hailo GStreamer plugins installed (usually via sudo apt install hailo-all).
"""

import cv2
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Hailo-10H YOLO inference")
    parser.add_argument("--model", type=str, required=True, help="Path to the Hailo-10H YOLO .hef model")
    parser.add_argument("--labels", type=str, default="", help="Path to a text file mapping class IDs to names")
    args = parser.parse_args()

    hef_path = args.model
    if not os.path.exists(hef_path):
        print(f"Error: Model file '{hef_path}' not found.")
        return

    # Construct the GStreamer pipeline string
    # This pipeline reads from libcamerasrc (Pi Camera), runs it through hailonet,
    # draws bounding boxes via hailofilter, and converts it to BGR for OpenCV.
    # Note: `hailofilter` often requires an .so shared object, or we use `hailooverlay`.
    
    pipeline = (
        "libcamerasrc ! "
        "video/x-raw, width=640, height=640, format=RGB ! "
        "videoconvert ! "
        f"hailonet hef-path={hef_path} ! "
        "hailofilter so-path=/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so ! "
        "hailooverlay ! "
        "videoconvert ! "
        "appsink sync=false max-buffers=1 drop=true"
    )

    print("Opening camera with GStreamer pipeline...")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: Could not open video capture. Ensure camera is connected and pipeline is correct.")
        return

    print("Running inference... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to fetch frame.")
            break

        cv2.imshow("Hailo-10H YOLO Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
