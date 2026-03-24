# YOLO on Raspberry Pi 5 with Hailo-10H

> **If this repo saved you hours of Hailo configuration headaches, please give it a :star: — it helps others find it too!**

One-click setup for running YOLO object detection on Raspberry Pi 5 with the Hailo-10H AI accelerator. Handles driver installation, PCIe configuration, model downloads, and diagnostics automatically.

## Hardware Requirements
| Component | Notes |
|---|---|
| **Raspberry Pi 5** | 4GB or 8GB RAM recommended |
| **Hailo-10H NPU module** | Via M.2 Hat+ or similar PCIe expansion board |
| **Active Cooler for Raspberry Pi 5** | Required — significant heat generation |
| **Thermal Pad** | Transfers heat from Hailo-10H to the expansion board |
| **Raspberry Pi Camera** *(optional)* | For real-time inference demos |

## Quick Start

```bash
git clone https://github.com/MichalAnatolSkora/yolo-on-rpi5-hailo10h.git
cd yolo-on-rpi5-hailo10h
./install_hailo.sh
sudo reboot
```

That's it. The script auto-detects your hardware (Hailo-10H or Hailo-8), installs the correct driver package, enables PCIe Gen 3, and blacklists conflicting drivers if needed.

## Verifying the Installation

After rebooting, check that the Hailo-10H NPU is recognized by the system:

```bash
hailortcli fw-control identify
```
You should see output similar to "Identifying board...", followed by details about the Hailo-10 structure and firmware version.

You can also verify that PCIe Gen 3 is active:
```bash
sudo lspci -vv | grep -i hailo -A 20 | grep -i speed
```
Look for `Speed 8GT/s` which indicates Gen 3 is active (Gen 2 would show `5GT/s`).

## YOLOv12 Object Detection

Install and run in two commands:

```bash
./install_yolo12.sh                                                       # downloads model + Python deps
python run_yolo12.py --model ~/hailo_models/yolov12n.hef --display        # run with live preview
```

The install script downloads a pre-compiled YOLOv12n HEF from Hailo Model Zoo and sets up a Python virtual environment. Idempotent — safe to re-run.

**Options:**
```bash
python run_yolo12.py --model ~/hailo_models/yolov12n.hef --display --source /dev/video0  # USB camera
python run_yolo12.py --model ~/hailo_models/yolov12n.hef --confidence 0.4 --iou 0.5      # headless (no window)
```

To find your USB camera device path: `ls /dev/video*` or `v4l2-ctl --list-devices`.

## Gesture Recognition

Detects 18 hand gestures (HaGRID dataset) and maps them to configurable actions — shell commands, on-screen messages, CSV logging.

```bash
./install_yolo12_gestures.sh                                                    # setup model + dataset
python run_gestures.py --model ~/hailo_models/yolov12n_gestures.hef --display   # run with live preview
```

The gesture model is separate from the object detection model above — different dataset (HaGRID vs COCO), different weights. The install script will prompt you to supply training images from [HaGRID](https://github.com/hukenovs/hagrid) or [Roboflow](https://universe.roboflow.com), then trains and compiles a Hailo-10H HEF. Reuses the existing venv if you already ran `install_yolo12.sh`.

**Options:**
```bash
python run_gestures.py --model ~/hailo_models/yolov12n_gestures.hef --display --source /dev/video0
python run_gestures.py --model ~/hailo_models/yolov12n_gestures.hef --log-csv gestures.csv   # headless
```

**Configure actions** in `gesture_actions.yaml`:
```yaml
gestures:
  like:
    message: "Thumbs Up!"
    command: "echo 'liked' >> /tmp/gesture_log.txt"
    cooldown: 3       # seconds before re-triggering
    hold_time: 0.5    # seconds gesture must be held
  palm:
    message: "Stop / Pause"
    hold_time: 0.8
```

Features: hold detection, cooldown, shell command triggers, HUD overlay, headless mode, CSV logging.

## Other Models

Pre-compiled HEFs for Hailo-10H from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) (DFC v5.2.0):

> **⚠️ Version Compatibility:** The `.hef` model's compiled version must be compatible with your system's `hailort` package version. The models below require `hailort` **5.2.0**. If you use them on an older version (like 5.1.1), you will get a `HAILO_NOT_IMPLEMENTED` error.

| Model | Size | Download |
|---|---|---|
| YOLOv12n | ~5.5 MB | [yolov12n.hef](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/yolov12n.hef) |
| YOLOv11n | ~4.9 MB | [yolov11n.hef](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/yolov11n.hef) |
| YOLOv8n | ~6.7 MB | [yolov8n.hef](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/yolov8n.hef) |
| YOLOv8s | ~13.1 MB | [yolov8s.hef](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/yolov8s.hef) |
| YOLOv8m | — | [yolov8m.hef](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/yolov8m.hef) |

**Hailo-8 models will NOT work on Hailo-10H.** Use only `hailo10h`-compiled HEFs. Place files in `~/hailo_models/`.

GStreamer pipeline (for YOLOv8 with Hailo post-process plugins):
```bash
python run_yolo.py --model ~/hailo_models/yolov8n.hef
```

## Troubleshooting

Run the diagnostic script to identify issues:
```bash
./troubleshoot.sh
```
It checks PCIe detection, kernel driver, firmware, device availability, Python bindings, model files, and camera — with color-coded PASS/WARN/FAIL output.

| Problem | Solution |
|---|---|
| `hailortcli` not found | Run `./install_hailo.sh` and reboot |
| `hailortcli fw-control identify` fails | Check M.2 seating, PCIe ribbon cable, and that the Hat+ is powered |
| PCIe shows Gen 2 speed (5GT/s) | Verify `dtparam=pciex1_gen=3` is in `/boot/firmware/config.txt` under `[all]` and reboot |
| GStreamer pipeline fails to open | Check camera connection with `rpicam-hello` first; ensure Hailo GStreamer plugins are installed |
| `.hef` model errors | Confirm the model was compiled for `hailo10h` arch — Hailo-8 models are **not** compatible |
| `HAILO_NOT_IMPLEMENTED` error | The `.hef` model requires a newer `hailort` version (e.g., model needs 5.2.0, system has 5.1.1). Update `hailort` (`sudo apt update && sudo apt install h10-hailort`) or recompile the model. |
| Poor thermal performance | Ensure active cooler is connected and thermal pad contacts the Hailo module |
