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

## Acquiring Hailo-10H YOLO Models (`.hef`)

**Models compiled for Hailo-8 will NOT work on Hailo-10H.** You must use `.hef` files specifically compiled for the Hailo-10 architecture.

**Option A — Automated (YOLOv12):**
```bash
./install_yolo12.sh
```
Downloads a pre-compiled YOLOv12n HEF from Hailo Model Zoo and installs Python dependencies. Idempotent — safe to re-run.

**Option B — Manual download:**

Pre-compiled HEF files for Hailo-10H are available from the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) S3 bucket (compiled with DFC v5.2.0):

| Model | Size | Download |
|---|---|---|
| YOLOv12n | ~5.5 MB | [yolov12n.hef](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/yolov12n.hef) |
| YOLOv11n | ~4.9 MB | [yolov11n.hef](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/yolov11n.hef) |
| YOLOv8n | ~6.7 MB | [yolov8n.hef](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/yolov8n.hef) |
| YOLOv8s | ~13.1 MB | [yolov8s.hef](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/yolov8s.hef) |
| YOLOv8m | — | [yolov8m.hef](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/yolov8m.hef) |

URL pattern: `https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/<model_name>.hef`

Place downloaded files in `~/hailo_models/`.

## Running Real-Time Inference (Camera)

If you have a Raspberry Pi Camera connected, you can test YOLO object detection natively using `rpicam-apps` which comes pre-integrated with Hailo post-processing.

```bash
rpicam-hello -t 0 --post-process-file /usr/share/rpi-camera-assets/hailo/yolov8s.json --info-text "Hailo-10H YOLO"
```

*Note: You may need to edit the JSON config to point to the correct `.hef` file path compiled for Hailo-10H.*

## Running Inference via Python

Two inference scripts are provided:

| Script | Backend | Best for |
|---|---|---|
| `run_yolo.py` | GStreamer pipeline | YOLOv8 with existing Hailo GStreamer post-process plugins |
| `run_yolo12.py` | HailoRT Python API | YOLOv12 (or any model without a GStreamer .so) |
| `run_gestures.py` | HailoRT + action engine | Hand gesture recognition with configurable triggers |

**Install Python dependencies:**
```bash
pip install -r requirements.txt
```

### YOLOv8 (GStreamer)

```bash
python run_yolo.py --model ~/hailo_models/yolov8n.hef
python run_yolo.py --model ~/hailo_models/yolov8n.hef --source /dev/video0  # USB camera
```

### YOLOv12 (HailoRT API)

```bash
python run_yolo12.py --model ~/hailo_models/yolov12n.hef
python run_yolo12.py --model ~/hailo_models/yolov12n.hef --source /dev/video0  # USB camera
python run_yolo12.py --model ~/hailo_models/yolov12n.hef --confidence 0.4 --iou 0.5
```

To find your USB camera device path, run `ls /dev/video*` or `v4l2-ctl --list-devices`.

### Gesture Recognition

Detects 18 hand gestures (HaGRID dataset) and maps them to configurable actions — shell commands, on-screen messages, CSV logging.

**Do I need to run this if I already ran `install_yolo12.sh`?**
Yes. The `install_yolo12.sh` script installs a general-purpose object detection model (YOLOv12n trained on COCO — 80 classes like person, car, dog). The gesture model is a **separate model** trained specifically on hand gestures (18 HaGRID classes like thumbs up, peace, fist). They use different datasets, different weights, and produce different `.hef` files. The install script will reuse the existing virtual environment and Hailo Model Zoo if you already ran `install_yolo12.sh`.

**Install the gesture model:**
```bash
./install_yolo12_gestures.sh
```
This sets up HaGRID data, trains YOLOv12n on gestures, and compiles a Hailo-10H HEF. You'll need to supply training images from [HaGRID](https://github.com/hukenovs/hagrid) or [Roboflow](https://universe.roboflow.com) — the script will prompt you.

**Run gesture recognition:**
```bash
python run_gestures.py --model ~/hailo_models/yolov12n_gestures.hef
python run_gestures.py --model ~/hailo_models/yolov12n_gestures.hef --source /dev/video0
python run_gestures.py --model ~/hailo_models/yolov12n_gestures.hef --headless --log-csv gestures.csv
```

**Configure actions** by editing `gesture_actions.yaml`:
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

Features:
- **Hold detection** — gesture must be held for a configurable duration before triggering
- **Cooldown** — prevents rapid re-firing of the same gesture
- **Action triggers** — run arbitrary shell commands on gesture detection
- **HUD overlay** — FPS, active hand count, gesture history, hold progress bars
- **Headless mode** — run without a display (SSH, automation, embedded)
- **CSV logging** — append every triggered gesture to a CSV file for analysis

> **Note:** Use the system Python if Hailo packages were installed globally via `apt`, or a virtual environment with `--system-site-packages` to inherit them.

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
| Poor thermal performance | Ensure active cooler is connected and thermal pad contacts the Hailo module |
