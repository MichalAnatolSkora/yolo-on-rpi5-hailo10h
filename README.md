# YOLO on Raspberry Pi 5 with Hailo-10H

> **If this repo saved you hours of Hailo configuration headaches, please give it a :star: — it helps others find it too!**

One-click setup for running YOLO object detection on Raspberry Pi 5 with the Hailo-10H AI accelerator. Also runs locally on macOS / Linux / Windows laptops for development and testing.

## Run Locally on MacBook / Laptop

No Hailo hardware needed. Uses Ultralytics with CPU or Apple Silicon MPS acceleration.

```bash
# Install dependencies
pip install opencv-python ultralytics numpy

# Object detection (uses built-in webcam)
python run_yolo11.py --model yolo11n.pt --source 0 --display

# Vehicle tracking
python run_yolo11_tracking.py --model yolo11n.pt --source 0 --display --all-classes
```

Ultralytics auto-downloads `yolo11n.pt` on first run. Use `--source 0` for the default webcam (or `1`, `2` for other cameras). On Apple Silicon Macs, MPS acceleration is used automatically when available.

The backend is selected by model file extension:
| Extension | Backend | Where it runs |
|---|---|---|
| `.pt` | Ultralytics | CPU / MPS / CUDA (any platform) |
| `.onnx` | Ultralytics | CPU / MPS / CUDA (any platform) |
| `.hef` | HailoRT | Hailo-10H NPU (Raspberry Pi) |

## Raspberry Pi 5 + Hailo-10H Setup

### Hardware Requirements
| Component | Notes |
|---|---|
| **Raspberry Pi 5** | 4GB or 8GB RAM recommended |
| **Hailo-10H NPU module** | Via M.2 Hat+ or similar PCIe expansion board |
| **Active Cooler for Raspberry Pi 5** | Required — significant heat generation |
| **Thermal Pad** | Transfers heat from Hailo-10H to the expansion board |
| **Raspberry Pi Camera** *(optional)* | For real-time inference demos |

### Quick Start

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

## YOLO Object Detection

Four model sizes are available, installed via a single script with a variant argument:

| Model | Install command | Size | Notes |
|---|---|---|---|
| YOLOv11n (nano) | `./install_yolo11.sh` or `./install_yolo11.sh n` | ~4.9 MB | Fastest, good for real-time on RPi 5 |
| YOLOv11m (medium) | `./install_yolo11.sh m` | ~20 MB | Balanced accuracy and speed |
| YOLOv11l (large) | `./install_yolo11.sh l` | ~25 MB | High accuracy, slower |
| YOLOv11x (extra-large) | `./install_yolo11.sh x` | ~46 MB | Highest accuracy, slowest |

Install and run:

```bash
# Nano (default — recommended for real-time)
./install_yolo11.sh
python run_yolo11.py --display

# Medium (balanced)
./install_yolo11.sh m
python run_yolo11.py --model ~/hailo_models/yolov11m.hef --display

# Large (high accuracy)
./install_yolo11.sh l
python run_yolo11.py --model ~/hailo_models/yolov11l.hef --display

# Extra-large (highest accuracy — may not hit real-time FPS)
./install_yolo11.sh x
python run_yolo11.py --model ~/hailo_models/yolov11x.hef --display
```

The install script downloads a pre-compiled HEF from Hailo Model Zoo and sets up a Python virtual environment. Idempotent — safe to re-run. All variants share the same venv and dependencies.

**Camera input resolution:**

| Flag | Resolution |
|---|---|
| `--input-small` | 640x480 (default) |
| `--input` | 1024x768 |
| `--input-large` | 1280x720 |

Higher input resolution gives the model more detail to work with but uses more CPU for rescaling.

**Display preview:**

| Flag | Resolution |
|---|---|
| `--display-small` | 640x480 |
| `--display` | 1024x768 |
| `--display-large` | 1280x720 |

Without any display flag the script runs headless (no preview window).

**Other options:**
```bash
python run_yolo11.py --display --source /dev/video0              # USB camera (Linux)
python run_yolo11.py --display --source 0                        # webcam by index (macOS / Linux)
python run_yolo11.py --input-large --display-large               # high-res capture + preview
python run_yolo11.py --display --confidence 0.4                  # lower threshold
python run_yolo11.py --model yolo11n.pt --source 0 --display     # local mode (.pt model)
```

To find your USB camera device path on Linux: `ls /dev/video*` or `v4l2-ctl --list-devices`.

## Vehicle Tracking & Counting

Track and count vehicles (cars, motorcycles, buses, trucks) crossing a configurable line using YOLO detection + IoU-based tracking. Works on both Hailo NPU and locally.

```bash
# Raspberry Pi + Hailo
python run_yolo11_tracking.py --display                      # count vehicles going down, line at 50%
python run_yolo11_tracking.py --display --line-y 0.6         # line at 60% of frame height
python run_yolo11_tracking.py --display --direction both     # count both directions

# MacBook / laptop
python run_yolo11_tracking.py --model yolo11n.pt --source 0 --display --all-classes
```

By default only vehicles (car, motorcycle, bus, truck) are tracked. Use `--all-classes` to track all COCO objects — handy for testing indoors without vehicles.

**Tracking options:**

| Flag | Default | Description |
|---|---|---|
| `--confidence` | `0.3` | Detection threshold (lower = more stable tracking) |
| `--line-y` | `0.5` | Counting line Y position (0.0 = top, 1.0 = bottom) |
| `--direction` | `down` | Count direction: `down`, `up`, or `both` |
| `--max-disappeared` | `50` | Frames before a lost track is removed |
| `--min-iou` | `0.15` | Minimum IoU overlap to match detection to track |
| `--all-classes` | off | Track all detected objects, not just vehicles |
| `--deduplicate` | off | Remove overlapping detections before tracking |

All camera/display/model flags from `run_yolo11.py` are supported (`--display-large`, `--input-large`, `--source`, `--model`, `--confidence`, etc.).

## Security Camera: Person Line-Crossing Alert

Tracks persons and fires a REST API alert (or mock log) when someone crosses a configurable trip-wire line. Works on both Hailo NPU and locally.

```bash
# Raspberry Pi + Hailo
python security_cameras/person_line_alert.py --display
python security_cameras/person_line_alert.py --display --line-y 0.6 --direction both

# MacBook / laptop
python security_cameras/person_line_alert.py --model yolo11n.pt --source 0 --display

# With a real webhook endpoint
python security_cameras/person_line_alert.py --display --webhook-url http://my-server/api/alert
```

Only persons (COCO class 0) are tracked. When a person crosses the line, a JSON payload is POSTed to the webhook URL. Without `--webhook-url`, alerts are logged to the console (mock mode).

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--confidence` | `0.4` | Detection threshold |
| `--line-y` | `0.5` | Trip-wire Y position (0.0 = top, 1.0 = bottom) |
| `--line-margin` | `30` | Half-height of crossing zone in pixels |
| `--direction` | `both` | Alert direction: `down`, `up`, or `both` |
| `--webhook-url` | none | REST endpoint to POST alerts to (mock mode if omitted) |

### v2: Multi-Line Crossing with Interactive Setup

The v2 script adds support for **arbitrary-angle trip-wire lines**, **multiple lines**, and an **interactive setup mode** where you draw lines directly on the camera feed.

**1. Draw your lines (setup mode):**

```bash
# Opens camera feed — click two points per line, press Enter to save
python security_cameras/person_line_alert_v2.py --setup --source 0

# Custom config file path
python security_cameras/person_line_alert_v2.py --setup --source 0 --config my_lines.json
```

In setup mode:
- **Click** two points to define a trip-wire line
- **Enter** — save config and exit
- **u** — undo last line
- **Esc** — cancel current in-progress line
- **q** — quit without saving

**2. Run detection:**

```bash
# Raspberry Pi + Hailo
python security_cameras/person_line_alert_v2.py --config line_config.json --display

# MacBook / laptop
python security_cameras/person_line_alert_v2.py --config line_config.json --model yolo11n.pt --source 0 --display

# With a real webhook endpoint
python security_cameras/person_line_alert_v2.py --config line_config.json --display --webhook-url http://my-server/api/alert
```

The config file stores lines as normalized coordinates (0.0–1.0), so it works at any resolution:

```json
{
  "lines": [
    {"name": "entrance", "p1": [0.15, 0.52], "p2": [0.78, 0.45], "direction": "both"},
    {"name": "exit_door", "p1": [0.80, 0.10], "p2": [0.85, 0.90], "direction": "positive"}
  ],
  "webhook_url": ""
}
```

Line names, directions (`both`, `positive`, `negative`), and webhook URL can be edited directly in the JSON. The arrow overlay on each line shows the "positive" crossing direction.

**Buffer zone:** By default a person must cross the line exactly. Use `--buffer <pixels>` to create a zone around the line — as soon as someone enters this zone, they're counted. The buffer is shown as a semi-transparent strip in both setup and detection mode.

```bash
# Setup with buffer preview
python security_cameras/person_line_alert_v2.py --setup --source 0 --buffer 40

# Detection with buffer
python security_cameras/person_line_alert_v2.py --config line_config.json --model yolo11n.pt --source 0 --display --buffer 40
```

**v2 options:**

| Flag | Default | Description |
|---|---|---|
| `--setup` | off | Interactive line drawing mode |
| `--config` | none | Path to line config JSON (required for detection) |
| `--confidence` | `0.4` | Detection threshold |
| `--buffer` | `0` | Buffer zone in pixels — person counted when entering this distance from line (0 = exact crossing) |
| `--webhook-url` | none | REST endpoint to POST alerts (overrides config value) |
| `--max-disappeared` | `60` | Frames before a lost track is removed |
| `--min-iou` | `0.20` | Minimum IoU overlap for track matching |
| `--max-distance` | `150` | Maximum pixel distance for fallback matching |

## Gesture Recognition (WIP)

> **This feature is experimental and not yet functional.** The code and install script are included for reference, but gesture recognition has not been validated end-to-end. Contributions welcome.

The goal is to detect 18 hand gestures (HaGRID dataset) and map them to configurable actions — shell commands, on-screen messages, CSV logging.

The gesture model is separate from the object detection model — different dataset (HaGRID vs COCO), different weights. The install script (`install_yolo11_gestures.sh`) sets up the dataset structure and attempts to train + compile a Hailo-10H HEF, but this pipeline has not been fully tested.

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

## Why YOLOv11 and not YOLOv12?

This repo uses **YOLOv11n** instead of the newer YOLOv12n. Here's why:

The Hailo Model Zoo includes a `yolov12n.hef`, but it is compiled with **DFC v5.2.0** and requires **HailoRT 5.2.0**. As of March 2025, Raspberry Pi OS ships **HailoRT 5.1.1** via the `hailo-h10-all` package — there is no 5.2.0 package available in the Raspberry Pi apt repository yet.

When you try to run a v5.2.0-compiled HEF on HailoRT 5.1.1, you get a `HAILO_NOT_IMPLEMENTED` error (error code 7). The HEF file loads fine, but the runtime cannot execute it because it contains operators or features introduced in 5.2.0 that the 5.1.1 runtime doesn't support.

On top of the version mismatch, YOLOv12 itself uses an attention-based architecture (replacing the pure CNN design of YOLOv8/v11), which may require additional Hailo compiler support beyond just matching the HailoRT version.

**YOLOv11n works perfectly** — the Hailo Model Zoo provides a `yolov11n.hef` compiled with DFC v5.1.0, which is fully compatible with HailoRT 5.1.1 on Raspberry Pi. It runs via the InferModel async API with on-chip NMS and delivers real-time performance.

**TL;DR:** The Raspberry Pi `hailo-h10-all` package is at version 5.1.1, and YOLOv12 HEFs need 5.2.0. Use YOLOv11n until Raspberry Pi ships an updated HailoRT package. Switching will be a one-line model path change.

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
