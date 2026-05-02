# YOLO on Raspberry Pi 5 with Hailo-10H

> **If this repo saved you hours of Hailo configuration headaches, please give it a :star: — it helps others find it too!**

One-click setup for running YOLO object detection on Raspberry Pi 5 with the Hailo-10H AI accelerator. Also runs locally on macOS / Linux / Windows laptops for development and testing.

## Quick Start: Vehicle Counting

The primary workflow — count vehicles crossing a line you draw on the camera image.

1. **Set up counting lines** — draw trip-wire lines interactively on a frame from your camera. Saves to `line_config.json`:
   ```bash
   python run_yolo11_tracking.py --setup --source 0
   ```
   Click two points per line, press **Enter** to save, **q** to quit. See [Multi-line counting with interactive setup](#multi-line-counting-with-interactive-setup).

2. **Start counting + recording** — config is auto-loaded from `./line_config.json`. Adds an MP4 recording of the annotated preview:
   ```bash
   python run_yolo11_tracking.py --source 0 --display --record traffic.mp4
   ```
   Press **q** in the preview window to stop. See [Recording the preview to video](#recording-the-preview-to-video).

3. **(optional) Build a ground-truth dataset** — record raw clips (or download from YouTube), manually count vehicles, then evaluate the tracker offline against the count:
   ```bash
   # Step 1a: capture a clean clip from your camera (no overlays)
   python evaluation/record_raw.py --display --duration 60 --output raw_morning.mp4

   # Step 1b (alternative): download a clip from a YouTube traffic livestream
   python evaluation/download_clip.py "https://www.youtube.com/watch?v=..." --duration 120

   # Step 2: manually count vehicles per counting line, write expected.json
   echo '{"line_1": 12}' > expected.json

   # Step 3: re-run the tracker on the file and compare
   python evaluation/evaluate.py raw_morning.mp4 --expected expected.json --tolerance 1
   ```
   See [Evaluation & testing](#evaluation--testing).

4. **(optional) Tune tracker parameters** — once you have a clip + ground truth, the tunable knobs (`confidence`, `min_hits`, `max_distance`, ...) live in [`tracker_config.json`](tracker_config.json). Edit by hand, or have the [`tune-tracker` Claude Code agent](tuning/README.md) do coordinate descent for you.

On Raspberry Pi use `--source picam` (libcamera) or `--source /dev/video0` (USB). On macOS/Linux laptops add `--model yolo11n.pt` (local Ultralytics). Full options below.

For setup mode you can also point `--source` at a recorded video (e.g. `--source raw_morning.mp4 --frame 200`) — handy for drawing lines on the same clip you'll later evaluate.

**This repo also includes:** [object detection](#object-detection) without tracking, [security camera person line-crossing alerts](#security-camera-person-line-crossing-alert), and an experimental [gesture recognition](#gesture-recognition-wip) pipeline.

## Installation

### Raspberry Pi 5 + Hailo-10H

**Hardware Requirements**

| Component | Notes |
|---|---|
| **Raspberry Pi 5** | 4GB or 8GB RAM recommended |
| **Hailo-10H NPU module** | Via M.2 Hat+ or similar PCIe expansion board |
| **Active Cooler for Raspberry Pi 5** | Required — significant heat generation |
| **Thermal Pad** | Transfers heat from Hailo-10H to the expansion board |
| **Raspberry Pi Camera** *(optional)* | For real-time inference demos |

**Install:**

```bash
git clone https://github.com/MichalAnatolSkora/yolo-on-rpi5-hailo10h.git
cd yolo-on-rpi5-hailo10h
./install_hailo.sh
sudo reboot
```

The script auto-detects your hardware (Hailo-10H or Hailo-8), installs the correct driver package, enables PCIe Gen 3, and blacklists conflicting drivers if needed.

**Verify:**

```bash
hailortcli fw-control identify
```
You should see "Identifying board..." followed by Hailo-10 structure and firmware version.

PCIe Gen 3 check (look for `Speed 8GT/s` — Gen 2 would show `5GT/s`):
```bash
sudo lspci -vv | grep -i hailo -A 20 | grep -i speed
```

Then install YOLO models — see [YOLO models](#yolo-models).

### Local development (macOS / Linux / Windows)

No Hailo hardware needed. Uses Ultralytics with CPU or Apple Silicon MPS acceleration.

```bash
pip install opencv-python ultralytics numpy
```

Sanity test (built-in webcam):
```bash
python run_yolo11.py --model yolo11n.pt --source 0 --display
```

Ultralytics auto-downloads `yolo11n.pt` on first run. On Apple Silicon Macs, MPS acceleration is used automatically.

## YOLO Models

The backend is selected by model file extension:

| Extension | Backend | Where it runs |
|---|---|---|
| `.pt` | Ultralytics | CPU / MPS / CUDA (any platform) |
| `.onnx` | Ultralytics | CPU / MPS / CUDA (any platform) |
| `.hef` | HailoRT | Hailo-10H NPU (Raspberry Pi) |

### Installing YOLOv11 on Raspberry Pi

Four model sizes are available, installed via a single script with a variant argument:

| Model | Install command | Size | Notes |
|---|---|---|---|
| YOLOv11n (nano) | `./install_yolo11.sh` or `./install_yolo11.sh n` | ~4.9 MB | Fastest, good for real-time on RPi 5 |
| YOLOv11m (medium) | `./install_yolo11.sh m` | ~20 MB | Balanced accuracy and speed |
| YOLOv11l (large) | `./install_yolo11.sh l` | ~25 MB | High accuracy, slower |
| YOLOv11x (extra-large) | `./install_yolo11.sh x` | ~46 MB | Highest accuracy, slowest |

```bash
# Nano (default — recommended for real-time)
./install_yolo11.sh
python run_yolo11.py --display

# Medium / Large / Extra-large — pass the variant
./install_yolo11.sh m
python run_yolo11.py --model ~/hailo_models/yolov11m.hef --display
```

The install script downloads a pre-compiled HEF from Hailo Model Zoo and sets up a Python virtual environment. Idempotent — safe to re-run. All variants share the same venv and dependencies.

### Other pre-compiled HEFs

Pre-compiled HEFs for Hailo-10H from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) (DFC v5.2.0):

> **⚠️ Version Compatibility:** The `.hef` model's compiled version must be compatible with your system's `hailort` package version. The models below require `hailort` **5.2.0**. If you use them on an older version (like 5.1.1), you will get a `HAILO_NOT_IMPLEMENTED` error. See [Why YOLOv11 and not YOLOv12](#why-yolov11-and-not-yolov12) for context.

| Model | Size | Download |
|---|---|---|
| YOLOv12n | ~5.5 MB | [yolov12n.hef](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/yolov12n.hef) |
| YOLOv11n | ~4.9 MB | [yolov11n.hef](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/yolov11n.hef) |
| YOLOv8n | ~6.7 MB | [yolov8n.hef](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/yolov8n.hef) |
| YOLOv8s | ~13.1 MB | [yolov8s.hef](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/yolov8s.hef) |
| YOLOv8m | — | [yolov8m.hef](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.2.0/hailo10h/yolov8m.hef) |

**Hailo-8 models will NOT work on Hailo-10H.** Use only `hailo10h`-compiled HEFs. Place files in `~/hailo_models/`.

## Object Detection

Plain object detection — bounding boxes + class labels, no tracking. Script: `run_yolo11.py`.

```bash
# Default: webcam, headless
python run_yolo11.py --display

# Local mode (.pt model, any platform)
python run_yolo11.py --model yolo11n.pt --source 0 --display

# Specific camera + higher resolution
python run_yolo11.py --display --source /dev/video0 --input-large
```

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
python run_yolo11.py --display --source 0                        # webcam by index (macOS / Linux)
python run_yolo11.py --input-large --display-large               # high-res capture + preview
python run_yolo11.py --display --confidence 0.4                  # lower threshold
```

To find your USB camera device path on Linux: `ls /dev/video*` or `v4l2-ctl --list-devices`.

## Vehicle Tracking & Counting

Track and count vehicles (cars, motorcycles, buses, trucks) crossing a configurable line using YOLO detection + IoU-based tracking. Script: `run_yolo11_tracking.py`. Works on both Hailo NPU and locally.

```bash
# Raspberry Pi + Hailo (single horizontal line at 50%)
python run_yolo11_tracking.py --display

# MacBook / laptop, all classes (handy for testing indoors)
python run_yolo11_tracking.py --model yolo11n.pt --source 0 --display --all-classes
```

By default only vehicles (car, motorcycle, bus, truck) are tracked. Use `--all-classes` to track all COCO objects.

### Multi-Line Counting with Interactive Setup

The primary mode — supports **arbitrary-angle counting lines**, **multiple lines at once**, and an **interactive setup mode** where you draw lines directly on the camera feed. Each line keeps its own per-direction count.

**1. Draw your lines (setup mode):**

```bash
# Opens camera feed — click two points per line, press Enter to save
python run_yolo11_tracking.py --setup --source 0

# Custom config file path
python run_yolo11_tracking.py --setup --source 0 --config my_lines.json

# From a recorded video file (uses the middle frame by default)
python run_yolo11_tracking.py --setup --source raw_morning.mp4

# Pick a specific frame from the video (handy if the middle frame is empty)
python run_yolo11_tracking.py --setup --source raw_morning.mp4 --frame 200
```

In setup mode:
- **Click** two points to define a counting line
- **Enter** — save config and exit
- **u** — undo last line
- **Esc** — cancel current in-progress line
- **q** — quit without saving

> **Want richer per-line metadata?** Two alternatives to the in-script `--setup` (which writes the simpler **v0**):
> - [`tools/setup_lines.py`](tools/setup_lines.py) — terminal prompts + cv2 window. Writes **schema v1**. Zero extra dependencies.
> - [`tools/visual_editor.py`](tools/visual_editor.py) — Streamlit web editor. Click two points to add a line, edit metadata in side panels, save v1. Run `pip install streamlit streamlit-image-coordinates pillow` then `streamlit run tools/visual_editor.py`.
>
> Both v0 and v1 configs are loaded transparently. To upgrade a v0 file to v1 by hand, see the diff snippet at the top of [`tools/setup_lines.py`](tools/setup_lines.py).

**2. Run counting:**

```bash
# Raspberry Pi + Hailo
python run_yolo11_tracking.py --config line_config.json --display

# MacBook / laptop
python run_yolo11_tracking.py --config line_config.json --model yolo11n.pt --source 0 --display

# With buffer zone (vehicles counted as soon as they enter the zone)
python run_yolo11_tracking.py --config line_config.json --display --buffer 20
```

The config file stores lines as normalized coordinates (0.0–1.0), so it works at any resolution:

```json
{
  "lines": [
    {"name": "northbound", "p1": [0.10, 0.55], "p2": [0.95, 0.55], "direction": "both"},
    {"name": "driveway", "p1": [0.24, 0.07], "p2": [0.26, 0.96], "direction": "positive"}
  ]
}
```

Line names and directions (`both`, `positive`, `negative`) can be edited directly in the JSON. The arrow overlay on each line shows the "positive" crossing direction; per-line counts appear next to the line label.

When `--config` is provided (or `./line_config.json` exists in the working directory — it is auto-loaded), the multi-line counter is used instead of the legacy `--line-y` horizontal line.

#### How `direction` works

A line has two endpoints, **p1** and **p2** (the order matters). The "positive" side of the line is **the left side when you stand at p1 looking towards p2**. The render shows this as a small arrow on the line — that arrow points to the positive side.

```
        ↑ arrow points here = "positive" side
        |
   p1 ●═══════════════● p2
        |
        ↓ "negative" side
```

When a tracked vehicle's centroid moves between frames, its position relative to the line is computed via the cross-product sign:

- centroid was on **negative** side → now on **positive** side: counted as **"positive"** crossing
- centroid was on **positive** side → now on **negative** side: counted as **"negative"** crossing

The `direction` field in the JSON controls which crossings are *kept*:

| Setting | Counts |
|---|---|
| `"both"` | both directions |
| `"positive"` | only crossings going **to** the positive side (in the arrow's direction) |
| `"negative"` | only crossings going **to** the negative side (against the arrow) |

**Practical example.** A road with traffic going both ways. You draw `p1` on the left curb and `p2` on the right curb. Looking from p1 to p2, "positive" (the arrow) points *up* the page (north). Cars going north count as "positive"; cars going south count as "negative".

Want to count only one direction? Set `"direction": "positive"` (or `"negative"`). To swap which way is "positive" without redrawing, just **swap p1 and p2** — the arrow flips.

Per-line counts in the overlay show the *total* (positive + negative) regardless of the filter, but only filtered crossings are added.

### Legacy single horizontal line

When neither `--config` nor `./line_config.json` is present, the tracker falls back to a single horizontal counting line at the Y position you choose:

```bash
python run_yolo11_tracking.py --display                      # line at 50%, count "down"
python run_yolo11_tracking.py --display --line-y 0.6         # line at 60% of frame height
python run_yolo11_tracking.py --display --direction both     # count both directions
```

| Flag | Default | Description |
|---|---|---|
| `--line-y` | `0.5` | Counting line Y position (0.0 = top, 1.0 = bottom) |
| `--line-margin` | `40` | Half-height of counting zone in pixels |
| `--direction` | `down` | Count direction: `down`, `up`, or `both` |
| `--no-config` | off | Force legacy mode even if `line_config.json` is present |

### Recording the preview to video

Use `--record` to save an MP4 of the annotated preview (with bounding boxes, lines, counts, and labels rendered in). Works in both legacy and multi-line modes, with or without `--display`.

```bash
# Auto-named file: recording_YYYYMMDD_HHMMSS.mp4 in the current directory
python run_yolo11_tracking.py --display --record

# Custom output path
python run_yolo11_tracking.py --display --record out.mp4

# Headless recording (no preview window)
python run_yolo11_tracking.py --record traffic.mp4

# Custom frame rate
python run_yolo11_tracking.py --display --record --record-fps 30
```

| Flag | Default | Description |
|---|---|---|
| `--record [PATH]` | off | Record annotated frames to MP4. Path optional — auto-timestamped if omitted. |
| `--record-fps` | `30` | Frame rate written to the file (matches RPi5 + Hailo real-time throughput) |

The recording is at the full capture resolution (`--input` / `--input-large`), not the resized `--display` size, so detail isn't lost. Stop with `q` in the preview window or `Ctrl+C` — the file is finalized on exit. Codec fallback is `avc1` → `mp4v` → `MJPG/.avi`, so files open in QuickTime / VLC / mpv without re-encoding.

### Tracker tuning options

All camera/display/model flags from `run_yolo11.py` are also supported.

The tunable defaults below are loaded from [`tracker_config.json`](tracker_config.json) at the repo root — edit that file to change defaults for *all* runs. CLI flags below still override the JSON for one-off experiments. Full reference (with code links and tuning advice): [`tuning/PARAMETERS.md`](tuning/PARAMETERS.md).

#### Detection — what YOLO emits

- **`--confidence`** (config: `confidence`, default `0.3`) — How sure the model has to be that a box is really an object before it counts. `0.3` means "30% confidence or above". Lower → sees more (including blurry/distant objects but also more false detections); higher → stricter, fewer false positives but you may miss real ones.

- **`--iou`** (config: `iou`, default `0.45`) — When YOLO emits two overlapping boxes for what is probably the same object, *non-max suppression* (NMS) keeps only the higher-confidence one and drops the other. This number is the overlap threshold for "probably the same object" — measured as IoU (intersection-over-union: shared area / total area, 0.0 to 1.0). `0.45` = drop the worse box if they overlap by 45% or more. Rarely worth tuning.

- **`--all-classes`** (off by default) — Off: count only vehicles (car, motorcycle, bus, truck). On: count every COCO class YOLO detects (people, animals, etc.) — handy when testing indoors with a webcam.

- **`--no-deduplicate`** (config: `deduplicate: false`, off by default — i.e. dedup is **on**) — YOLO sometimes emits two boxes for the same vehicle with *different class labels* (e.g. one says "car", the other says "truck" on the same SUV). Dedup removes these duplicates. Leave on. Flip off only to see what YOLO emits raw.

#### Tracking — connecting detections across frames

- **`--min-iou`** (config: `min_iou`, default `0.15`) — When a new detection arrives, the tracker tries to match it to an existing track by box overlap (IoU). Below this threshold the detection is treated as a *new* vehicle (new track ID). Higher → stricter matching, fewer ID swaps mid-track, but a real fast-moving vehicle may get a new ID every few frames (= overcounting).

- **`--max-distance`** (config: `max_distance`, default `200`) — Fallback when IoU matching fails (boxes don't overlap at all because the vehicle moved too fast between frames). The tracker then matches by how far the box *center* moved, in pixels. Above this distance, no match. Higher → rescues fast vehicles; too high → a brand-new vehicle near a lost track inherits the old ID. **Not normalized to resolution** — `200` was tuned for 1024×768; halve for 640×480, double for 1920×1080.

- **`--max-disappeared`** (config: `max_disappeared`, default `50`) — A track is being followed; the vehicle then briefly disappears (passes behind a tree, goes out of frame). This is how many frames the tracker holds onto the track before giving up and deleting it. At 30 fps, `50` ≈ 1.7 sec grace period. Higher → survives longer occlusions; too high → a different vehicle appearing later may get matched to the old track.

#### Counting — when a track triggers a crossing

- **`--min-hits`** (config: `min_hits`, default `3`) — A track must appear in at least this many frames before its line crossings start counting. Filters out 1- or 2-frame *ghost detections* (flickers from noise) that would otherwise inflate counts. **The biggest knob for fixing overcounting** — bump up (3 → 5 → 7) when actual count > expected.

- **`--buffer`** (config: `buffer`, default `0`) — By default a vehicle is counted only when its centroid *exactly* crosses the geometric line. Set this to N pixels to count a vehicle when its centroid enters a strip of N pixels *around* the line. Useful when crossings are sometimes missed because the track briefly disappears right at the line. Most setups leave at `0`.

#### Where to go next

For systematic tuning against a ground-truth count, see [`tuning/`](tuning/README.md) — a Claude Code agent does coordinate descent over these knobs automatically. For reproducible regression testing across multiple clips, use [`evaluation/run_suite.py`](evaluation/run_suite.py).

## Evaluation & Testing

Reproducible tracker tuning workflow: record a clip → manually count → replay offline → compare. The `evaluation/` folder has four scripts:

- `download_clip.py` — grab clips from YouTube / Twitch (no camera trip needed)
- `record_raw.py` — capture clean clips from your own camera
- `evaluate.py` — replay through the tracker, compare to ground truth, exit 0/1 (PASS/FAIL) — usable as a CI regression check
- `run_suite.py` — run `evaluate.py` over a whole folder of clips (`evaluation/tests/*.mp4` + `*.expected.json`) and detect regressions vs. the previous baseline

Full docs, flag tables, verified livestream URLs, and the end-to-end tuning loop: [docs/evaluation.md](docs/evaluation.md). For automatic *parameter* tuning (vs. testing) on a single clip, see [`tuning/`](tuning/README.md).

## Security Camera: Person Line-Crossing Alert

Tracks persons and fires a REST API alert (or mock log) when someone crosses a configurable trip-wire line. Works on both Hailo NPU and locally. The v2 script supports arbitrary-angle lines, multiple lines, and an interactive setup mode for drawing them on the camera feed.

```bash
# Quick start (single line, persons only)
python security_cameras/person_line_alert.py --display --webhook-url http://my-server/api/alert

# v2: draw lines interactively, then run detection
python security_cameras/person_line_alert_v2.py --setup --source 0
python security_cameras/person_line_alert_v2.py --config line_config.json --display
```

Full docs, flag tables, config format, and v2 setup workflow: [docs/security-camera.md](docs/security-camera.md).

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

## Reference

### Why YOLOv11 and not YOLOv12?

This repo uses **YOLOv11n** instead of the newer YOLOv12n. Here's why:

The Hailo Model Zoo includes a `yolov12n.hef`, but it is compiled with **DFC v5.2.0** and requires **HailoRT 5.2.0**. As of March 2025, Raspberry Pi OS ships **HailoRT 5.1.1** via the `hailo-h10-all` package — there is no 5.2.0 package available in the Raspberry Pi apt repository yet.

When you try to run a v5.2.0-compiled HEF on HailoRT 5.1.1, you get a `HAILO_NOT_IMPLEMENTED` error (error code 7). The HEF file loads fine, but the runtime cannot execute it because it contains operators or features introduced in 5.2.0 that the 5.1.1 runtime doesn't support.

On top of the version mismatch, YOLOv12 itself uses an attention-based architecture (replacing the pure CNN design of YOLOv8/v11), which may require additional Hailo compiler support beyond just matching the HailoRT version.

**YOLOv11n works perfectly** — the Hailo Model Zoo provides a `yolov11n.hef` compiled with DFC v5.1.0, which is fully compatible with HailoRT 5.1.1 on Raspberry Pi. It runs via the InferModel async API with on-chip NMS and delivers real-time performance.

**TL;DR:** The Raspberry Pi `hailo-h10-all` package is at version 5.1.1, and YOLOv12 HEFs need 5.2.0. Use YOLOv11n until Raspberry Pi ships an updated HailoRT package. Switching will be a one-line model path change.

### Troubleshooting

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
| Camera not authorized on macOS | *System Settings → Privacy & Security → Camera* → enable Terminal/iTerm/VS Code, then restart the terminal |
| Recording file won't open in QuickTime | The `mp4v` codec was used as fallback. Open in VLC instead, or re-encode with `ffmpeg -i in.mp4 -c:v libx264 out.mp4` |
| Poor thermal performance | Ensure active cooler is connected and thermal pad contacts the Hailo module |
