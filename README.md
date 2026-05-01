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

On Raspberry Pi use `--source picam` (libcamera) or `--source /dev/video0` (USB). On macOS/Linux laptops add `--model yolo11n.pt` (local Ultralytics). Full options below.

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

GStreamer pipeline (for YOLOv8 with Hailo post-process plugins):
```bash
python run_yolo.py --model ~/hailo_models/yolov8n.hef
```

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
```

In setup mode:
- **Click** two points to define a counting line
- **Enter** — save config and exit
- **u** — undo last line
- **Esc** — cancel current in-progress line
- **q** — quit without saving

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

| Flag | Default | Description |
|---|---|---|
| `--confidence` | `0.3` | Detection threshold (lower = more stable tracking) |
| `--iou` | `0.45` | NMS IoU threshold |
| `--max-disappeared` | `50` | Frames before a lost track is removed |
| `--min-iou` | `0.15` | Minimum IoU overlap to match detection to track |
| `--max-distance` | `200` | Max centroid distance fallback (catch fast movers) |
| `--min-hits` | `3` | Frames a track must be seen before its crossings count — prevents ghost detections from inflating counts |
| `--all-classes` | off | Track all detected objects, not just vehicles |
| `--no-deduplicate` | off | Disable overlap dedup (on by default: suppresses same-class overlaps and cross-vehicle-class overlaps like "car + truck" on one box) |
| `--buffer` | `0` | Buffer zone in pixels around each multi-line trip-wire (0 = exact crossing) |

For tuning these in a reproducible way, use the [evaluation tooling](#evaluation--testing).

## Evaluation & Testing

When tuning the tracker, the most reliable test is: record a clip, manually count the vehicles, then run the tracker on the recording offline and compare. The `evaluation/` folder has three scripts that compose into this workflow:

- [`download_clip.py`](#downloading-test-footage) — grab a clip from YouTube / Twitch / etc. (no camera trip needed)
- [`record_raw.py`](#recording-ground-truth-clips) — capture clean (no annotation) clips from your own camera
- [`evaluate.py`](#offline-evaluation) — replay a clip through the tracker, compare to ground truth, exit 0/1 (PASS/FAIL)

The full loop is documented in [End-to-end tuning workflow](#end-to-end-tuning-workflow).

### Downloading test footage

Don't have a camera pointed at a busy street? Grab clips from YouTube traffic livestreams or any other yt-dlp supported source (Twitch, Vimeo, etc.) — same workflow, no recording trip required.

Requires **yt-dlp** + **ffmpeg** installed system-wide:

```bash
brew install yt-dlp ffmpeg               # macOS
sudo apt install yt-dlp ffmpeg           # Debian/Ubuntu/RPi
pip install yt-dlp                       # any platform (still needs ffmpeg)
```

Examples:

```bash
# 2 minutes from a YouTube video — auto-named raw_yt_<id>_<timestamp>.mp4
python evaluation/download_clip.py "https://www.youtube.com/watch?v=VIDEO_ID" --duration 120

# Slice 5 minutes from a longer VOD, starting at 10 minutes in
python evaluation/download_clip.py "URL" --start 600 --duration 300

# Live stream — record 60s from "now" (skip backlog)
python evaluation/download_clip.py "URL" --live --duration 60

# Live stream — start from the very beginning of the broadcast
python evaluation/download_clip.py "URL" --live-from-start --duration 120

# Custom output path
python evaluation/download_clip.py "URL" --duration 120 --output raw_intersection.mp4

# Lower quality for faster downloads / smaller files (default 720p)
python evaluation/download_clip.py "URL" --duration 120 --height 480

# Inspect available formats without downloading
python evaluation/download_clip.py "URL" --list-formats

# Batch — one URL per line in urls.txt; same duration applied to each
python evaluation/download_clip.py --batch urls.txt --duration 120
```

`urls.txt` format:
```
# Morning rush — Times Square live cam
https://www.youtube.com/watch?v=...
# Tokyo Shibuya
https://www.youtube.com/watch?v=...
```

| Flag | Default | Description |
|---|---|---|
| `url` (positional) | (required unless `--batch`) | Source URL — anything yt-dlp supports |
| `--batch FILE` | none | Process a list of URLs (one per line, `#` comments) |
| `--output` / `-o` | auto: `raw_yt_<id>_<timestamp>.mp4` | Output path (single-URL mode) |
| `--duration` | full | Clip length in seconds |
| `--start` | `0` | Offset for VODs |
| `--live` | off | Treat URL as live stream (record from now) |
| `--live-from-start` | off | Live stream from beginning of broadcast |
| `--height` | `720` | Max resolution — higher = bigger files & slower inference |
| `--list-formats` | off | Show available qualities/codecs and exit |

**Verified-working YouTube live streams** (all tested April 2026 with `yt-dlp --print is_live` — URLs may rotate over time, re-search if any goes offline):

*Best for highway counting* (multi-lane, clean traffic flow, no pedestrian noise):

| Stream | URL | Notes |
|---|---|---|
| **I-70 Eisenhower Tunnel** (CO, USA) | `https://www.youtube.com/watch?v=ckvtwmD7tq8` | 2 lanes each way, mountain interstate, clean separation — ideal for directional counts |
| **I-25 Monument Hill** (CO, USA) | `https://www.youtube.com/watch?v=T2_fTYxVBx8` | 2+2 lanes interstate, north/southbound view, classic divided highway |
| **SF–Oakland Bay Bridge** (CA, USA) | `https://www.youtube.com/watch?v=CXYr04BWvmc` | High-traffic highway 24/7 (ABC7 News), side-on view |

*Best for intersection counting* (clear roads, traffic crosses the frame):

| Stream | URL | Notes |
|---|---|---|
| **Jackson Hole Town Square**, WY | `https://www.youtube.com/watch?v=1EiC9bvVGnk` | 4-way intersection, day/night cycle, moderate traffic |
| **Scottsville Main Street**, KY | `https://www.youtube.com/watch?v=MsiHVaomJ04` | Small-town main street (EarthCam), lighter traffic |
| **Walworth Road**, London | `https://www.youtube.com/watch?v=8JCk5M_xrBs` | Two-way urban road, straightforward angle |

*Mixed traffic + pedestrians* (also good for testing person tracking):

| Stream | URL | Notes |
|---|---|---|
| **Times Square 4K**, NYC | `https://www.youtube.com/watch?v=rnXIjl_Rzy4` | EarthCam 4K — vehicles + heavy foot traffic |
| **Abbey Road Crossing**, London | `https://www.youtube.com/watch?v=M3EYAY2MftI` | Iconic zebra crossing — cars stop for pedestrians |
| **Shibuya Scramble**, Tokyo | `https://www.youtube.com/watch?v=dfVK7ld38Ys` | Busiest pedestrian crossing in the world (FNN) |
| **Dublin, Ireland** | `https://www.youtube.com/watch?v=3nyPER2kzqk` | Street-level European city center (EarthCam) |
| **New Orleans Street View** | `https://www.youtube.com/watch?v=Ksrleaxxxhw` | French Quarter street life (EarthCam) |

*City panoramas* (lower vehicle resolution, but useful for wide-angle tests):

| Stream | URL | Notes |
|---|---|---|
| **Berlin Alexanderplatz** | `https://www.youtube.com/watch?v=IRqboacDNFg` | Panoramic Berlin (Livespotting) |
| **Destination Deutschland** | `https://www.youtube.com/watch?v=Li3Dvqlo5uE` | Rotating multi-cam across Germany (Feratel) |

**Quick start** with the most ground-truth-friendly one:
```bash
python evaluation/download_clip.py "https://www.youtube.com/watch?v=1EiC9bvVGnk" --live --duration 120 --output raw_jackson_hole.mp4
```

**Verify a URL is still live** before downloading a long clip:
```bash
yt-dlp --skip-download --print "live=%(is_live)s | %(title)s" "URL"
```

**More sources:**
- YouTube — search `"traffic camera live"`, `"intersection live cam"`, `"highway camera"` (and Polish: `"kamera skrzyżowanie live"`, `"rondo na żywo"`). Channels: *Earth Cam* (`@EarthCam/streams`), *VirtualRailfan*, *Skylinewebcams*, *See Jackson Hole*
- Pexels / Pixabay — free stock clips (CC0 license usually): https://www.pexels.com/search/videos/traffic/
- TrafficVision.live — directory of 140k+ traffic cameras worldwide: https://trafficvision.live/
- Academic datasets for serious benchmarking: **UA-DETRAC** (~100h Chinese traffic), **AI City Challenge**, **MIO-TCD**

**Legal note:** check each site's Terms of Service. Personal/research use of public livestreams is generally accepted; commercial redistribution usually isn't. Academic datasets and CC0 stock clips are explicitly free to use.

### Recording ground-truth clips

When you want a recording of your own camera (no detection, no overlays), use `record_raw.py`. The output is the truth — what the camera saw — without any annotation contaminating the frames.

```bash
# Quick recording, stop with Ctrl+C
python evaluation/record_raw.py

# Live preview + auto-stop after 60s
python evaluation/record_raw.py --display --duration 60

# Native Full HD on Raspberry Pi camera, 2 minutes
python evaluation/record_raw.py --source picam --input-fhd --duration 120 --output street_morning.mp4
```

Output filename defaults to `raw_YYYYMMDD_HHMMSS.mp4` so it's obvious there's no annotation in the file.

| Flag | Default | Description |
|---|---|---|
| `--source` | platform default | Camera source — same semantics as the tracking script (`0`, `/dev/video0`, `picam`) |
| `--output` / `-o` | `raw_<timestamp>.mp4` | Output file path |
| `--duration` | none | Auto-stop after N seconds (otherwise runs until `q` / `Ctrl+C`) |
| `--fps` | `30` | Frame rate written to the file |
| `--input-small` / `--input` / `--input-large` / `--input-fhd` | `--input-small` | Capture resolution (640×480 / 1024×768 / 1280×720 / 1920×1080) |
| `--display-small` / `--display` / `--display-large` | off | Show preview window while recording |

Same safe-shutdown behaviour as the tracking script: `q` in preview, window-X close, `Ctrl+C`, `kill`, terminal-close (SIGHUP) — all flush and finalize the MP4 cleanly.

### Offline evaluation

Once you have a raw recording and a manually counted ground-truth number, run the tracker on the file (no camera, no real-time pressure) and compare. Useful for tuning `--min-hits`, `--min-iou`, `--buffer`, and as a regression check after any algorithm change.

```bash
# Total expected = 12 (across all lines), exact match required
python evaluation/evaluate.py raw_morning.mp4 --expected 12

# Allow off-by-1 tolerance
python evaluation/evaluate.py raw_morning.mp4 --expected 12 --tolerance 1

# Per-line expected, inline form
python evaluation/evaluate.py raw_morning.mp4 --expected line_1=5,line_2=7

# Per-line expected, from JSON file (handy for committed test fixtures)
python evaluation/evaluate.py raw_morning.mp4 --expected expected.json

# Visual debugging — write annotated copy and watch live
python evaluation/evaluate.py raw_morning.mp4 --expected 12 --annotate annotated.mp4 --display
```

`expected.json` format:
```json
{ "line_1": 5, "line_2": 7 }
```

Output is a per-line table plus a final `RESULT: PASS` / `RESULT: FAIL`. **Exit code is 0 on PASS and 1 on FAIL**, so the script doubles as a CI check:

```bash
python evaluation/evaluate.py raw_morning.mp4 --expected expected.json --tolerance 1 \
    && echo "tracker still OK" \
    || echo "regression!"
```

Each frame in the file is read sequentially (no frame dropping like with a live camera), so the count is **deterministic** — re-running the same file with the same config and model gives the same result. This is what makes it suitable as a unit-test-like check.

| Flag | Default | Description |
|---|---|---|
| `recording` | (required) | Path to recorded video file |
| `--config` | `line_config.json` | Line config JSON |
| `--expected` | none | Ground-truth: integer (total), `name=N,...` (per-line), or path to a `.json` file |
| `--tolerance` | `0` | Allowed `|actual - expected|` per line |
| `--annotate PATH` | off | Write an annotated copy of the recording for visual debugging |
| `--display` | off | Show preview window while evaluating (slower) |
| `--model` | platform default | Same semantics as the tracking script |
| `--min-hits` / `--min-iou` / `--buffer` / etc. | same as tracker | Tuning knobs — pass them through to compare against the same algorithm settings used live |

Add a JSON file with expected counts next to each recording, commit both, and you have a reproducible regression suite.

### End-to-end tuning workflow

The three scripts compose into a tight tune-and-test loop. Instead of guessing whether a parameter change helps or hurts on a live camera (where you can't reproduce the exact same traffic twice), bake the traffic into a file and replay it against different tracker settings.

**Step 1 — Capture a representative clip.** Record long enough to cover the cases you care about (rush hour, occlusions, fast vehicles, near-misses). Higher resolution is better here because you only do it once:

```bash
python evaluation/record_raw.py --source picam --input-fhd --duration 120 --output raw_morning.mp4
```

**Step 2 — Manually count.** Open the file in any player, count vehicles per line by eye. This is your ground truth — write it down once and reuse forever:

```bash
cat > expected.json <<EOF
{ "line_1": 47, "line_2": 12 }
EOF
```

Commit both `raw_morning.mp4` (or just the JSON if the file is large — keep raws on a NAS) and `expected.json`. They define the test fixture.

**Step 3 — Baseline.** Run the tracker on the recording with current defaults to see where you stand:

```bash
$ python evaluation/evaluate.py raw_morning.mp4 --expected expected.json
  Line                     Actual    Expected    Diff  Status
  ---------------------- --------  ----------  ------  ------
  line_1                       52          47      +5    FAIL
  line_2                       14          12      +2    FAIL
  TOTAL                        66          59      +7
RESULT: FAIL — 2 line(s) outside tolerance ±0
```

**Step 4 — Tune.** The diff column tells you the direction of error:

- **Actual > Expected** = over-counting → ID churn or duplicate detections. Try `--min-hits 5` (require longer track confirmation), `--min-iou 0.10` (more forgiving matching), or check for `--no-deduplicate` accidentally on.
- **Actual < Expected** = under-counting → tracker losing fast objects. Try `--max-distance 350`, `--max-disappeared 80`, or a higher confidence model (`--model ~/hailo_models/yolov11l.hef`).
- **One line off, others fine** = line geometry issue. Re-run `--setup` and redraw, or add `--buffer 20` so vehicles within 20px of the line still count.

```bash
# Try tighter confirmation gate
$ python evaluation/evaluate.py raw_morning.mp4 --expected expected.json --min-hits 5
  line_1   →   48 (was 52, target 47)  +1   FAIL
  line_2   →   12 (was 14, target 12)  +0   OK
  TOTAL    →   60 (was 66, target 59)  +1
RESULT: FAIL — 1 line(s) outside tolerance ±0

# Add small tolerance for camera-jitter ambiguity
$ python evaluation/evaluate.py raw_morning.mp4 --expected expected.json --min-hits 5 --tolerance 1
RESULT: PASS — all lines within tolerance ±1
```

**Step 5 — Lock in and watch live.** Once the file passes, use the exact same flags live so what you saw offline is what you get on the camera:

```bash
python run_yolo11_tracking.py --display --record --min-hits 5
```

**Step 6 — Regression check after future changes.** Every time you touch the tracker code (or update YOLO weights, or change the model), re-run evaluate against the saved fixture. If the numbers still match, the change didn't break counting:

```bash
python evaluation/evaluate.py raw_morning.mp4 --expected expected.json --tolerance 1 \
    || echo "regression — investigate before merging"
```

Keep multiple fixtures for different scenarios (`raw_morning.mp4`, `raw_rain.mp4`, `raw_night.mp4`) and run them all in CI to catch tuning that helps one case but hurts another.

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
