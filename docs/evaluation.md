# Evaluation & Testing

When tuning the tracker, the most reliable test is: record a clip, manually count the vehicles, then run the tracker on the recording offline and compare. The `evaluation/` folder has three scripts that compose into this workflow:

- [`download_clip.py`](#downloading-test-footage) — grab a clip from YouTube / Twitch / etc. (no camera trip needed)
- [`record_raw.py`](#recording-ground-truth-clips) — capture clean (no annotation) clips from your own camera
- [`evaluate.py`](#offline-evaluation) — replay a clip through the tracker, compare to ground truth, exit 0/1 (PASS/FAIL)

The full loop is documented in [End-to-end tuning workflow](#end-to-end-tuning-workflow).

## Downloading test footage

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

## Recording ground-truth clips

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

## Offline evaluation

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

## End-to-end tuning workflow

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
