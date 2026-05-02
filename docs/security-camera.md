# Security Camera: Person Line-Crossing Alert

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

## v2: Multi-Line Crossing with Interactive Setup

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
