"""
Streamlit-based visual editor for line_config.json (schema v1).

Click-based: pick a video frame, click two points to define a line, repeat.
Edit per-line metadata (name, description, classes, direction, enabled,
webhook_url) in side panels. Save as v1 JSON.

Usage:
    pip install streamlit streamlit-image-coordinates pillow
    streamlit run tools/visual_editor.py

Workflow:
    1. Sidebar: Browse… / pick a file from the dropdown / type a custom path.
    2. (For videos) pick a frame with the slider.
    3. Click two points on the image — that's a line. Repeat for more.
    4. Expand each line in the right panel and edit name/classes/etc.
    5. Click "Save" to write line_config.json (v1).

If the output file already exists and is v1 (or v0, with defaults filled
in for missing fields), its lines + metadata are loaded as the starting
state.
"""

import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hailo_common import COCO_CLASSES  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_CLASSES = ["car", "motorcycle", "bus", "truck"]
DIRECTIONS = ["both", "positive", "negative"]
MAX_DISPLAY_WIDTH = 900  # cap displayed image width (side-by-side layout)

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
SCAN_DIRS = (".", "evaluation/tests", "evaluation")

HIT_RADIUS_PX = 25  # endpoint pick-up radius (in display pixels)
DELETE_RADIUS_PX = 18  # line click radius (perpendicular distance) for delete mode

MODE_ADD = "add"
MODE_MOVE = "move"
MODE_DELETE = "delete"

# Palette (RGB) — matplotlib tab10, perceptually distinct, high contrast on most footage
LINE_COLORS = [
    (31, 119, 180),   # blue
    (255, 127, 14),   # orange
    (44, 160, 44),    # green
    (214, 39, 40),    # red
    (148, 103, 189),  # purple
    (140, 86, 75),    # brown
    (227, 119, 194),  # pink
    (188, 189, 34),   # olive
    (23, 190, 207),   # cyan
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Best-effort cross-platform font loader. Falls back to PIL default if nothing else works."""
    candidates = [
        "Arial.ttf", "Helvetica.ttc",                                # macOS
        "/System/Library/Fonts/Helvetica.ttc",                       # macOS absolute
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",      # Linux
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",                              # Windows
        "DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()

st.set_page_config(page_title="Line config editor", layout="wide", initial_sidebar_state="expanded")
st.title("Line config editor")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_sources():
    found = set()
    for d in SCAN_DIRS:
        if not os.path.isdir(d):
            continue
        for ext in VIDEO_EXTS + IMAGE_EXTS:
            for path in glob.glob(os.path.join(d, f"*{ext}")):
                found.add(os.path.normpath(path))
    return sorted(found)


@st.cache_data(show_spinner=False)
def load_frame(source: str, frame_no: int, mtime: float):
    """Load and cache a frame. `mtime` participates in the cache key so that
    editing the source file invalidates the cache automatically."""
    del mtime  # only used to bust the cache; not used inside
    if not source or not os.path.isfile(source):
        return None
    ext = os.path.splitext(source)[1].lower()
    if ext in IMAGE_EXTS:
        bgr = cv2.imread(source)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frame_no, total - 1)))
    ret, bgr = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False)
def get_video_info(source: str, mtime: float):
    """Cache total_frames lookup so the script doesn't reopen the file every rerun."""
    del mtime
    if not os.path.isfile(source):
        return 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def load_existing_config(path: str):
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def metadata_for(i: int, existing_lines: list, default_classes: list):
    if existing_lines and i < len(existing_lines):
        e = existing_lines[i]
        return {
            "name": e.get("name", f"line_{i + 1}"),
            "description": e.get("description", ""),
            "direction": e.get("direction", "both"),
            "classes": e.get("classes", list(default_classes)),
            "enabled": e.get("enabled", True),
            "webhook_url": e.get("webhook_url", ""),
        }
    return {
        "name": f"line_{i + 1}",
        "description": "",
        "direction": "both",
        "classes": list(default_classes),
        "enabled": True,
        "webhook_url": "",
    }


def native_file_picker():
    """Open a native OS file-open dialog. Returns (path, error)."""
    import platform
    import subprocess

    osa_snippet = '''
try
    set theFile to choose file with prompt "Pick a video or image"
    return POSIX path of theFile
on error
    return ""
end try
'''
    tk_snippet = '''
import sys
from tkinter import Tk, filedialog
root = Tk(); root.withdraw(); root.wm_attributes("-topmost", 1)
path = filedialog.askopenfilename(
    title="Pick a video or image",
    filetypes=[("Video / image", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp"), ("All", "*.*")],
)
root.destroy()
sys.stdout.write(path or "")
'''
    if platform.system() == "Darwin":
        try:
            r = subprocess.run(["osascript", "-e", osa_snippet],
                               capture_output=True, text=True, timeout=300)
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            return None, f"osascript failed: {exc}"
        if r.returncode != 0:
            return None, f"osascript returned {r.returncode}: {r.stderr.strip() or '(no stderr)'}"
        return r.stdout.strip(), None

    try:
        r = subprocess.run([sys.executable, "-c", tk_snippet],
                           capture_output=True, text=True, timeout=300)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return None, f"Tk subprocess failed: {exc}"
    if r.returncode != 0:
        return None, f"Tk returned {r.returncode}: {r.stderr.strip() or '(no stderr)'}"
    return r.stdout.strip(), None


def find_nearest_endpoint(click_xy_frame, lines, max_dist_frame):
    """Return (line_idx, endpoint_idx 0=p1 or 1=p2) of nearest endpoint within
    max_dist_frame, or None. Distances are in frame pixels."""
    cx, cy = click_xy_frame
    best = None
    best_d2 = max_dist_frame ** 2
    for i, ln in enumerate(lines):
        for j, (ex, ey) in enumerate([(ln["x1"], ln["y1"]), (ln["x2"], ln["y2"])]):
            d2 = (cx - ex) ** 2 + (cy - ey) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = (i, j)
    return best


def find_nearest_line(click_xy_frame, lines, max_dist_frame):
    """Return line_idx of nearest line within max_dist_frame (perpendicular
    distance to segment), or None."""
    cx, cy = click_xy_frame
    best = None
    best_d = max_dist_frame
    for i, ln in enumerate(lines):
        x1, y1, x2, y2 = ln["x1"], ln["y1"], ln["x2"], ln["y2"]
        dx, dy = x2 - x1, y2 - y1
        seg_len2 = dx * dx + dy * dy
        if seg_len2 == 0:
            continue
        # Project click onto segment, clamped to [0, 1]
        t = ((cx - x1) * dx + (cy - y1) * dy) / seg_len2
        t = max(0.0, min(1.0, t))
        px = x1 + t * dx
        py = y1 + t * dy
        d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        if d < best_d:
            best_d = d
            best = i
    return best


def render_overlay(frame_rgb: np.ndarray, lines: list, metadata: list,
                   pending_click: tuple | None, scale: float,
                   mode: str = MODE_ADD,
                   pending_move: tuple | None = None) -> Image.Image:
    """Draw lines + labels + pending-click marker onto the frame, return PIL image at display size."""
    h, w = frame_rgb.shape[:2]
    disp_w = int(w * scale)
    disp_h = int(h * scale)
    img = Image.fromarray(frame_rgb).resize((disp_w, disp_h), Image.LANCZOS) if scale < 1.0 else Image.fromarray(frame_rgb).copy()
    draw = ImageDraw.Draw(img, "RGBA")

    font = _load_font(18)

    for i, ln in enumerate(lines):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        x1, y1, x2, y2 = [v * scale for v in (ln["x1"], ln["y1"], ln["x2"], ln["y2"])]
        md = metadata[i] if i < len(metadata) else None
        is_enabled = bool(md and md.get("enabled", True))
        line_color = color if is_enabled else (140, 140, 140)

        # Line — disabled gets thinner, dashed-look via shorter alpha
        if is_enabled:
            # White halo for contrast on any background
            draw.line([(x1, y1), (x2, y2)], fill=(255, 255, 255), width=5)
            draw.line([(x1, y1), (x2, y2)], fill=line_color, width=3)
        else:
            draw.line([(x1, y1), (x2, y2)], fill=(*line_color, 160), width=2)

        # Endpoints — bigger "grip" handles in move/delete modes for clearer affordance
        endpoint_r = 10 if mode in (MODE_MOVE, MODE_DELETE) else 7
        inner_r = 7 if mode in (MODE_MOVE, MODE_DELETE) else 5
        for j, (px, py) in enumerate([(x1, y1), (x2, y2)]):
            is_selected = pending_move == (i, j)
            if is_selected:
                # Pulsing-style highlight: outer yellow ring + green core
                draw.ellipse([px - 14, py - 14, px + 14, py + 14], outline=(255, 230, 0), width=3)
                draw.ellipse([px - inner_r, py - inner_r, px + inner_r, py + inner_r], fill=(0, 200, 0))
            else:
                draw.ellipse([px - endpoint_r, py - endpoint_r, px + endpoint_r, py + endpoint_r], fill=(255, 255, 255))
                draw.ellipse([px - inner_r, py - inner_r, px + inner_r, py + inner_r], fill=line_color)

        # Direction arrow (perpendicular to p1→p2, pointing left side = "positive")
        if is_enabled:
            dx, dy = x2 - x1, y2 - y1
            length = max(1.0, (dx * dx + dy * dy) ** 0.5)
            nx, ny = -dy / length * 22, dx / length * 22
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax, ay = mx + nx, my + ny
            draw.line([(mx, my), (ax, ay)], fill=(255, 255, 255), width=4)
            draw.line([(mx, my), (ax, ay)], fill=line_color, width=2)
            # arrowhead
            head = 6
            angle = np.arctan2(ny, nx)
            for offset in (0.5, -0.5):
                hx = ax - head * np.cos(angle + offset)
                hy = ay - head * np.sin(angle + offset)
                draw.line([(ax, ay), (hx, hy)], fill=line_color, width=2)

        # Label slightly offset above midpoint, with rounded-rect background
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        name = md["name"] if md else f"line_{i+1}"
        label = f"{name}" + ("" if is_enabled else " (off)")
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = len(label) * 9, 18
        # position above the line, but flip below if too close to top
        ly = my - th - 14 if my - th - 14 > 4 else my + 14
        lx = mx - tw / 2
        pad = 4
        draw.rectangle([lx - pad, ly - pad, lx + tw + pad, ly + th + pad],
                       fill=(0, 0, 0, 200))
        draw.text((lx, ly), label, fill=line_color if is_enabled else (200, 200, 200), font=font)

    # Pending click — high-contrast bullseye (white halo + green inner)
    if pending_click is not None:
        px, py = pending_click[0] * scale, pending_click[1] * scale
        draw.ellipse([px - 12, py - 12, px + 12, py + 12], outline=(255, 255, 255), width=4)
        draw.ellipse([px - 8, py - 8, px + 8, py + 8], outline=(0, 200, 0), width=3)
        draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(0, 200, 0))
        # Small text caption with white halo
        msg = "click second point"
        offset_x, offset_y = px + 16, py - 10
        for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((offset_x + ox, offset_y + oy), msg, fill=(255, 255, 255), font=font)
        draw.text((offset_x, offset_y), msg, fill=(0, 150, 0), font=font)

    return img


# ---------------------------------------------------------------------------
# Sidebar — source picker
# ---------------------------------------------------------------------------

CUSTOM_SENTINEL = "Custom path…"

with st.sidebar:
    st.subheader("📁 Source")
    if "picked_path" not in st.session_state:
        st.session_state.picked_path = ""

    if st.button("Browse…", help="Native OS file dialog (only when running locally).",
                 use_container_width=True):
        chosen, err = native_file_picker()
        if err:
            st.error(f"File picker failed — use the dropdown / custom path below.\n\n{err}")
        elif chosen:
            st.session_state.picked_path = chosen

    discovered = discover_sources()
    options = discovered + [CUSTOM_SENTINEL]
    default_idx = 0 if discovered and not st.session_state.picked_path else len(options) - 1
    if st.session_state.picked_path and st.session_state.picked_path in discovered:
        default_idx = discovered.index(st.session_state.picked_path)
        st.session_state.picked_path = ""
    picked = st.selectbox(
        "Recent / discovered" if discovered else "Source",
        options, index=default_idx,
        format_func=lambda p: os.path.basename(p) if p != CUSTOM_SENTINEL else "📝 " + p,
    )
    if picked == CUSTOM_SENTINEL:
        source = st.text_input("Path", value=st.session_state.picked_path,
                               placeholder="/path/to/video.mp4 or relative")
    else:
        source = picked
        st.caption(f"`{picked}`")

    st.divider()
    st.subheader("💾 Output")
    output = st.text_input("Config path", value="line_config.json", label_visibility="collapsed")
    st.caption(f"`{output}`")

    st.divider()
    st.subheader("⚙️ Defaults for new lines")
    default_classes = st.multiselect(
        "Classes", COCO_CLASSES, default=DEFAULT_CLASSES,
        label_visibility="collapsed",
        help="Applied to lines you draw from now on. Existing lines keep their own classes.",
    )

    st.divider()
    st.subheader("🖥️ Layout")
    side_by_side = st.toggle(
        "Edit lines beside image", value=True,
        help="Off: line metadata stacks below the image (full-width canvas). "
             "On: image on the left, line metadata on the right (better for wide monitors).",
    )


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

if not source:
    st.warning("Pick a file in the sidebar.")
    st.stop()

try:
    src_mtime = os.path.getmtime(source)
except OSError:
    st.error(f"Could not stat `{source}`.")
    st.stop()

frame_no = 0
if os.path.splitext(source)[1].lower() in VIDEO_EXTS:
    total_frames = get_video_info(source, src_mtime)
    if total_frames > 1:
        frame_no = st.slider("Frame", 0, total_frames - 1, total_frames // 2)

frame = load_frame(source, frame_no, src_mtime)
if frame is None:
    st.error(f"Could not load frame from `{source}`.")
    st.stop()

frame_h, frame_w = frame.shape[:2]
# In side-by-side mode the image lives in a ~60% column → cap tighter.
max_w = MAX_DISPLAY_WIDTH if side_by_side else 1100
scale = min(1.0, max_w / frame_w)
disp_w = int(frame_w * scale)
disp_h = int(frame_h * scale)


# ---------------------------------------------------------------------------
# Session state — load existing config on first run, then track edits
# ---------------------------------------------------------------------------

if "lines" not in st.session_state:
    st.session_state.lines = []
    st.session_state.line_metadata = []
    st.session_state.pending_click = None
    st.session_state.pending_move = None  # (line_idx, endpoint_idx) or None
    st.session_state.mode = MODE_ADD
    existing = load_existing_config(output)
    if existing and existing.get("lines"):
        for i, ln in enumerate(existing["lines"]):
            st.session_state.lines.append({
                "x1": ln["p1"][0] * frame_w,
                "y1": ln["p1"][1] * frame_h,
                "x2": ln["p2"][0] * frame_w,
                "y2": ln["p2"][1] * frame_h,
            })
            st.session_state.line_metadata.append(metadata_for(i, existing["lines"], default_classes))
        st.toast(f"Loaded {len(st.session_state.lines)} line(s) from {output}")

# Existing webhook_url for save
existing = load_existing_config(output)
existing_alerts = (existing or {}).get("alerts") or {}
existing_webhook = existing_alerts.get("webhook_url") or (existing or {}).get("webhook_url", "") if existing else ""


# ---------------------------------------------------------------------------
# Status banner — tells user what to do next at a glance
# ---------------------------------------------------------------------------

n_lines = len(st.session_state.lines)
mode = st.session_state.mode

if mode == MODE_ADD:
    if st.session_state.pending_click is not None:
        st.warning("**Click the second point** to complete the line.", icon="🎯")
    elif n_lines == 0:
        st.info("**Click two points on the image** to create your first counting line.", icon="👉")
    else:
        st.success(f"**{n_lines} line(s) drawn.** Click two more points to add another.", icon="✅")
elif mode == MODE_MOVE:
    if st.session_state.pending_move is not None:
        i, j = st.session_state.pending_move
        which = "p1" if j == 0 else "p2"
        name = st.session_state.line_metadata[i]["name"]
        st.warning(f"**Click anywhere to move {which} of `{name}`** to that position.", icon="✋")
    elif n_lines == 0:
        st.info("Move mode: no lines yet. Switch to **Add** to draw one.", icon="ℹ️")
    else:
        st.info("**Click an endpoint** (circle) to pick it up, then click again to drop it in a new place.", icon="✋")
elif mode == MODE_DELETE:
    if n_lines == 0:
        st.info("Delete mode: no lines to delete.", icon="ℹ️")
    else:
        st.warning("**Click any line** to delete it. Click outside any line to cancel.", icon="🗑️")


# ---------------------------------------------------------------------------
# Two-column layout (image on left, lines on right) on wide screens.
# In stacked mode `image_area` and `lines_area` both point to the same
# container so calls to either render sequentially in the main column.
# ---------------------------------------------------------------------------

if side_by_side:
    # Wider right column so form fields don't feel cramped; large gap for
    # visual separation between canvas and edit panel.
    image_area, lines_area = st.columns([5, 4], gap="large")
else:
    _shared = st.container()
    image_area = lines_area = _shared


# ---------------------------------------------------------------------------
# Render image with overlays + capture clicks (left column)
# ---------------------------------------------------------------------------

overlay_img = render_overlay(
    frame, st.session_state.lines, st.session_state.line_metadata,
    st.session_state.pending_click, scale,
    mode=mode, pending_move=st.session_state.pending_move,
)

if "last_click" not in st.session_state:
    st.session_state.last_click = None


def _set_mode(new_mode: str):
    st.session_state.mode = new_mode
    # Clear in-progress pickups when switching modes
    st.session_state.pending_click = None
    st.session_state.pending_move = None


with image_area:
    # Mode selector + secondary controls
    mode_cols = st.columns([3, 1.3, 1.3])
    with mode_cols[0]:
        new_mode = st.radio(
            "Mode", [MODE_ADD, MODE_MOVE, MODE_DELETE],
            index=[MODE_ADD, MODE_MOVE, MODE_DELETE].index(mode),
            horizontal=True,
            format_func=lambda m: {"add": "➕ Add line", "move": "✋ Move endpoint", "delete": "🗑 Delete line"}[m],
            label_visibility="collapsed",
            key="mode_selector",
        )
        if new_mode != mode:
            _set_mode(new_mode)
            st.rerun()
    with mode_cols[1]:
        if st.button("↶ Undo last line", disabled=not st.session_state.lines, use_container_width=True):
            st.session_state.lines.pop()
            if st.session_state.line_metadata:
                st.session_state.line_metadata.pop()
            st.rerun()
    with mode_cols[2]:
        if st.button("🗑 Clear all", type="secondary",
                     disabled=not (st.session_state.lines or st.session_state.pending_click or st.session_state.pending_move),
                     use_container_width=True):
            st.session_state.lines = []
            st.session_state.line_metadata = []
            st.session_state.pending_click = None
            st.session_state.pending_move = None
            st.rerun()

    click = streamlit_image_coordinates(overlay_img, key="canvas_click")

    if click is not None:
        click_tuple = (click["x"], click["y"])
        if click_tuple != st.session_state.last_click:
            st.session_state.last_click = click_tuple
            fx = click["x"] / scale
            fy = click["y"] / scale

            if mode == MODE_ADD:
                if st.session_state.pending_click is None:
                    st.session_state.pending_click = (fx, fy)
                else:
                    x1, y1 = st.session_state.pending_click
                    st.session_state.lines.append({"x1": x1, "y1": y1, "x2": fx, "y2": fy})
                    st.session_state.line_metadata.append(
                        metadata_for(len(st.session_state.lines) - 1, [], default_classes)
                    )
                    st.session_state.pending_click = None
                st.rerun()

            elif mode == MODE_MOVE:
                hit_threshold = HIT_RADIUS_PX / scale  # convert display→frame
                if st.session_state.pending_move is None:
                    # Pick up nearest endpoint within threshold
                    hit = find_nearest_endpoint((fx, fy), st.session_state.lines, hit_threshold)
                    if hit is not None:
                        st.session_state.pending_move = hit
                        st.rerun()
                    else:
                        st.toast("No endpoint nearby — click closer to a circle.", icon="🎯")
                else:
                    # Drop the endpoint at new location
                    i, j = st.session_state.pending_move
                    if j == 0:
                        st.session_state.lines[i]["x1"] = fx
                        st.session_state.lines[i]["y1"] = fy
                    else:
                        st.session_state.lines[i]["x2"] = fx
                        st.session_state.lines[i]["y2"] = fy
                    st.session_state.pending_move = None
                    st.rerun()

            elif mode == MODE_DELETE:
                hit_threshold = DELETE_RADIUS_PX / scale
                hit = find_nearest_line((fx, fy), st.session_state.lines, hit_threshold)
                if hit is not None:
                    name = st.session_state.line_metadata[hit]["name"]
                    st.session_state.lines.pop(hit)
                    st.session_state.line_metadata.pop(hit)
                    st.toast(f"Deleted `{name}`", icon="🗑️")
                    st.rerun()
                else:
                    st.toast("No line under click. Click closer to a line.", icon="🎯")


# ---------------------------------------------------------------------------
# Save bar — placed right after the canvas so it's always visible without
# scrolling past line metadata. Holds the alerts/webhook side-by-side.
# ---------------------------------------------------------------------------

st.divider()

save_cols = st.columns([1.5, 3, 1])
with save_cols[0]:
    save = st.button(
        "💾 Save v1 config", type="primary", use_container_width=True,
        disabled=not st.session_state.lines,
    )
with save_cols[1]:
    webhook_url = st.text_input(
        "Global webhook URL (optional)", value=existing_webhook,
        help="Used for any line that doesn't set its own webhook_url.",
        label_visibility="collapsed",
        placeholder="Global webhook URL (optional) — leave empty for MOCK alerts",
    )
with save_cols[2]:
    if st.session_state.lines:
        st.metric("Lines", len(st.session_state.lines), label_visibility="collapsed")

save_status_slot = st.empty()


def _build_config():
    return {
        "schema_version": SCHEMA_VERSION,
        "calibrated_at": {"width": frame_w, "height": frame_h},
        "lines": [
            {
                "name": md["name"],
                "description": md["description"],
                "p1": [round(ln["x1"] / frame_w, 4), round(ln["y1"] / frame_h, 4)],
                "p2": [round(ln["x2"] / frame_w, 4), round(ln["y2"] / frame_h, 4)],
                "direction": md["direction"],
                "classes": md["classes"],
                "enabled": md["enabled"],
                **({"webhook_url": md["webhook_url"]} if md.get("webhook_url") else {}),
            }
            for ln, md in zip(st.session_state.lines, st.session_state.line_metadata)
        ],
        "alerts": {"webhook_url": webhook_url},
    }


if save:
    try:
        with open(output, "w") as f:
            json.dump(_build_config(), f, indent=2)
        save_status_slot.success(f"Saved {len(st.session_state.lines)} line(s) to `{output}`.", icon="✅")
    except OSError as exc:
        save_status_slot.error(f"Failed to write {output}: {exc}")


# ---------------------------------------------------------------------------
# Per-line forms — placed in `lines_area` (right col when side-by-side)
# ---------------------------------------------------------------------------

def _render_line_form(i: int):
    md = st.session_state.line_metadata[i]
    ln = st.session_state.lines[i]
    col_a, col_b = st.columns([2, 1])
    with col_a:
        md["name"] = st.text_input("Name", md["name"], key=f"name_{i}")
        md["description"] = st.text_input("Description (optional)", md["description"], key=f"desc_{i}")
        md["classes"] = st.multiselect("Classes", COCO_CLASSES, default=md["classes"], key=f"cls_{i}")
        md["webhook_url"] = st.text_input(
            "Webhook URL (optional)", md.get("webhook_url", ""),
            key=f"hook_{i}",
            help="Per-line override. Empty → use global webhook above.",
            placeholder="https://example.com/webhook",
        )
    with col_b:
        md["direction"] = st.radio(
            "Direction", DIRECTIONS,
            index=DIRECTIONS.index(md["direction"]),
            key=f"dir_{i}",
            help=(
                "**positive** = crossed onto the LEFT side of p1→p2 (arrow direction).\n\n"
                "**negative** = opposite. **both** = count both ways. "
                "Swap p1↔p2 to flip."
            ),
        )
        md["enabled"] = st.checkbox("Enabled", md["enabled"], key=f"en_{i}")
        p1 = (ln["x1"] / frame_w, ln["y1"] / frame_h)
        p2 = (ln["x2"] / frame_w, ln["y2"] / frame_h)
        st.caption(f"p1: ({p1[0]:.3f}, {p1[1]:.3f})  \np2: ({p2[0]:.3f}, {p2[1]:.3f})")
        if st.button("🗑 Delete this line", key=f"del_{i}", type="secondary", use_container_width=True):
            st.session_state.lines.pop(i)
            st.session_state.line_metadata.pop(i)
            st.rerun()


with lines_area:
    if not side_by_side:
        st.divider()

    n_lines = len(st.session_state.lines)
    panel = st.container(border=side_by_side)
    with panel:
        header_cols = st.columns([4, 1])
        with header_cols[0]:
            st.markdown(f"### ✏️ Edit lines ({n_lines})")
        with header_cols[1]:
            if existing and existing.get("lines"):
                st.caption(f"from `{os.path.basename(output)}`")

        if n_lines == 0:
            st.markdown(
                "👈 _Click two points on the image to draw your first line._  \n"
                "Each line gets its own panel here — name, classes, direction, webhook, etc."
            )
        elif n_lines == 1:
            md = st.session_state.line_metadata[0]
            label = md["name"] + (" (disabled)" if not md.get("enabled", True) else "")
            st.markdown(f"**{label}**")
            _render_line_form(0)
        else:
            labels = [
                md["name"] + ("·off" if not md.get("enabled", True) else "")
                for md in st.session_state.line_metadata
            ]
            tabs = st.tabs(labels)
            for i, tab in enumerate(tabs):
                with tab:
                    _render_line_form(i)


# ---------------------------------------------------------------------------
# Diagnostics + JSON preview (collapsed)
# ---------------------------------------------------------------------------

st.divider()

with st.expander("Preview JSON", expanded=False):
    if st.session_state.lines:
        st.code(json.dumps(_build_config(), indent=2), language="json")
    else:
        st.caption("(empty)")

with st.expander("Diagnostics", expanded=False):
    st.caption(
        f"Frame: {frame_w}×{frame_h}  |  display: {disp_w}×{disp_h}  |  scale: {scale:.2f}  \n"
        f"Source: `{source}`  |  output: `{output}`  \n"
        f"Schema: v{SCHEMA_VERSION}  |  pending_click: {st.session_state.pending_click}"
    )
