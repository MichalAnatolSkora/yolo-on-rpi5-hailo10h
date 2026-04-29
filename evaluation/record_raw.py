#!/usr/bin/env python3
"""
Clean camera recording — no detection, no overlay, no annotations.

Captures raw frames from the camera and writes them straight to MP4.
Used to build ground-truth datasets for evaluating the tracker offline:

    1. Record a clean clip:        python evaluation/record_raw.py --display --duration 60
    2. Manually count vehicles in the clip
    3. (later) Run the tracker on the recording and compare counts

Output filename has 'raw' in it so it's obvious there's no detection in the file.

Usage:
    # Quick recording, stop with Ctrl+C
    python evaluation/record_raw.py

    # Fixed duration with live preview
    python evaluation/record_raw.py --display --duration 30

    # High-res native HD from RPi camera
    python evaluation/record_raw.py --source picam --input-fhd --output street.mp4

    # Custom output path (note: 'raw_' is added automatically if you skip it)
    python evaluation/record_raw.py --output raw_morning_traffic.mp4
"""

import argparse
import atexit
import logging
import os
import signal
import sys
import time

import cv2

# Allow running as a script from the evaluation/ subfolder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hailo_common import ThreadedCamera, default_source, open_camera

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
log = logging.getLogger(__name__)

_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True


def _open_writer(path: str, w: int, h: int, fps: float):
    """Open a VideoWriter, trying multiple codecs.

    Returns (writer or None, final_path). The final_path may differ from the
    input if we had to fall back to .avi for the MJPG codec.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".avi":
        candidates = [("MJPG", path), ("XVID", path)]
    else:
        candidates = [
            ("avc1", path),
            ("mp4v", path),
            ("MJPG", os.path.splitext(path)[0] + ".avi"),
        ]
    for codec, p in candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(p, fourcc, fps, (w, h))
        if writer.isOpened():
            log.info("Recording to %s (codec=%s, %dx%d @ %.1f fps)", p, codec, w, h, fps)
            return writer, p
        writer.release()
    return None, path


def run(args: argparse.Namespace) -> None:
    output = args.output or time.strftime("raw_%Y%m%d_%H%M%S.mp4")
    out_dir = os.path.dirname(os.path.abspath(output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cap_w, cap_h = args.input_size
    log.info("Opening camera (source=%s, capture=%dx%d)...", args.source, cap_w, cap_h)
    raw_cap = open_camera(args.source, cap_w, cap_h)
    if not raw_cap.isOpened():
        log.error("Could not open camera. Check connection / permissions.")
        sys.exit(1)
    cap = ThreadedCamera(raw_cap)

    writer: cv2.VideoWriter | None = None
    final_path = output
    frames_written = 0
    start_time: float | None = None

    # Safety net: release writer on any exit path (SIGHUP, uncaught exception, sys.exit).
    _writer_ref = {"w": None, "path": None}

    def _atexit_release():
        if _writer_ref["w"] is not None:
            _writer_ref["w"].release()
            log.info("Saved recording (atexit): %s", _writer_ref["path"])
            _writer_ref["w"] = None

    atexit.register(_atexit_release)

    log.info("Recording. Press 'q' (in preview window) or Ctrl+C to stop.")
    if args.duration:
        log.info("Auto-stop after %.1fs", args.duration)

    try:
        while not _shutdown:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Lazily create the writer on the first valid frame so we know real H,W
            if writer is None:
                rec_h, rec_w = frame.shape[:2]
                writer, final_path = _open_writer(final_path, rec_w, rec_h, args.fps)
                if writer is None:
                    log.error("Could not open any video writer for %s", output)
                    return
                _writer_ref["w"] = writer
                _writer_ref["path"] = final_path
                start_time = time.time()

            writer.write(frame)
            frames_written += 1

            if frames_written % 60 == 0:
                elapsed = time.time() - start_time
                log.info(
                    "Recorded %d frames (%.1fs elapsed, real fps=%.1f)",
                    frames_written, elapsed, frames_written / max(elapsed, 1e-3),
                )

            if args.duration and (time.time() - start_time) >= args.duration:
                log.info("Duration reached (%.1fs), stopping.", args.duration)
                break

            if args.display_size:
                dw, dh = args.display_size
                if (frame.shape[1], frame.shape[0]) != (dw, dh):
                    show = cv2.resize(frame, (dw, dh))
                else:
                    show = frame
                cv2.imshow("Raw Recording", show)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                # Detect window close (X button)
                try:
                    if cv2.getWindowProperty("Raw Recording", cv2.WND_PROP_VISIBLE) < 1:
                        log.info("Preview window closed — stopping.")
                        break
                except cv2.error:
                    pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            _writer_ref["w"] = None  # prevent double-release from atexit
        if args.display_size:
            cv2.destroyAllWindows()
        elapsed = (time.time() - start_time) if start_time else 0.0
        if frames_written > 0:
            log.info(
                "Saved %s — %d frames, %.1fs (real fps=%.1f, file fps=%.1f)",
                final_path, frames_written, elapsed,
                frames_written / max(elapsed, 1e-3), args.fps,
            )
        else:
            log.warning("No frames captured. File not written.")


def main() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _signal_handler)

    parser = argparse.ArgumentParser(
        description="Record raw camera feed to MP4 (no detection, no overlay)."
    )
    parser.add_argument(
        "--source", default=default_source(),
        help="Camera source: device index (0, 1), /dev/videoN, or 'picam'",
    )
    parser.add_argument(
        "--output", "-o", default=None, metavar="PATH",
        help="Output file (default: raw_YYYYMMDD_HHMMSS.mp4 in cwd)",
    )
    parser.add_argument(
        "--duration", type=float, default=None, metavar="SECONDS",
        help="Auto-stop after N seconds (default: run until 'q' or Ctrl+C)",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Frame rate written into the file (default: 30)",
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--input-small", action="store_const", dest="input_size", const=(640, 480))
    input_group.add_argument("--input", action="store_const", dest="input_size", const=(1024, 768))
    input_group.add_argument("--input-large", action="store_const", dest="input_size", const=(1280, 720))
    input_group.add_argument("--input-fhd", action="store_const", dest="input_size", const=(1920, 1080),
                             help="Native Full HD (1920x1080) — best for ground-truth recordings")

    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument("--display-small", action="store_const", dest="display_size", const=(640, 480))
    display_group.add_argument("--display", action="store_const", dest="display_size", const=(1024, 768))
    display_group.add_argument("--display-large", action="store_const", dest="display_size", const=(1280, 720))

    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.input_size is None:
        args.input_size = (640, 480)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run(args)


if __name__ == "__main__":
    main()
