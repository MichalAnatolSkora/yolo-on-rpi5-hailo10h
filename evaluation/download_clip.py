#!/usr/bin/env python3
"""
Download a clip from YouTube / Twitch / Vimeo / any yt-dlp supported source.

Wraps yt-dlp with sensible defaults for ground-truth dataset building:
- output named `raw_<timestamp>.mp4` (matches record_raw.py convention)
- prefers MP4 at 720p (best compromise between quality and inference cost)
- handles livestreams (record N seconds from now or from start)
- handles VODs (extract a slice with --start and --duration)
- batch mode via a URL list file

Requires yt-dlp:
    brew install yt-dlp ffmpeg               # macOS
    sudo apt install yt-dlp ffmpeg           # Debian/Ubuntu/RPi
    pip install yt-dlp                       # any platform

Usage:
    # Download first 2 minutes of a YouTube video / livestream
    python evaluation/download_clip.py "https://www.youtube.com/watch?v=..." --duration 120

    # 5-minute snippet starting 10 minutes into a VOD
    python evaluation/download_clip.py "URL" --start 600 --duration 300

    # Live stream — record 60s from "now" (skip backlog)
    python evaluation/download_clip.py "URL" --live --duration 60

    # Custom output filename
    python evaluation/download_clip.py "URL" --duration 120 --output raw_intersection_morning.mp4

    # Batch — one URL per line in urls.txt; download 2 min from each
    python evaluation/download_clip.py --batch urls.txt --duration 120

    # Inspect available formats (qualities) without downloading
    python evaluation/download_clip.py "URL" --list-formats

Note: respect each site's Terms of Service. Most allow personal/research use of
publicly broadcast streams; commercial redistribution often does not.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
log = logging.getLogger(__name__)


def check_ytdlp() -> str:
    """Verify yt-dlp is available, return its version."""
    if shutil.which("yt-dlp") is None:
        log.error("yt-dlp not found in PATH. Install one of:")
        log.error("  macOS:        brew install yt-dlp ffmpeg")
        log.error("  Debian/Ubuntu/RPi: sudo apt install yt-dlp ffmpeg")
        log.error("  pip:          pip install yt-dlp  (also need ffmpeg system-wide)")
        sys.exit(2)
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"], capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        log.warning("yt-dlp --version timed out; continuing anyway.")
        return "(unknown)"


def auto_output_name(url: str) -> str:
    """Generate a default output filename based on the URL or current time."""
    # Try to extract a YouTube video id (?v=XXXX or /XXXX)
    import re
    m = re.search(r"(?:v=|/)([A-Za-z0-9_-]{11})(?:[?&]|$)", url)
    if m:
        return f"raw_yt_{m.group(1)}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    return f"raw_{time.strftime('%Y%m%d_%H%M%S')}.mp4"


def build_cmd(
    url: str,
    output: str,
    duration: float | None,
    start: float | None,
    height: int,
    live: bool,
    live_from_start: bool,
    list_formats: bool,
) -> list[str]:
    cmd = ["yt-dlp"]

    if list_formats:
        cmd += ["--list-formats", url]
        return cmd

    # Format selection: prefer MP4 at <= height, fall back gracefully.
    # Using "bv*+ba/b" lets yt-dlp merge best video + best audio when needed,
    # then we force the container to MP4 via --merge-output-format.
    fmt = (
        f"bv*[height<={height}][ext=mp4]+ba[ext=m4a]"
        f"/bv*[height<={height}]+ba"
        f"/b[height<={height}][ext=mp4]"
        f"/b[height<={height}]"
        f"/b"
    )
    cmd += ["-f", fmt, "--merge-output-format", "mp4"]

    if live:
        # Live streams need a different mechanism: --download-sections won't work
        # because the live manifest doesn't support range requests. We use ffmpeg
        # as the downloader and stop it with -t after `duration` seconds.
        if live_from_start:
            cmd += ["--live-from-start"]
        if duration is not None:
            cmd += [
                "--downloader", "ffmpeg",
                "--downloader-args", f"ffmpeg:-t {int(duration)}",
            ]
    else:
        # VOD: --download-sections works for partial downloads
        if duration is not None or start is not None:
            s = int(start or 0)
            if duration is not None:
                e = s + int(duration)
                section = f"*{s}-{e}"
            else:
                section = f"*{s}-"
            cmd += ["--download-sections", section]
            # force_keyframes guarantees the section starts on a keyframe so the
            # MP4 actually plays from the requested point (without it, the first
            # second can be black/glitchy).
            cmd += ["--force-keyframes-at-cuts"]

    # Output template
    cmd += ["-o", output]

    # Quieter output, but keep progress
    cmd += ["--no-warnings", "--newline"]

    cmd += [url]
    return cmd


def download_one(args: argparse.Namespace, url: str, output: str | None = None) -> bool:
    if not output:
        output = args.output or auto_output_name(url)

    out_dir = os.path.dirname(os.path.abspath(output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cmd = build_cmd(
        url=url,
        output=output,
        duration=args.duration,
        start=args.start,
        height=args.height,
        live=args.live,
        live_from_start=args.live_from_start,
        list_formats=args.list_formats,
    )

    log.info("URL: %s", url)
    log.info("Output: %s", output)
    log.debug("Command: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd)
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        return False

    if result.returncode != 0:
        log.error("yt-dlp failed (exit code %d) for %s", result.returncode, url)
        return False

    if args.list_formats:
        return True

    if not os.path.exists(output):
        # yt-dlp may add an extension if the template didn't have one
        log.warning("Expected file not found at %s — yt-dlp may have used a different name.", output)
        return False

    size_mb = os.path.getsize(output) / (1024 * 1024)
    log.info("Saved %s (%.1f MB)", output, size_mb)
    return True


def load_url_list(path: str) -> list[str]:
    urls = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download video clips from YouTube / Twitch / etc. for ground-truth datasets."
    )
    parser.add_argument(
        "url", nargs="?", default=None,
        help="Source URL (YouTube, Twitch, Vimeo, etc.). Required unless --batch is used.",
    )
    parser.add_argument(
        "--batch", default=None, metavar="FILE",
        help="Path to a text file with one URL per line (lines starting with # are ignored)",
    )
    parser.add_argument(
        "--output", "-o", default=None, metavar="PATH",
        help="Output file (default: raw_yt_<id>_<timestamp>.mp4 or raw_<timestamp>.mp4). "
             "Ignored in --batch mode (uses auto-naming).",
    )
    parser.add_argument(
        "--duration", type=float, default=None, metavar="SECONDS",
        help="Length of clip in seconds (default: full video / indefinite for live)",
    )
    parser.add_argument(
        "--start", type=float, default=None, metavar="SECONDS",
        help="Start offset in seconds (VOD only — for live streams use --live)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Treat URL as a live stream — record from now (or use --live-from-start for backlog)",
    )
    parser.add_argument(
        "--live-from-start", action="store_true",
        help="For live streams: download from the very beginning of the broadcast",
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Max video height in pixels (default: 720). Higher = bigger files & slower inference.",
    )
    parser.add_argument(
        "--list-formats", action="store_true",
        help="List available formats for the URL and exit (does not download)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.url and not args.batch:
        parser.error("either URL or --batch FILE is required")
    if args.url and args.batch:
        parser.error("pass either a URL or --batch, not both")
    if args.live_from_start:
        args.live = True  # implied

    version = check_ytdlp()
    log.info("yt-dlp version: %s", version)

    # Single-URL mode
    if args.url:
        ok = download_one(args, args.url)
        sys.exit(0 if ok else 1)

    # Batch mode
    urls = load_url_list(args.batch)
    if not urls:
        log.error("No URLs in %s", args.batch)
        sys.exit(2)

    log.info("Batch: %d URL(s) from %s", len(urls), args.batch)
    failed: list[str] = []
    for i, url in enumerate(urls, 1):
        log.info("--- [%d/%d] ---", i, len(urls))
        out = auto_output_name(url)
        ok = download_one(args, url, output=out)
        if not ok:
            failed.append(url)

    print()
    print(f"Batch complete: {len(urls) - len(failed)} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed URLs:")
        for u in failed:
            print(f"  {u}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
