# Scripts overview

What each script does and whether it's still worth keeping. Last reviewed: 2026-05-02.

Legend: 🟢 actively used · 🟡 niche / situational · 🔴 candidate for removal

## Main entry points (repo root)

### 🟢 [`run_yolo11_tracking.py`](run_yolo11_tracking.py)
Vehicle tracking + line-crossing counter. **Primary script.** Supports multi-line setup (`--setup`), recording (`--record`), and loads tunable params from [`tracker_config.json`](tracker_config.json).
Runs on both Hailo NPU (`.hef`) and laptop (`.pt` / `.onnx`).

### 🟢 [`run_yolo11.py`](run_yolo11.py)
Plain object detection — bounding boxes, no tracking. Useful as a sanity check (does the camera + model + Hailo pipeline work at all?) before debugging the tracker.

### 🔴 [`run_gestures.py`](run_gestures.py)
Hand gesture recognition (HaGRID dataset). Marked as **WIP / not validated end-to-end** in the README. The install script `install_yolo11_gestures.sh` doesn't fully work either. Either invest time to finish it or delete the whole gesture path.

### 🟢 [`hailo_common.py`](hailo_common.py)
Shared library — camera helpers, `HailoSession` / `UltralyticsSession`, threaded capture, label loading, tracker config loader. Imported by everything. Not standalone.

## Security cameras

### 🟢 [`security_cameras/person_line_alert_v2.py`](security_cameras/person_line_alert_v2.py)
Person line-crossing → REST webhook alerts. v2 is the one to use — multi-line, arbitrary angles, interactive `--setup`.

### 🔴 [`security_cameras/person_line_alert.py`](security_cameras/person_line_alert.py)
v1 — single horizontal line only. Superseded by v2. Worth removing once you confirm no external scripts/docs reference v1.

## Evaluation

### 🟢 [`evaluation/evaluate.py`](evaluation/evaluate.py)
Replay a recorded clip through the tracker and compare to ground truth. Exit 0/1 for PASS/FAIL. `--json` for machine-readable output. Loads defaults from [`tracker_config.json`](tracker_config.json).

### 🟢 [`evaluation/record_raw.py`](evaluation/record_raw.py)
Record clean clips from your camera (no overlay, no detection). Source for ground-truth dataset.

### 🟢 [`evaluation/download_clip.py`](evaluation/download_clip.py)
Pull clips from YouTube / Twitch / Vimeo via yt-dlp. Alternative to `record_raw.py` when you don't have a camera handy.

### 🟢 [`evaluation/run_suite.py`](evaluation/run_suite.py)
Regression-test runner. Globs `evaluation/tests/*.mp4` + `*.expected.json`, runs `evaluate.py` on each, compares total absolute error to the previous baseline (`evaluation/results/latest.json`). Use before commits.

## Install / setup

### 🟢 [`install_hailo.sh`](install_hailo.sh)
Auto-detects Hailo-10H vs Hailo-8, installs the right driver package (`hailo1x_pci` + `hailo-h10-all` for 10H), enables PCIe Gen 3, blacklists conflicts. Idempotent. **Run once on each new Pi.**

### 🟢 [`install_yolo11.sh`](install_yolo11.sh)
Downloads pre-compiled YOLOv11 HEF from Hailo Model Zoo (variants n/m/l/x), sets up venv. Idempotent.

### 🔴 [`install_yolo11_gestures.sh`](install_yolo11_gestures.sh)
Tries to train + compile a HaGRID HEF for Hailo-10H. **Not fully tested** per README. Tied to `run_gestures.py` — fate of both is the same: finish or delete.

### 🟢 [`troubleshoot.sh`](troubleshoot.sh)
Diagnostic script: PCIe detection, kernel driver, firmware, Python bindings, camera. Color-coded PASS/WARN/FAIL. Run when something breaks.

## Recommendations

**Quick wins (low risk):**
1. Delete `security_cameras/person_line_alert.py` (v1) — fully superseded by v2, only ~150 LOC saved but removes user confusion about which to pick.
2. Decide on the gesture path: either land it (validate `install_yolo11_gestures.sh` end-to-end and add a quick eval) or remove `run_gestures.py` + `install_yolo11_gestures.sh` + `gesture_actions.yaml`. Currently it's documented as "WIP not functional" which is the worst state to leave it in — readers don't know if it works.

**Keep as-is:**
- Everything in `evaluation/` and the install scripts.

**Out of scope here but worth noting:**
- The `tune-tracker` Claude Code agent at [`.claude/agents/tune-tracker.md`](.claude/agents/tune-tracker.md) is documented in [`tuning/README.md`](tuning/README.md) — not a script, but part of the workflow.
