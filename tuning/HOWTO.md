# How to tune tracker parameters — step by step

A complete walkthrough for tuning the vehicle tracker on a recorded clip.
Two paths: **manual** (you change the JSON, you re-run) and **agent-driven**
(the `tune-tracker` Claude Code agent does it for you). Manual path first
because it explains what's happening; the agent path is the same loop, just
automated.

---

## 0. What you'll need

| Thing | What it is | How to get it |
|---|---|---|
| **A recorded clip** | An `.mp4` of the camera view you want to count | `python evaluation/record_raw.py --duration 60` or `python evaluation/download_clip.py <url> --duration 120` |
| **Ground truth** | Number of vehicles you (a human) counted crossing each line | Watch the clip, count by hand. Save as JSON or pass inline. |
| **Line config** | Where the counting lines are drawn on the frame | `python run_yolo11_tracking.py --setup --source 0` (live camera) or `--source raw_morning.mp4 [--frame N]` (from the same clip you tune on) → creates `line_config.json` |
| **Tracker config** | The 8 tunable knobs | [`tracker_config.json`](../tracker_config.json) at repo root (already there with defaults) |

The ground-truth file format (`expected.json`) matches what `evaluate.py` expects:

```json
{"line_1": 12, "line_2": 7}
```

If you only care about the total across all lines, you can pass an integer:
`--expected 19`. Per-line is more diagnostic — you'll see *which* line is
off, which usually points at the problem.

---

## 1. Establish a baseline

Run the evaluator once with default params to see where you stand.

```bash
python evaluation/evaluate.py raw_morning.mp4 \
  --expected expected.json \
  --tolerance 0
```

The output ends with a table:

```
  Line                   Actual    Expected    Diff  Status
  ---------------------- --------  ----------  ------  ------
  line_1                       15          12      +3    FAIL
  line_2                        6           7      -1    FAIL
  TOTAL                        21          19      +2

RESULT: FAIL — 2 line(s) outside tolerance ±0
```

**This is your baseline.** Write it down. You're now going to change one
knob at a time and see if the table gets better.

---

## 2. Read the error pattern

The *direction* of error tells you which knobs to touch. Three cases:

### Case A: Overcounting on every line (`Diff` is `+N`, `+N`, ...)
```
line_1   15   12   +3   FAIL
line_2    9    7   +2   FAIL
```

**What's happening**: the tracker is creating *new* track IDs for the same
vehicle multiple times — usually because it loses a track briefly (occlusion,
flicker) and reassigns. The reassigned track then crosses the line again.

**Knobs to push**, in priority order:
- `min_hits`: bump UP (3 → 5). A track must be seen for N frames before its
  crossings count. Higher = ghost detections die before counting.
- `min_iou`: bump UP (0.15 → 0.20 → 0.25). Stricter matching between frames
  → less ID switching.
- `confidence`: bump UP (0.3 → 0.4). Removes weak detections that flicker.
- `deduplicate`: must be `true` (it is, by default). Suppresses overlapping
  same-class boxes.

### Case B: Undercounting on every line (`Diff` is `-N`, `-N`, ...)
```
line_1   10   12   -2   FAIL
line_2    5    7   -2   FAIL
```

**What's happening**: vehicles are crossing without being counted. Either
the detection misses them, or the tracker drops them mid-cross.

**Knobs to push**:
- `confidence`: bump DOWN (0.3 → 0.2). Detect weaker objects.
- `max_disappeared`: bump UP (50 → 80). Tracker keeps a lost track alive
  longer — survives occlusion.
- `max_distance`: bump UP (200 → 300). Allows fast-moving vehicles to be
  matched across larger jumps between frames.
- `min_hits`: bump DOWN (3 → 1). Counts fast vehicles that only appear in
  a few frames.
- `buffer`: bump UP (0 → 10 → 20). Counts vehicles entering a zone *near*
  the line, not just exact crossings — helps if line crossings are missed.

### Case C: Mixed (some lines over, some under)
```
line_1   15   12   +3   FAIL
line_2    5    7   -2   FAIL
```

**Stop tuning.** This is almost never a parameter problem. Likely causes:
- The counting line is positioned where bounding-box centroids jitter
  (e.g. line right where the model's box wobbles between frames). Re-draw
  with `python run_yolo11_tracking.py --setup`.
- The model doesn't see vehicles on `line_2` (occlusion, distance).
- Lines are too close together — same vehicle crosses both, but the
  detector sees it on one and not the other.

Visualize what the tracker actually sees:
```bash
python evaluation/evaluate.py raw_morning.mp4 \
  --expected expected.json \
  --annotate debug.mp4 \
  --display
```

Watch `debug.mp4` and find the misbehavior. Tuning won't fix this.

---

## 3. The tuning loop (manual path)

Pick **one** knob from your error pattern. Edit `tracker_config.json`:

```bash
# Before
$ cat tracker_config.json
{
  "confidence": 0.3,
  "min_hits": 3,
  ...
}
```

Change one value:

```json
{
  "confidence": 0.3,
  "min_hits": 5,
  ...
}
```

Re-run:

```bash
python evaluation/evaluate.py raw_morning.mp4 \
  --expected expected.json \
  --tolerance 0
```

Compare new `TOTAL Diff` to baseline:
- **Better** (smaller absolute diff): keep this value, try pushing the same
  knob further (`min_hits: 5 → 6`).
- **Same**: this knob doesn't matter for this clip. Revert and try a different one.
- **Worse**: revert. Try the next knob from the list.

Continue until the diff stops improving. Then move to the next knob.

**This is coordinate descent.** It's much cheaper than grid search and
usually within 1–2 of the global optimum.

### Stopping criteria
Stop when **any** of these is true:
- All lines within `tolerance: 0` (perfect — done).
- Last 3 changes didn't improve the diff (you've hit the floor for this
  param subset).
- You've spent ~25 runs (~30 min on a laptop). Diminishing returns.

If you're still off and you've tried every knob, re-read **Case C** above —
the problem may not be tunable.

---

## 4. The tuning loop (agent path)

If you don't want to do steps 2–3 by hand, use the `tune-tracker` agent.
In Claude Code (this repo), describe the situation:

```
Stuń tracker na raw_morning.mp4, expected.json — tracker liczy o 2 za dużo na line_1
```

The agent will:

1. Snapshot current `tracker_config.json` so it can restore later.
2. Check `tuning/results/` for prior runs on the same clip (re-uses prior best as starting point if found).
3. Run the baseline.
4. Read the error pattern (Case A/B/C from §2 above).
5. Pick the right knobs and sweep them, one at a time, ~12–16 runs.
6. Save the full history + best config to `tuning/results/<timestamp>_<clip>.json`.
7. Print the winning JSON values + diff vs. baseline.
8. Ask whether to write the best params back to `tracker_config.json`.
   - If yes: `tracker_config.json` is updated, tuning result file kept.
   - If no: original `tracker_config.json` restored, result file still kept (you can apply by hand later).

Time: roughly `clip_duration × 15 runs`. A 60s clip → ~15 min on a Mac, faster on Hailo NPU.

---

## 5. After tuning — verify no regression

You found `min_hits: 5` works great on `raw_morning.mp4`. But did it break
the night clip you tuned last week?

Add the morning clip to your test suite (if not there yet):

```bash
mkdir -p evaluation/tests
cp raw_morning.mp4 evaluation/tests/morning.mp4
cp expected.json evaluation/tests/morning.expected.json
```

Run the suite against your new defaults:

```bash
python evaluation/run_suite.py
```

You'll see per-clip status with comparison to the previous baseline:

```
Case               | AbsErr | Diff             | Status
morning            |      1 | line_1=+1        | NEW
night_trucks       |      4 | line_1=+2,line_2=+2 | REGRESSION (2→4)
```

If something regressed, you have two options:
- **Revert** `tracker_config.json` to the previous values — single-clip wins
  aren't worth global regressions.
- **Find a middle value** that's good enough on both. Re-run `tune-tracker`
  with both clips loaded as test cases (talk to the agent about this — it
  doesn't do multi-clip tuning automatically).

If everything's stable or improved → commit `tracker_config.json` with a
message like `tune tracker: min_hits 3→5 (fixed +2 overcounting on morning clip)`.

---

## 6. Reference — parameter cheat sheet

| Param | Default | Increase if... | Decrease if... |
|---|---|---|---|
| `confidence` | 0.3 | overcounting (suppress flicker) | undercounting (detect weak objects) |
| `iou` | 0.45 | NMS leaving duplicates | NMS killing real adjacent objects |
| `min_iou` | 0.15 | track IDs switching mid-frame | tracks fragmenting (shouldn't matter much) |
| `max_distance` | 200 | fast vehicles losing tracks | n/a (going lower rarely helps) |
| `max_disappeared` | 50 | tracks dropped during brief occlusion | stale tracks getting matched to new vehicles |
| `min_hits` | 3 | ghost detections inflating counts | fast vehicles missed (only 1–2 frames visible) |
| `buffer` | 0 | exact line crossings being missed | duplicate counts in the buffer zone |
| `deduplicate` | true | (leave on; only flip if you suspect it's hiding real distinct objects) | — |

For the *full* parameter reference (code locations, units, sensible ranges, interactions between params), see [PARAMETERS.md](PARAMETERS.md).

---

## 7. Common pitfalls

- **Tuning on too short a clip.** A 10-second clip with 2 vehicles isn't
  enough signal — you'll overfit to noise. Use ≥60s with ≥10 crossings if
  possible.
- **Wrong ground truth.** Re-watch the clip if numbers look weirdly off.
  Easy to miscount by 1–2 by hand.
- **Tuning to a single clip.** Always run [`run_suite.py`](../evaluation/run_suite.py)
  on a held-out test set after tuning. One clip's optimum often regresses others.
- **Touching the wrong knobs.** If error is mixed (Case C), no knob will fix
  it — go back to line geometry / model choice.
- **Forgetting CLI overrides win.** If you have a script somewhere that
  passes `--confidence 0.5`, tuning `tracker_config.json` to `0.4` won't
  affect that script. CLI flags override the JSON.
