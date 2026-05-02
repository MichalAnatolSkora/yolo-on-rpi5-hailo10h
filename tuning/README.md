# Tracker tuning

Automatic search for the best tracker parameters on a given clip + ground-truth
count. Driven by the **`tune-tracker`** Claude Code subagent, defined in
[`.claude/agents/tune-tracker.md`](../.claude/agents/tune-tracker.md).

> **Want the full step-by-step?** → [HOWTO.md](HOWTO.md) — manual + agent paths, common pitfalls, regression check.
> **What does each parameter actually do?** → [PARAMETERS.md](PARAMETERS.md) — full reference: code location, units, sensible ranges, interactions.

## What gets tuned

All tunable knobs live in [`tracker_config.json`](../tracker_config.json) at
the repo root:

```json
{
  "confidence": 0.3,
  "iou": 0.45,
  "min_iou": 0.15,
  "max_distance": 200.0,
  "max_disappeared": 50,
  "min_hits": 3,
  "buffer": 0,
  "deduplicate": true
}
```

Both `run_yolo11_tracking.py` and `evaluation/evaluate.py` load these as their
argparse defaults. **CLI flags still override** the config — so e.g.
`--confidence 0.5` wins over whatever's in the JSON. The agent tunes by
editing the JSON, then running scripts with no extra flags.

## Why an agent and not a script?

Tuning needs *judgment* between iterations:

- if the tracker is overcounting, push `--min-hits` up and `--confidence` up
- if it's undercounting, push `--max-distance` up and `--confidence` down
- if some lines over- and others undercount, it's a line-geometry problem, not a tuning problem — stop

A naive grid search over all 8 knobs would take hundreds of runs. The agent does
**coordinate descent** with these heuristics and usually finds a good config in
~15 runs. See [tune-tracker.md](../.claude/agents/tune-tracker.md) for the full strategy.

For a deterministic non-tuning *regression check* (fixed config, many clips), use
[`evaluation/run_suite.py`](../evaluation/run_suite.py) instead — that one is a
plain script.

## Prerequisites

1. **A clip** — record one with `evaluation/record_raw.py` or download with
   `evaluation/download_clip.py`. See [docs/evaluation.md](../docs/evaluation.md).
2. **Ground truth** — manually count vehicles per counting line. Either pass an
   integer total or write `expected.json`:
   ```json
   {"line_1": 12, "line_2": 7}
   ```
3. **Line config** — `line_config.json` in repo root (created via
   `python run_yolo11_tracking.py --setup --source 0`).

## Running the agent

In Claude Code, just describe what you want. The agent fires automatically based
on its `description` field — you don't need to type the agent name.

Examples:

```
Stuń tracker na raw_yt_M3EYAY2MftI_20260502_111709.mp4, expected total = 47
```

```
Mam klip night_trucks.mp4 i expected.json — znajdź najlepsze parametry
```

```
Tracker liczy podwójnie ciężarówki na morning_rush.mp4 (expected line_1=12) — 
znajdź params co to naprawiają
```

The agent will:
1. Check `tuning/results/` for prior runs on the same clip and start from there if found.
2. Run a baseline `evaluate.py` to see direction of error.
3. Pick the relevant knobs (over- vs undercounting strategy).
4. Sweep them one at a time (coordinate descent), ~12–16 runs total.
5. Save the best config + full history to `tuning/results/<timestamp>_<clipname>.json`.
6. Print the winning command line.

Each `evaluate.py` run takes roughly the clip duration (or faster on Hailo NPU,
slower on macOS CPU). A 60s clip × 15 runs ≈ 15 min on a laptop.

## Manually invoking the agent

If you want to force-route to it (e.g. the description didn't match):

```
@tune-tracker raw_morning.mp4, expected line_1=12,line_2=7
```

## Results layout

```
tuning/
  README.md                                  ← you are here
  results/
    20260502T143000Z_raw_morning.json        ← one file per tuning session
    20260503T091200Z_night_trucks.json
    ...
```

Each result file has the best params, the abs error achieved, and the full
iteration history (so you can see what was tried and rule out re-trying it).

## Applying the winning config

At the end of a tuning session the agent asks whether to write the best
params back to `tracker_config.json`. If you confirm, those values become
the new defaults for *all* future runs of `run_yolo11_tracking.py` and
`evaluate.py` — no CLI flag needed.

Then run `python evaluation/run_suite.py` to confirm the new defaults don't
regress your other test clips.

If you decline, the agent restores the original `tracker_config.json` and
the best params are still saved in `tuning/results/<timestamp>_<clip>.json`
for reference — you can apply them later by editing the file by hand.

## When tuning won't help

If after a few iterations the abs error stays the same and you see *mixed*
over- and undercounting across lines, the problem is upstream of the tracker:

- **Line geometry** — counting line is in a place where the model's bounding
  box centroid jitters across it. Re-draw with `--setup`.
- **Detection quality** — model is missing some objects entirely (look at the
  recording with `--annotate out.mp4 --display`). Try a bigger model
  (`yolo11m.pt` or `.hef`) — that's a model swap, not a tuning step.
- **Tracker design limit** — if the same object is being counted twice with
  different track IDs, the IoU tracker is losing it across the line. Tuning
  `--max-disappeared` and `--max-distance` is the right fix; if it doesn't
  help, you may need a stronger tracker (Kalman/SORT) which is out of scope here.

The agent will flag these cases automatically.
