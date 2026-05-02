#!/usr/bin/env python3
"""
Regression-test runner for the tracker.

Runs evaluate.py on every clip under evaluation/tests/ that has a paired
.expected.json, aggregates results, and compares total absolute error
against the previous run (evaluation/results/latest.json).

Layout:
    evaluation/tests/
        morning_rush.mp4
        morning_rush.expected.json
        ...
    evaluation/results/
        latest.json          (overwritten each run, used as baseline)
        <timestamp>.json     (history)

Exit code: 0 if no regressions, 1 if any case got worse vs baseline.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "evaluation" / "tests"
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"
EVALUATE_PY = REPO_ROOT / "evaluation" / "evaluate.py"


def discover_cases():
    """Return list of (name, mp4_path, expected_json_path) for each valid pair."""
    if not TESTS_DIR.is_dir():
        return []
    cases = []
    for mp4 in sorted(TESTS_DIR.glob("*.mp4")):
        expected = mp4.with_suffix(".expected.json")
        if expected.is_file():
            cases.append((mp4.stem, mp4, expected))
        else:
            print(f"  ! skipping {mp4.name} — no {expected.name}")
    return cases


def run_case(name, mp4, expected, config, model, extra_args):
    """Run evaluate.py once, return parsed results dict (or None on failure)."""
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tmp:
        json_out = tmp.name
    try:
        cmd = [
            sys.executable, str(EVALUATE_PY), str(mp4),
            "--expected", str(expected),
            "--config", str(config),
            "--model", str(model),
            "--tolerance", "0",
            "--json", json_out,
        ] + extra_args
        print(f"  running: {name} ...", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0
        if proc.returncode not in (0, 1):
            # 0=PASS, 1=FAIL — both are valid evaluation outcomes. Anything else is a crash.
            print(f"    ERROR (exit {proc.returncode}) after {elapsed:.1f}s")
            print(proc.stderr[-500:] if proc.stderr else "(no stderr)")
            return None
        with open(json_out) as f:
            data = json.load(f)
        actual = data["actual_per_line"]
        exp_raw = data["expected"]
        # Normalize expected to a per-line dict for diff reporting
        if isinstance(exp_raw, int):
            exp_per_line = {"_total": exp_raw}
            actual_for_diff = {"_total": data["actual_total"]}
        else:
            exp_per_line = exp_raw
            actual_for_diff = actual
        diff = {k: actual_for_diff.get(k, 0) - v for k, v in exp_per_line.items()}
        abs_error = sum(abs(d) for d in diff.values())
        print(f"    done in {elapsed:.1f}s — abs_error={abs_error}")
        return {
            "actual": actual,
            "expected": exp_raw,
            "actual_total": data["actual_total"],
            "diff": diff,
            "abs_error": abs_error,
            "elapsed_s": round(elapsed, 1),
        }
    finally:
        try:
            os.unlink(json_out)
        except OSError:
            pass


def compare(name, current, baseline):
    """Return status string vs baseline."""
    if baseline is None or name not in baseline.get("results", {}):
        return "NEW"
    prev = baseline["results"][name]["abs_error"]
    curr = current["abs_error"]
    if curr < prev:
        return f"IMPROVED ({prev}→{curr})"
    if curr > prev:
        return f"REGRESSION ({prev}→{curr})"
    return f"STABLE ({curr})"


def fmt_diff(diff):
    return ", ".join(f"{k}={v:+d}" for k, v in diff.items()) or "-"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="line_config.json", help="Line config JSON (default: ./line_config.json)")
    parser.add_argument("--model", default=None, help="Model path (default: whatever evaluate.py picks)")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Overwrite latest.json even if regressions are detected")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't write results files (dry run)")
    parser.add_argument("rest", nargs=argparse.REMAINDER,
                        help="Extra args passed to evaluate.py (e.g. -- --confidence 0.4)")
    args = parser.parse_args()

    extra = [a for a in args.rest if a != "--"]

    cases = discover_cases()
    if not cases:
        print(f"No test cases found in {TESTS_DIR}")
        print("Add pairs of <name>.mp4 + <name>.expected.json there.")
        return 2

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.is_file():
        print(f"Line config not found: {config_path}")
        return 2

    # Resolve model: if user didn't pass --model, let evaluate.py pick its default
    # by importing default_model() and using whatever it returns.
    if args.model:
        model = args.model
    else:
        sys.path.insert(0, str(REPO_ROOT))
        from hailo_common import default_model
        model = default_model()

    print(f"Test suite: {len(cases)} case(s)")
    print(f"  config: {config_path}")
    print(f"  model:  {model}")
    print(f"  extra:  {' '.join(extra) if extra else '(none)'}")
    print()

    baseline = None
    baseline_path = RESULTS_DIR / "latest.json"
    if baseline_path.is_file():
        with open(baseline_path) as f:
            baseline = json.load(f)
        print(f"Baseline: {baseline_path} ({baseline.get('timestamp', '?')})")
    else:
        print("Baseline: none (first run)")
    print()

    results = {}
    suite_t0 = time.time()
    for name, mp4, expected in cases:
        r = run_case(name, mp4, expected, config_path, model, extra)
        if r is not None:
            results[name] = r
    suite_secs = time.time() - suite_t0

    # Report
    print()
    print("=" * 78)
    print(f"{'Case':<22} {'AbsErr':>7} {'Diff':<24} {'Status':<22}")
    print("-" * 78)
    regressions = []
    for name, r in results.items():
        status = compare(name, r, baseline)
        if status.startswith("REGRESSION"):
            regressions.append(name)
        print(f"{name:<22} {r['abs_error']:>7} {fmt_diff(r['diff']):<24} {status:<22}")
    if baseline:
        for name in baseline.get("results", {}):
            if name not in results:
                print(f"{name:<22} {'-':>7} {'-':<24} REMOVED")
    print("-" * 78)
    print(f"{len(results)} case(s) in {suite_secs:.1f}s, "
          f"{len(regressions)} regression(s)")
    print()

    # Save
    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        payload = {
            "timestamp": timestamp,
            "config": str(config_path),
            "model": str(model),
            "extra_args": extra,
            "suite_seconds": round(suite_secs, 1),
            "results": results,
        }
        history_path = RESULTS_DIR / f"{timestamp}.json"
        with open(history_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"History: {history_path}")

        if regressions and not args.update_baseline:
            print(f"Baseline NOT updated due to regressions: {', '.join(regressions)}")
            print("Re-run with --update-baseline to accept the new results as the baseline.")
        else:
            with open(baseline_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"Baseline updated: {baseline_path}")

    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
