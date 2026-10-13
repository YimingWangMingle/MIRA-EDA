"""Measure a fresh local experiment process and its descendants.

Example: python scripts/measure_experiment.py --output output/resources/run1
         -- python RL-launcher.py ...

The wall time includes imports, initialization, training, inline evaluation
and final saving. Peak memory is sampled SUM(RSS) over the local process tree,
not unique physical memory (shared pages can be counted more than once).
Use an isolated local Ray instance; remote/existing Ray workers are excluded.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revision_support.resources import summarize_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument('--profile-mode', choices=('events', 'aggregate'), default='events')
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval <= 0:
        parser.error("A command and a positive sampling interval are required")
    try:
        import psutil
    except ImportError:
        parser.error("Install psutil in this Python environment before measurement")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    events_dir = output / "query_profile"
    events_dir.mkdir()
    env = os.environ.copy()
    env["MIRA_RESOURCE_DIR"] = str(events_dir)
    env['MIRA_PROFILE_MODE'] = args.profile_mode
    env["MIRA_PROFILE_STARTED"] = str(time.time())
    started = time.perf_counter()
    peak_rss = samples = 0
    last_snapshot = started - 30.
    status = "launch_failed"
    returncode = None
    try:
        with (output / "console.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            root = psutil.Process(process.pid)
            while True:
                rss = 0
                try:
                    children = root.children(recursive=True)
                except psutil.NoSuchProcess:
                    children = []
                for child in [root, *children]:
                    try:
                        rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                peak_rss = max(peak_rss, rss)
                samples += 1
                now = time.perf_counter()
                if now-last_snapshot >= 30.:
                    progress = dict(status='running', elapsed_seconds=now-started,
                        peak_process_tree_rss_bytes=peak_rss, memory_samples=samples)
                    temporary = output/'resources_progress.tmp'
                    temporary.write_text(json.dumps(progress, indent=2), encoding='utf-8')
                    temporary.replace(output/'resources_progress.json')
                    last_snapshot = now
                returncode = process.poll()
                if returncode is not None:
                    break
                time.sleep(args.interval)
            status = "completed" if returncode == 0 else "failed"
    finally:
        wall_seconds = time.perf_counter() - started
        profile = summarize_events(events_dir)
        report = {
            "command": command, "cwd": os.getcwd(), "status": status,
            "returncode": returncode, "full_run_wall_seconds": wall_seconds,
            "peak_process_tree_rss_bytes": peak_rss, "sampling_interval_seconds": args.interval,
            "memory_samples": samples, "query_profile": profile,
            'query_profiling_mode': args.profile_mode,
            "memory_definition": "Sampled sum of local descendant RSS; shared pages may be counted multiple times.",
            "event_snapshot_note": "Events flush every 100 calls, at supported episode boundaries and normal exit; abrupt worker termination may lose the tail.",
        }
        (output / "resources.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        row = dict(status=status, full_run_wall_seconds=wall_seconds,
            peak_process_tree_rss_bytes=peak_rss, peak_rss_gib=peak_rss/1024**3,
            query_success_count=profile['query_count']-profile['query_errors'],
            query_errors=profile['query_errors'],
            query_mean_ms=profile['query_mean_seconds']*1000 if profile['query_mean_seconds'] is not None else None,
            query_p95_ms=profile['query_p95_seconds']*1000 if profile['query_p95_seconds'] is not None else None)
        with (output/'efficiency.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        (output/'resources_progress.json').write_text(json.dumps(dict(status=status,
            elapsed_seconds=wall_seconds, peak_process_tree_rss_bytes=peak_rss, memory_samples=samples),
            indent=2), encoding='utf-8')
        if profile["time_windows"]:
            with (output / "query_profile_over_time.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(profile["time_windows"][0]))
                writer.writeheader()
                writer.writerows(profile["time_windows"])
        if status == 'completed' and args.profile_mode == 'aggregate' and profile['query_p95_seconds'] is not None:
            # These files are created only by this measurement's profiler. Final
            # exact percentiles are now in resources.json and the two CSV reports.
            for path in events_dir.glob('latencies_*.bin'):
                path.unlink()
    print(output / "resources.json")
    return returncode if returncode is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
