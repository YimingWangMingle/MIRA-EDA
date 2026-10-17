from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

WINDOW_SECONDS = 60.


def flush_aggregate(directory, pid, events, totals):
    directory = Path(directory)
    durations = defaultdict(list)
    for event in events:
        if event['kind'] != 'query':
            continue
        backend = event['backend']
        window = int(event['elapsed_seconds'] // WINDOW_SECONDS)
        key = (backend, window)
        if key not in totals:
            tag = hashlib.sha256(backend.encode()).hexdigest()[:12]
            totals[key] = dict(backend=backend, window_start_seconds=window*WINDOW_SECONDS,
                query_count=0, query_errors=0, duration_sum_seconds=0.,
                latency_file='latencies_{}_{}.bin'.format(pid, tag))
        row = totals[key]
        if event['error']:
            row['query_errors'] += 1
        else:
            row['query_count'] += 1
            row['duration_sum_seconds'] += event['duration_seconds']
            durations[row['latency_file']].append((window, event['duration_seconds']))
    for filename, values in durations.items():
        with (directory/filename).open('ab') as stream:
            stream.write(np.asarray(values, dtype='<f8').tobytes())
    target = directory/'aggregate_{}.json'.format(pid)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(window_seconds=WINDOW_SECONDS,
        windows=[totals[key] for key in sorted(totals)]), allow_nan=False), encoding='utf-8')
    temporary.replace(target)


def summarize_aggregate(directory, window_seconds):
    directory = Path(directory)
    if window_seconds != WINDOW_SECONDS:
        raise ValueError('Compact query snapshots use fixed 60-second windows')
    groups = {}
    all_durations = []
    complete = True
    successes = errors = 0
    duration_sum = 0.
    for path in sorted(directory.glob('aggregate_*.json')):
        snapshot = json.loads(path.read_text(encoding='utf-8'))
        if snapshot['window_seconds'] != WINDOW_SECONDS:
            raise ValueError('Query window size mismatch')
        cached = {}
        for row in snapshot['windows']:
            key = (row['backend'], row['window_start_seconds'])
            bucket = groups.setdefault(key, dict(count=0, errors=0, total=0., durations=[], complete=True))
            count = int(row['query_count'])
            successes += count
            errors += int(row['query_errors'])
            duration_sum += float(row['duration_sum_seconds'])
            bucket['count'] += count
            bucket['errors'] += int(row['query_errors'])
            bucket['total'] += float(row['duration_sum_seconds'])
            filename = row['latency_file']
            if Path(filename).name != filename:
                raise ValueError('Invalid temporary latency filename')
            if filename not in cached:
                raw = directory/filename
                if raw.is_file():
                    values = np.fromfile(raw, dtype='<f8')
                    if len(values) % 2:
                        raise ValueError('Incomplete temporary latency record: '+str(raw))
                    cached[filename] = values.reshape((-1, 2))
                else:
                    cached[filename] = np.empty((0, 2))
            values = cached[filename]
            durations = values[values[:, 0] == row['window_start_seconds']/WINDOW_SECONDS, 1]
            if len(durations) != count:
                complete = False
                bucket['complete'] = False
            if len(durations):
                bucket['durations'].append(durations)
                all_durations.append(durations)
    def p95(parts, valid):
        return float(np.percentile(np.concatenate(parts), 95)) if parts and valid else None
    series = []
    for (backend, start), bucket in sorted(groups.items()):
        series.append(dict(backend=backend, window_start_seconds=start,
            query_count=bucket['count'], query_errors=bucket['errors'],
            query_mean_seconds=bucket['total']/bucket['count'] if bucket['count'] else None,
            query_p95_seconds=p95(bucket['durations'], bucket['complete'])))
    return dict(query_count=successes+errors, query_errors=errors,
        query_mean_seconds=duration_sum/successes if successes else None,
        query_p95_seconds=p95(all_durations, complete), time_windows=series,
        profiling_mode='aggregate', p95_method='exact from temporary successful-query durations' if complete
            else 'unavailable: temporary durations absent or incomplete')
