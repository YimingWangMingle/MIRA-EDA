"""Opt-in query timing and time-window summaries."""

import atexit
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import os
from pathlib import Path
import time

import numpy as np

_depth = ContextVar('query_depth', default=0)
_context = ContextVar('query_context', default={})
_identity = None
_pending = []
_sequence = 0
_aggregate = {}


def _enabled():
    return bool(os.environ.get('MIRA_RESOURCE_DIR'))


def _initialize():
    global _identity, _pending, _sequence, _aggregate
    identity = (os.getpid(), os.environ.get('MIRA_RESOURCE_DIR'), os.environ.get('MIRA_PROFILE_MODE', 'events'))
    if identity != _identity:
        flush_events()
        _identity, _pending, _sequence = identity, [], 0
        _aggregate = {}


def _record(event):
    global _sequence
    _initialize()
    _sequence += 1
    now = time.time()
    event.update(_context.get())
    event.update(pid=os.getpid(), event_index=_sequence, timestamp=now,
                 elapsed_seconds=max(0., now-float(os.environ.get('MIRA_PROFILE_STARTED', now))))
    _pending.append(event)
    if len(_pending) >= 100:
        flush_events()


def flush_events():
    if not _pending or _identity is None or _identity[0] != os.getpid():
        return
    directory = Path(_identity[1])
    if not directory.exists():
        return
    if _identity[2] == 'aggregate':
        from .query_aggregate import flush_aggregate
        flush_aggregate(directory, os.getpid(), _pending, _aggregate)
        _pending.clear()
        return
    with (directory / f'events_{os.getpid()}.jsonl').open('a', encoding='utf-8') as stream:
        stream.writelines(json.dumps(event, allow_nan=False) + '\n' for event in _pending)
    _pending.clear()


@contextmanager
def query_timer(backend, operation, **context):
    if not _enabled() or _depth.get():
        yield
        return
    token = _depth.set(1)
    started = time.perf_counter()
    error = None
    try:
        yield
    except Exception as exc:
        error = type(exc).__name__
        raise
    finally:
        duration = time.perf_counter() - started
        _depth.reset(token)
        _record(dict(kind='query', backend=backend, operation=operation,
                     duration_seconds=duration, error=error, **context))


def timed_query(backend, operation):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with query_timer(backend, operation):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def query_step(function):
    """Attach the episode-local step; do not include the whole step in latency."""
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        if not _enabled():
            return function(self, *args, **kwargs)
        context = dict(step=int(getattr(self, 'step_count', 0)) + 1,
                       worker_id=int(getattr(self, 'agentId', -1)))
        token = _context.set(context)
        try:
            return function(self, *args, **kwargs)
        finally:
            _context.reset(token)
    return wrapped




def summarize_events(directory, window_seconds=60.):
    if list(Path(directory).glob('aggregate_*.json')):
        if list(Path(directory).glob('events_*.jsonl')):
            raise ValueError('Mixed raw and aggregate profiling in one experiment')
        from .query_aggregate import summarize_aggregate
        return summarize_aggregate(directory, window_seconds)
    latencies = []
    windows = defaultdict(lambda: {'query': [], 'errors': 0})
    query_count = query_errors = 0
    for path in sorted(Path(directory).glob('events_*.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            event = json.loads(line)
            if event['kind'] != 'query':
                continue  # Ignore retired diagnostic events in historical logs.
            window = int(event['elapsed_seconds'] // window_seconds)
            bucket = windows[(event['backend'], window)]
            if event['kind'] == 'query':
                query_count += 1
                if event['error']:
                    query_errors += 1
                    bucket['errors'] += 1
                else:
                    latencies.append(event['duration_seconds'])
                    bucket['query'].append(event['duration_seconds'])
    def average(values):
        return float(np.mean(values)) if values else None
    def percentile(values):
        return float(np.percentile(values, 95)) if values else None
    series = []
    for (backend, window), bucket in sorted(windows.items()):
        series.append(dict(backend=backend, window_start_seconds=window*window_seconds,
            query_count=len(bucket['query']), query_errors=bucket['errors'],
            query_mean_seconds=average(bucket['query']), query_p95_seconds=percentile(bucket['query'])))
    return dict(query_count=query_count, query_errors=query_errors,
        query_mean_seconds=average(latencies), query_p95_seconds=percentile(latencies),
        time_windows=series)


atexit.register(flush_events)
