from collections import OrderedDict, defaultdict
from concurrent.futures import Future
import hashlib
import os
import pickle
import threading
import time


class QueryScheduler:
    def __init__(self, result_reuse=True, shared_execution=True,
                 cache_bytes=128*1024**2, batch_seconds=.002):
        self.result_reuse = bool(result_reuse)
        self.shared_execution = bool(shared_execution)
        self.cache_limit = max(0, int(cache_bytes))
        self.batch_seconds = max(0., float(batch_seconds))
        self._condition = threading.Condition()
        self._backends = {}
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._pending = {}
        self._queue = []
        self._closed = False
        self._metrics = dict(requests=0, cache_hits=0, coalesced_requests=0,
            evaluations=0, failed_evaluations=0, execution_groups=0,
            shared_intermediates=0, backend_seconds=0.)
        self._thread = threading.Thread(target=self._run, name='shared-query-scheduler', daemon=True)
        self._thread.start()

    def register(self, payload):

        snapshot = hashlib.sha256(payload).hexdigest()
        with self._condition:
            if snapshot not in self._backends:
                self._backends[snapshot] = pickle.loads(payload)
        return snapshot

    def query(self, snapshot, payload):
        key = (snapshot, payload)
        with self._condition:
            if self._closed:
                raise RuntimeError('Query scheduler is closed')
            if snapshot not in self._backends:
                raise ValueError('Unregistered immutable dataset snapshot')
            self._metrics['requests'] += 1
            if self.result_reuse and key in self._cache:
                self._metrics['cache_hits'] += 1
                self._cache.move_to_end(key)
                return self._cache[key]
            future = self._pending.get(key) if self.result_reuse else None
            if future is not None:
                self._metrics['coalesced_requests'] += 1
            else:
                future = Future()
                if self.result_reuse:
                    self._pending[key] = future
                self._queue.append((key, future))
                self._condition.notify()
        return future.result()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._queue or self._closed)
                if not self._queue and self._closed:
                    return
                deadline = time.monotonic() + self.batch_seconds
                while not self._closed and time.monotonic() < deadline:
                    self._condition.wait(deadline-time.monotonic())
                pending, self._queue = self._queue, []
            groups = defaultdict(list)
            for index, (key, future) in enumerate(pending):
                snapshot, payload = key
                try:
                    request = pickle.loads(payload)
                    group = self._backends[snapshot].group_key(request) if self.shared_execution else index
                    groups[(snapshot, group)].append((key, future, request))
                except Exception as exc:
                    self._finish(key, future, error=exc)
            for (snapshot, _), items in groups.items():
                started = time.perf_counter()
                error = None
                reused = 0
                try:
                    values, reused = self._backends[snapshot].evaluate_many(
                        [item[2] for item in items], share=self.shared_execution)
                    if len(values) != len(items):
                        raise ValueError('Backend returned the wrong number of query results')

                    results = [pickle.dumps(value, protocol=4) for value in values]
                except Exception as exc:
                    error = exc
                finally:
                    with self._condition:
                        self._metrics['failed_evaluations'] += len(items) if error is not None else 0
                        self._metrics['shared_intermediates'] += int(reused) if error is None else 0
                        self._metrics['evaluations'] += len(items)
                        self._metrics['execution_groups'] += 1
                        self._metrics['backend_seconds'] += time.perf_counter()-started

                if error is None:
                    for (key, future, _), result in zip(items, results):
                        self._finish(key, future, result=result)
                else:
                    for key, future, _ in items:
                        self._finish(key, future, error=error)

    def _finish(self, key, future, result=None, error=None):
        with self._condition:
            if self.result_reuse:
                self._pending.pop(key, None)
                if error is None:
                    size = len(key[1]) + len(result)
                    if size <= self.cache_limit:
                        while self._cache and self._cache_bytes + size > self.cache_limit:
                            old_key, old_result = self._cache.popitem(last=False)
                            self._cache_bytes -= len(old_key[1]) + len(old_result)
                        self._cache[key] = result
                        self._cache_bytes += size
            if error is None:
                future.set_result(result)
            else:
                future.set_exception(error)

    def stats(self):
        with self._condition:
            return dict(self._metrics, cached_bytes=self._cache_bytes,
                server_pid=os.getpid(),
                registered_snapshots=len(self._backends), result_reuse=self.result_reuse,
                shared_execution=self.shared_execution, cache_limit_bytes=self.cache_limit,
                batch_seconds=self.batch_seconds)

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
