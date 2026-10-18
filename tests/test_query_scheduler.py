import importlib
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest


class ExampleBackend:
    def group_key(self, request):
        return request[0]

    def evaluate_many(self, requests, share=True):
        time.sleep(.02)
        if any(value == 'error' for _, value in requests):
            raise ValueError('backend failure')
        return [[value] for _, value in requests], max(0, len(requests)-1)


def scheduler(**options):
    spec = importlib.util.find_spec('revision_support.query_scheduler')
    assert spec is not None, 'Shared query scheduler is missing'
    module = importlib.import_module('revision_support.query_scheduler')
    return module.QueryScheduler(batch_seconds=.03, **options)


def register(service, backend=None):
    return service.register(pickle.dumps(backend or ExampleBackend(), protocol=4))


def request(service, snapshot, value):
    return pickle.loads(service.query(snapshot, pickle.dumps(value, protocol=4)))


def concurrent(service, snapshot, values):
    barrier = threading.Barrier(len(values))
    def run(value):
        barrier.wait(timeout=5)
        return request(service, snapshot, value)
    with ThreadPoolExecutor(max_workers=len(values)) as pool:
        return list(pool.map(run, values))


def test_coalesces_pending_and_reuses_completed_without_aliasing():
    with scheduler() as service:
        snapshot = register(service)
        assert concurrent(service, snapshot, [('parent', 1)]*4) == [[1]]*4
        result = request(service, snapshot, ('parent', 1))
        result.append('local mutation')
        assert request(service, snapshot, ('parent', 1)) == [1]
        metrics = service.stats()
        assert metrics['evaluations'] == 1
        assert metrics['coalesced_requests'] == 3
        assert metrics['cache_hits'] == 2


def test_batches_distinct_queries_and_no_shared_execution_keeps_reuse():
    for shared, saved in [(True, 2), (False, 0)]:
        with scheduler(shared_execution=shared) as service:
            snapshot = register(service)
            assert concurrent(service, snapshot, [('p', i) for i in range(3)]) == [[0], [1], [2]]
            assert request(service, snapshot, ('p', 0)) == [0]
            assert service.stats()['shared_intermediates'] == saved
            assert service.stats()['evaluations'] == 3
            assert service.stats()['cache_hits'] == 1


def test_no_reuse_disables_both_cache_and_coalescing():
    with scheduler(result_reuse=False) as service:
        snapshot = register(service)
        concurrent(service, snapshot, [('p', 1)]*3)
        request(service, snapshot, ('p', 1))
        metrics = service.stats()
        assert metrics['evaluations'] == 4
        assert metrics['cache_hits'] == metrics['coalesced_requests'] == 0


def test_failure_releases_waiters_and_is_not_cached():
    with scheduler() as service:
        snapshot = register(service)
        with pytest.raises(ValueError, match='backend failure'):
            concurrent(service, snapshot, [('p', 'error')]*3)
        assert request(service, snapshot, ('p', 2)) == [2]
        with pytest.raises(ValueError, match='backend failure'):
            request(service, snapshot, ('p', 'error'))
        assert service.stats()['failed_evaluations'] == 2


def test_bounded_cache_and_different_snapshot():
    with scheduler(cache_bytes=1) as service:
        first = register(service)
        other = ExampleBackend()
        other.snapshot_version = 2
        second = register(service, other)
        assert first != second
        request(service, first, ('p', 1))
        request(service, first, ('p', 1))
        request(service, second, ('p', 1))
        assert service.stats()['evaluations'] == 3
        assert service.stats()['cached_bytes'] <= 1
