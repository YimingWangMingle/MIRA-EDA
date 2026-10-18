import atexit
from contextlib import contextmanager
from functools import wraps
import json
from multiprocessing.managers import BaseManager
import os
from pathlib import Path
import pickle
import secrets
import subprocess
import sys
import tempfile
import threading
import time

_ENDPOINT = 'MIRA_QUERY_ENDPOINT'
_client = None
_client_identity = None
_client_lock = threading.Lock()
_owner = None
_reports = set()


class QueryManager(BaseManager):
    pass


QueryManager.register('scheduler')


def execution_options():
    def flag(name):
        value = os.environ.get(name, '1')
        if value not in ('0', '1'):
            raise ValueError(name+' must be 0 or 1')
        return value == '1'
    cache_mb = float(os.environ.get('MIRA_QUERY_CACHE_MB', '128'))
    batch_ms = float(os.environ.get('MIRA_QUERY_BATCH_MS', '2'))
    if not (0 <= cache_mb < float('inf') and 0 <= batch_ms < float('inf')):
        raise ValueError('Query cache and batching limits must be finite and nonnegative')
    return dict(result_reuse=flag('MIRA_RESULT_REUSE'),
        shared_execution=flag('MIRA_SHARED_EXECUTION'),
        cache_bytes=int(cache_mb*1024**2), batch_seconds=batch_ms/1000.)


def service_available():
    return bool(os.environ.get(_ENDPOINT)) and not os.environ.get('MIRA_QUERY_SERVER')


def _service():
    global _client, _client_identity
    identity = (os.getpid(), os.environ.get(_ENDPOINT))
    with _client_lock:
        if identity != _client_identity:
            if not service_available():
                raise RuntimeError('Start the shared query service before starting workers')
            endpoint = json.loads(identity[1])
            manager = QueryManager(address=('127.0.0.1', endpoint['port']),
                                   authkey=bytes.fromhex(endpoint['auth']))
            manager.connect()
            _client = manager.scheduler()
            _client_identity = identity
    return _client


def register_backend(backend):
    return _service().register(pickle.dumps(backend, protocol=4))


def submit_query(snapshot, request):
    return pickle.loads(_service().query(snapshot, pickle.dumps(request, protocol=4)))


def service_stats():
    from .query_config import heuristic_pruning_enabled, QUERY_PROTOCOL_VERSION
    return dict(_service().stats(), heuristic_pruning=heuristic_pruning_enabled(),
                query_protocol_version=QUERY_PROTOCOL_VERSION)


def set_query_report_path(path):
    _reports.add(str(Path(path).resolve()))


def ensure_query_service():
    global _owner
    if service_available():
        return False
    execution_options()
    temporary = tempfile.TemporaryDirectory(prefix='mira_query_service_')
    endpoint_path = Path(temporary.name)/'endpoint.json'
    auth = secrets.token_hex(32)
    from .training_runtime import single_thread_environment
    env = single_thread_environment()
    env['MIRA_QUERY_AUTH'] = auth
    env['MIRA_QUERY_SERVER'] = '1'

    paths = [str(Path(__file__).resolve().parents[1])] + [os.path.abspath(p or os.getcwd()) for p in sys.path]
    env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(paths))
    env['PYTHONDONTWRITEBYTECODE'] = '1'

    env.pop('MIRA_RESOURCE_DIR', None)
    log = (Path(temporary.name)/'server.log').open('w+', encoding='utf-8')
    process = subprocess.Popen([sys.executable, '-m', 'revision_support.query_service', str(endpoint_path)],
        env=env, stdout=log, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    try:
        deadline = time.monotonic()+30
        while not endpoint_path.exists():
            if process.poll() is not None or time.monotonic() >= deadline:
                log.seek(0)
                raise RuntimeError('Shared query service did not start: '+log.read())
            time.sleep(.02)
        endpoint = json.loads(endpoint_path.read_text(encoding='utf-8'))
        endpoint['auth'] = auth
        os.environ[_ENDPOINT] = json.dumps(endpoint)
        _owner = (os.getpid(), process, temporary, log)

        service_stats()
        if os.environ.get('MIRA_RESOURCE_DIR'):
            set_query_report_path(Path(os.environ['MIRA_RESOURCE_DIR']).parent/'query_scheduler.json')
        return True
    except BaseException:
        process.terminate()
        process.wait(timeout=10)
        log.close()
        temporary.cleanup()
        os.environ.pop(_ENDPOINT, None)
        os.environ.pop('MIRA_COVERTYPE_QUERY_SNAPSHOT', None)
        _owner = None
        raise


def stop_query_service():
    global _owner, _client, _client_identity
    if _owner is None or _owner[0] != os.getpid():
        return
    _, process, temporary, log = _owner
    try:
        if process.poll() is None:
            report = service_stats()
            for path in _reports:
                target = Path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    finally:
        process.terminate()
        process.wait(timeout=10)
        log.close()
        temporary.cleanup()
        os.environ.pop(_ENDPOINT, None)
        _owner = _client = _client_identity = None
        os.environ.pop('MIRA_COVERTYPE_QUERY_SNAPSHOT', None)
        _reports.clear()


@contextmanager
def query_service_scope(report_path=None):
    started = ensure_query_service()
    if report_path is not None:
        set_query_report_path(report_path)
    try:
        yield
    finally:
        if started:
            stop_query_service()


def shared_query_run(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        runtime_args = args[0] if args and hasattr(args[0], '__dict__') else kwargs.get('args')
        if runtime_args is not None:
            from .query_config import configure_query_processing
            configure_query_processing(runtime_args)
        with query_service_scope():
            return function(*args, **kwargs)
    return wrapped


def main():
    from .query_scheduler import QueryScheduler
    scheduler = QueryScheduler(**execution_options())
    QueryManager.register('scheduler', callable=lambda: scheduler)
    manager = QueryManager(address=('127.0.0.1', 0), authkey=bytes.fromhex(os.environ['MIRA_QUERY_AUTH']))
    server = manager.get_server()
    path = Path(sys.argv[1])
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(port=server.address[1])), encoding='utf-8')
    temporary.replace(path)
    server.serve_forever()


atexit.register(stop_query_service)

if __name__ == '__main__':
    main()
