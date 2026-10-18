import os
from types import SimpleNamespace
import weakref

from .query_service import register_backend, submit_query


def register_covertype_backend(data):
    from .query_backends import CovertypeBackend
    snapshot = register_backend(CovertypeBackend(data))
    os.environ['MIRA_COVERTYPE_QUERY_SNAPSHOT'] = snapshot
    return snapshot


def covertype_query(universe, set_id):
    snapshot = os.environ.get('MIRA_COVERTYPE_QUERY_SNAPSHOT')
    if not snapshot:
        raise RuntimeError('Register Covertype original rows before starting workers')
    return submit_query(snapshot, tuple(int(v) for v in universe.constraints[int(set_id)]))


def galaxy_query(pipeline, dataset, action):
    from .query_backends import GalaxyBackend
    endpoint = os.environ.get('MIRA_QUERY_ENDPOINT')
    snapshot = getattr(pipeline, '_query_snapshot', None)
    if snapshot is None or snapshot[0] != endpoint:

        pipeline.__dict__.pop('_query_snapshot', None)
        snapshot = (endpoint, register_backend(GalaxyBackend(pipeline)))
        pipeline._query_snapshot = snapshot
    request = (action, dataset.data, dataset.predicate, dataset.joins, dataset.tables, dataset.set_id)
    return submit_query(snapshot[1], request)


def benchmark_display(calculator, state):
    from .query_backends import DisplayBackend
    from atena.simulation.display import DisplayCalculator
    dataset = calculator.dataset
    endpoint = os.environ.get('MIRA_QUERY_ENDPOINT')
    snapshot = getattr(dataset, '_query_snapshot', None)
    if snapshot is None or snapshot[0] != endpoint:
        pure_dataset = SimpleNamespace(dataset_df=dataset.dataset_df,
            columns=dataset.columns, numeric_columns=dataset.numeric_columns)
        snapshot = (endpoint, register_backend(DisplayBackend(DisplayCalculator(pure_dataset))))
        dataset._query_snapshot = snapshot
    return submit_query(snapshot[1], state)


def native_display(properties, dataframe, state):
    from .query_backends import NativeDisplayBackend
    endpoint = os.environ.get('MIRA_QUERY_ENDPOINT')
    snapshots = getattr(properties, '_query_snapshots', {})
    key = id(dataframe)
    snapshot = snapshots.get(key)
    if snapshot is None or snapshot[0] != endpoint or snapshot[1]() is not dataframe:
        snapshot = (endpoint, weakref.ref(dataframe), register_backend(NativeDisplayBackend(properties, dataframe)))
        snapshots[key] = snapshot
        properties._query_snapshots = snapshots
    return submit_query(snapshot[2], state)
