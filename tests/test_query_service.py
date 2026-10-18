import importlib
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ATENA_dataset/ATENA-A-EDA/benchmark'))


def test_service_runs_in_separate_process_and_returns_independent_results(tmp_path):
    spec = importlib.util.find_spec('revision_support.query_service')
    assert spec is not None, 'Cross-worker query service is missing'
    service = importlib.import_module('revision_support.query_service')
    from revision_support.query_backends import DisplayBackend
    from atena.simulation.display import DisplayCalculator
    from atena.simulation.state import EnvStateTuple
    from types import SimpleNamespace
    calculator = DisplayCalculator(SimpleNamespace(
        dataset_df=pd.DataFrame(dict(x=[1., 2., 3.])), columns=['x'], numeric_columns=['x']))
    with service.query_service_scope(report_path=tmp_path/'scheduler.json'):
        snapshot = service.register_backend(DisplayBackend(calculator))
        state = EnvStateTuple.create_empty_state()
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda _: service.submit_query(snapshot, state), range(3)))
        results[0][1][0].iloc[0, 0] = 900
        assert service.submit_query(snapshot, state)[1][0].iloc[0, 0] == 1
        stats = service.service_stats()
        assert stats['server_pid'] != os.getpid()
        assert stats['evaluations'] == 1
        assert stats['requests'] == 4
    assert (tmp_path/'scheduler.json').is_file()
    assert not service.service_available()


def test_galaxy_common_execution_caches_exact_results_and_static_summaries(tmp_path):
    from revision_support.query_service import query_service_scope, service_stats
    from revision_support.galaxy_execution import execute_galaxy_operator
    from app.pipelines.pipeline import Pipeline
    from app.pipelines.dataset import Dataset
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.initial_collection = pd.DataFrame(dict(a=[1, 1, 2], b=[2, 3, 2]))
    pipeline.exploration_columns = ['a', 'b']
    pipeline.ordered_dimensions = {}
    pipeline.discrete_categories_count = 10
    dataset = Dataset(data=pipeline.initial_collection.copy())
    original = execute_galaxy_operator(pipeline, dataset.copy(), 'by_facet-&-a')
    with query_service_scope(report_path=tmp_path/'galaxy.json'):
        assert execute_galaxy_operator(pipeline, dataset.copy(), 'by_neighbors-&-a') == []
        assert service_stats()['requests'] == 0
        for _ in range(2):
            actual = execute_galaxy_operator(pipeline, dataset.copy(), 'by_facet-&-a')
            for got, expected in zip(actual, original):
                pd.testing.assert_frame_equal(got.data, expected.data)
                assert hasattr(got, '_query_summary')
        assert service_stats()['evaluations'] == 1
        assert service_stats()['cache_hits'] == 1


def test_benchmark_display_integration_keeps_back_context_local(tmp_path):
    from revision_support.query_service import query_service_scope, service_stats
    from atena.simulation.actions_simulator import ActionsSimulator
    from atena.simulation.actions import GroupAction, BackAction, AggregationFunction
    from types import SimpleNamespace
    dataset = SimpleNamespace(dataset_df=pd.DataFrame(dict(a=['x', 'y'], n=[1., 2.])),
                              columns=['a', 'n'], numeric_columns=['n'])
    with query_service_scope(report_path=tmp_path/'display.json'):
        left, right = ActionsSimulator(dataset), ActionsSimulator(dataset)
        left.execute_action(GroupAction('a', 'n', AggregationFunction.COUNT))
        right.execute_action(GroupAction('n', 'n', AggregationFunction.COUNT))
        before = len(right.simulation_state.states_stack)
        left.execute_action(BackAction())
        assert len(left.simulation_state.states_stack) == 1
        assert len(right.simulation_state.states_stack) == before
        assert service_stats()['requests'] == 5
        assert service_stats()['cache_hits'] == 2


def test_native_atena_shared_result_preserves_context_and_honors_no_reuse(tmp_path, monkeypatch):
    import operator
    from types import SimpleNamespace
    import numpy as np
    sys.path.insert(0, str(ROOT/'ATENA_dataset/ATENA-A-EDA/atena-basic'))
    from gym_atena.envs.env_properties import EnvProp
    from gym_atena.lib.helpers import EnvStateTuple
    import Utilities.Configuration.config as cfg
    from revision_support.query_service import query_service_scope, service_stats
    prop = object.__new__(EnvProp)
    prop.env_dataset_prop = SimpleNamespace(KEYS=['a', 'n'], NUMERIC_KEYS=['n'])
    prop.INT_OPERATOR_MAP_ATENA = {0: operator.eq}
    prop.AGG_MAP_ATENA = {0: len}
    df = pd.DataFrame(dict(a=['x', 'x', 'y'], n=[1., 2., 3.]))
    state = EnvStateTuple.create_empty_state()
    monkeypatch.setattr(cfg, 'obs_with_step_num', True)
    monkeypatch.setattr(cfg, 'stack_obs_num', 1)
    expected = prop.calc_display_vector(df, state, step_number=1)
    monkeypatch.setenv('MIRA_RESULT_REUSE', '0')
    with query_service_scope(report_path=tmp_path/'native.json'):
        for step in (1, 2):
            actual = prop.calc_display_vector(df, state, memo={}, dataset_number=0, step_number=step)
            np.testing.assert_array_equal(actual[0][cfg.MAX_NUM_OF_STEPS:], expected[0][cfg.MAX_NUM_OF_STEPS:])
            assert actual[0][step-1] == 1
            assert actual[0][:cfg.MAX_NUM_OF_STEPS].sum() == 1
            assert actual[1] == expected[1]
            pd.testing.assert_frame_equal(actual[2][0], expected[2][0])
        assert service_stats()['evaluations'] == 2
        assert service_stats()['cache_hits'] == 0
