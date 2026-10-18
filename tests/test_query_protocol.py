import argparse
import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'covertype-exploration'))


def config_module():
    assert importlib.util.find_spec('revision_support.query_config') is not None, 'Query ablation protocol is missing'
    return importlib.import_module('revision_support.query_config')


@pytest.mark.parametrize('variant, expected', [
    ('none', (True, True, True)),
    ('no_result_reuse', (False, True, True)),
    ('no_heuristic_pruning', (True, False, True)),
    ('no_shared_execution', (True, True, False)),
])
def test_formal_protocol_is_one_factor_and_overrides_ambient_flags(monkeypatch, variant, expected):
    config = config_module()
    for name in ('MIRA_RESULT_REUSE', 'MIRA_HEURISTIC_PRUNING', 'MIRA_SHARED_EXECUTION'):
        monkeypatch.setenv(name, '0')
    parser = argparse.ArgumentParser()
    config.add_query_arguments(parser)
    args = parser.parse_args(['--query_ablation', variant])
    config.configure_query_processing(args, method='mira')
    actual = args.query_processing
    assert tuple(actual[k] for k in ('result_reuse', 'heuristic_pruning', 'shared_execution')) == expected
    assert tuple(__import__('os').environ[k] == '1' for k in
                 ('MIRA_RESULT_REUSE', 'MIRA_HEURISTIC_PRUNING', 'MIRA_SHARED_EXECUTION')) == expected


@pytest.mark.parametrize('method', ['dora', 'atena', 'pure_a3c', 'random', 'greedy_eda'])
def test_baselines_cannot_be_mislabeled_as_mira_query_ablations(method):
    config = config_module()
    with pytest.raises(ValueError, match='MIRA'):
        config.configure_query_processing(SimpleNamespace(query_ablation='no_result_reuse'), method=method)


@pytest.mark.parametrize('encoder,reward', [('vae', 'directional'), ('learned', 'rnd')])
def test_query_replacement_cannot_be_combined_with_other_replacements(encoder, reward):
    config = config_module()
    args = SimpleNamespace(query_ablation='no_shared_execution', encoder_variant=encoder, exploration_reward=reward)
    with pytest.raises(ValueError, match='one component'):
        config.configure_query_processing(args, method='mira')


def test_runner_registers_three_query_variants_with_full_defaults():
    from verify import load_runner
    run = load_runner(ROOT)
    required = {'query_no_result_reuse', 'query_no_heuristic_pruning', 'query_no_shared_execution'}
    assert required <= set(run.ABLATIONS), 'Three query variants must be scheduled'
    jobs = run.build_jobs(seeds=(1,))
    for job in jobs:
        protocol = job['query_processing']
        disabled = [name for name in ('result_reuse', 'heuristic_pruning', 'shared_execution') if not protocol[name]]
        if job['method'] in required:
            assert disabled == [job['method'][len('query_no_'):]]
        else:
            assert disabled == []
        if job['dataset'] == 'galaxy' and job['method'] == 'dora':
            assert job['rewards_path'].endswith('_dora_rewards.csv')


def test_every_formal_command_accepts_the_common_query_protocol():
    from verify import parser_from_source, load_runner
    for job in load_runner(ROOT).build_jobs(seeds=(1,)):
        args = parser_from_source(ROOT, job['dataset'], job['method']).parse_args(job['command'][2:])
        assert args.query_ablation == job['query_processing']['query_ablation']


def test_covertype_online_backend_matches_exact_rows_and_shares_prefixes():
    from revision_support import query_backends
    from covertype_rl.fixed_sets import _state_for_indices
    assert hasattr(query_backends, 'CovertypeBackend'), 'Live Covertype execution is missing'
    data = SimpleNamespace(continuous_bins=np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8),
        continuous_norm=np.array([[.1, .2], [.3, .4], [.5, .6], [.7, .8]], dtype=np.float32),
        cover_type=np.array([1, 2, 1, 2]), wilderness=np.array([1, 2, 1, 2]), soil=np.array([1, 2, 1, 2]))
    backend = query_backends.CovertypeBackend(data)
    queries = [(0, 0, -1, -1, -1), (0, 1, -1, -1, -1)]
    full, saved = backend.evaluate_many(queries, share=True)
    separate, none_saved = backend.evaluate_many(queries, share=False)
    assert saved > 0 and none_saved == 0
    for i, (actual, independent) in enumerate(zip(full, separate)):
        np.testing.assert_array_equal(actual['indices'], [i])
        np.testing.assert_array_equal(actual['state'], _state_for_indices(data, queries[i], np.array([i])))
        np.testing.assert_array_equal(actual['state'], independent['state'])
        assert actual['set_size'] == 1


def test_covertype_environment_keeps_rewards_local_on_cache_hits(tmp_path):
    from revision_support import query_clients
    from revision_support.query_service import query_service_scope, service_stats
    from covertype_rl.environment import FixedSetEnvironment
    from covertype_rl.fixed_sets import _state_for_indices
    assert hasattr(query_clients, 'register_covertype_backend'), 'Covertype workers need a shared query backend'
    data = SimpleNamespace(continuous_bins=np.array([[0], [1]], dtype=np.int8),
        continuous_norm=np.array([[.1], [.9]], dtype=np.float32),
        cover_type=np.array([1, 2]), wilderness=np.array([1, 2]), soil=np.array([1, 2]))
    constraint = (0, -1, -1, -1)
    expected = _state_for_indices(data, constraint, np.array([0]))
    universe = SimpleNamespace(metadata={}, constraints=np.array([constraint]), target_set_size=1,
        state_for_set=lambda _: expected.copy(), size_for_set=lambda _: 1,
        targets_for_set=lambda _: np.array([0]), next_set=lambda *_: 0)
    with query_service_scope(report_path=tmp_path/'query.json'):
        query_clients.register_covertype_backend(data)
        left, right = [FixedSetEnvironment(universe, []) for _ in range(2)]
        left.reset(); right.reset()
        first = left.step(0)[1]['extrinsic_reward']
        repeat = left.step(0)[1]['extrinsic_reward']
        independent = right.step(0)[1]['extrinsic_reward']
        assert first > 0 and repeat == 0 and independent == first
        assert service_stats()['evaluations'] == 1
        assert service_stats()['cache_hits'] == 2
    assert 'MIRA_COVERTYPE_QUERY_SNAPSHOT' not in __import__('os').environ


@pytest.mark.parametrize('relative', ['covertype-exploration/baselines/random/run.py',
                                     'covertype-exploration/baselines/greedy_eda/run.py'])
def test_standalone_covertype_cli_bootstraps_imports(relative, tmp_path):
    import subprocess
    result = subprocess.run([sys.executable, '-B', str(ROOT / relative), '--help'],
                            cwd=tmp_path, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert '--query_ablation' in result.stdout


def test_random_family_sampler_only_selects_retained_actions():
    from baselines.random.random_baseline import sample_random_operator_action
    rng = np.random.default_rng(42)
    groups = {'by_facet': [0, 1], 'by_superset': [2], 'by_neighbors': [3], 'by_distribution': [4]}
    samples = {sample_random_operator_action(groups, rng, valid_ids=[1, 3]) for _ in range(100)}
    assert samples == {1, 3}


@pytest.mark.parametrize('variant', ['none', 'no_result_reuse', 'no_heuristic_pruning', 'no_shared_execution'])
def test_protocol_reaches_real_service_and_preserves_query_results(variant, tmp_path, monkeypatch):
    import os
    import json
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from revision_support.query_config import configure_query_processing, query_worker_environment
    from revision_support.query_service import query_service_scope, submit_query, service_stats
    from revision_support.query_clients import register_covertype_backend
    for key in ('MIRA_RESULT_REUSE', 'MIRA_HEURISTIC_PRUNING', 'MIRA_SHARED_EXECUTION',
                'MIRA_QUERY_CACHE_MB', 'MIRA_QUERY_BATCH_MS'):
        monkeypatch.setenv(key, os.environ.get(key, '1'))
    args = SimpleNamespace(query_ablation=variant, query_batch_ms=100.)
    configure_query_processing(args)
    data = SimpleNamespace(continuous_bins=np.array([[0, 0], [0, 1]], dtype=np.int8),
        continuous_norm=np.array([[.1, .2], [.3, .4]], dtype=np.float32),
        cover_type=np.array([1, 2]), wilderness=np.array([1, 2]), soil=np.array([1, 2]))
    queries = [(0, 0, -1, -1, -1), (0, 1, -1, -1, -1)]
    with query_service_scope(tmp_path/'report.json'):
        snapshot = register_covertype_backend(data)
        worker_env = query_worker_environment()
        assert worker_env['MIRA_COVERTYPE_QUERY_SNAPSHOT'] == snapshot
        assert 'MIRA_QUERY_ENDPOINT' in worker_env
        assert worker_env['MIRA_HEURISTIC_PRUNING'] == str(int(args.query_processing['heuristic_pruning']))
        barrier = threading.Barrier(3)
        def request(index):
            barrier.wait()
            return submit_query(snapshot, queries[index])
        with ThreadPoolExecutor(max_workers=3) as pool:
            values = list(pool.map(request, [0, 0, 1]))
        np.testing.assert_array_equal(values[0]['indices'], [0])
        np.testing.assert_array_equal(values[2]['indices'], [1])
        submit_query(snapshot, queries[0])
        stats = service_stats()
        assert stats['server_pid'] != os.getpid()
        assert stats['result_reuse'] == args.query_processing['result_reuse']
        assert stats['shared_execution'] == args.query_processing['shared_execution']
        if variant == 'no_result_reuse':
            assert stats['evaluations'] == 4
            assert stats['cache_hits'] == stats['coalesced_requests'] == 0
        else:
            assert stats['evaluations'] == 2
            assert stats['cache_hits'] >= 1
        if variant == 'no_shared_execution':
            assert stats['shared_intermediates'] == 0
        else:
            assert stats['shared_intermediates'] > 0
    report = json.loads((tmp_path/'report.json').read_text())
    assert report == stats


def test_live_covertype_matches_bundled_regions_and_local_rewards(tmp_path):
    from covertype_rl.data import load_covertype
    from covertype_rl.fixed_sets import load_fixed_universe
    from covertype_rl.environment import FixedSetEnvironment
    from revision_support.query_clients import register_covertype_backend
    from revision_support.query_service import query_service_scope
    universe = load_fixed_universe(ROOT/'covertype-exploration/preprocessed/by_distribution_path100k_seed1')
    data = load_covertype(ROOT/'covertype-exploration/covertype.csv', n_bins=10)
    ids = [0, 1, 20, 99, 1000, 13000, 25000, 50000, 75000, 99999]
    offline = FixedSetEnvironment(universe, [])
    offline.reset()
    expected = [offline._metrics(i, valid=True) for i in ids]
    with query_service_scope(tmp_path/'real.json'):
        register_covertype_backend(data)
        live = FixedSetEnvironment(universe, [])
        live.reset()
        for i, reference in zip(ids, expected):
            actual = live._metrics(i, valid=True)
            for key in reference:
                np.testing.assert_allclose(actual[key], reference[key], rtol=1e-6, atol=1e-6,
                                           err_msg=f'region {i}, {key}')


def test_covertype_unpruned_candidates_are_not_truncated(monkeypatch):
    from covertype_rl.candidates import build_candidate_view
    n = 17
    universe = SimpleNamespace(graph=np.array([list(range(1, n+1))]), action_dim=n,
        size_for_set=lambda index: 10+index,
        state_for_set=lambda index: np.array([index], dtype=np.float32))
    actions = [SimpleNamespace(family='by_facet') for _ in range(n)]
    monkeypatch.setenv('MIRA_HEURISTIC_PRUNING', '1')
    pruned, _ = build_candidate_view(universe, actions, 0, candidate_slots=10)
    monkeypatch.setenv('MIRA_HEURISTIC_PRUNING', '0')
    full, _ = build_candidate_view(universe, actions, 0, candidate_slots=10)
    assert len(pruned) == len(full) == n, 'Storage capacity must not depend on the ablation'
    assert len([i for i in pruned if i >= 0]) == 10
    assert set(full) == set(range(1, n+1)), 'No-pruning must retain every valid successor'


@pytest.mark.parametrize('module', ['rl', 'baselines.dora', 'baselines.atena', 'baselines.atena_no_ext'])
def test_galaxy_pruning_is_shared_and_keeps_exact_legality(monkeypatch, module):
    from app.pipelines.dataset import Dataset
    from app.pipelines.predicateitem import PredicateItem
    manager = importlib.import_module(module + '.A3C_2_actors.action_manager').ActionManager
    pipeline = SimpleNamespace(exploration_columns=['a', 'b'], discrete_categories_count=10)
    dataset = Dataset(data=pd.DataFrame(dict(a=[1, 1, 1], b=[1, 2, 3])))
    dataset.predicate.append(PredicateItem('b', '==', 1))
    actions = manager(pipeline, ['by_facet', 'by_neighbors', 'by_superset'])
    facet_a = actions.set_action_types.index('by_facet-&-a')
    facet_b = actions.set_action_types.index('by_facet-&-b')
    monkeypatch.setenv('MIRA_HEURISTIC_PRUNING', '1')
    reduced = actions.fix_possible_operation_action_probs(dataset, np.ones(len(actions.set_action_types)))
    monkeypatch.setenv('MIRA_HEURISTIC_PRUNING', '0')
    complete = actions.fix_possible_operation_action_probs(dataset, np.ones(len(actions.set_action_types)))
    assert reduced[facet_a] == 0, 'Single-valued facets must be excluded only by heuristic pruning'
    assert complete[facet_a] > 0
    assert reduced[facet_b] == complete[facet_b] == 0, 'Exact operator validity must survive the ablation'
