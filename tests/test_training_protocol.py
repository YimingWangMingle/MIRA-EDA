import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'covertype-exploration', ROOT / 'ATENA_dataset',
                  ROOT / 'ATENA_dataset/MIRA', ROOT / 'ATENA_dataset/scripts'):
    sys.path.insert(0, str(directory))

from revision_support.training_runtime import configure_training_threads

configure_training_threads()


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verification_uses_package_sources(tmp_path):
    package = tmp_path / 'package'
    package.mkdir()
    (package / 'entry.py').write_text('value = 1\n', encoding='utf-8')
    dependency = package / '.venv/lib/site-packages/dependency.py'
    dependency.parent.mkdir(parents=True)
    dependency.write_text('requires another Python version\n', encoding='utf-8')
    verifier = load_file('source_verifier_test', 'verify.py')
    assert verifier.verify_source(package)['python_sources'] == 1


def worker_kwargs(relative, environment, args, extra):
    tree = ast.parse((ROOT / relative).read_text(encoding='utf-8'))
    worker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WorkerAgent')
    constructor = next(n for n in ast.walk(worker) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Name) and n.func.id == environment)
    scope = dict(args=args, self=SimpleNamespace(args=args, episode_steps=12, agent_id=0),
                 MIRA_BASELINES={'mira', 'mira_no_ext'}, episode_steps=12, agentId=0)
    scope.update(extra)
    return {k.arg: eval(compile(ast.Expression(k.value), '<argument>', 'eval'), scope)
            for k in constructor.keywords}


def test_galaxy_worker_uses_shared_observation():
    from rl.A3C_2_actors.pipeline_environment import PipelineEnvironment
    pipeline = SimpleNamespace(exploration_columns=['x'],
        ordered_dimensions={'x': ['0', '1']}, discrete_categories_count=10,
        initial_collection=pd.DataFrame({'x': ['0', '1'], 'galaxies.objID': [1, 2]}))
    dataset = SimpleNamespace(data=pipeline.initial_collection, set_id=None,
                              predicate=SimpleNamespace(components=[]))
    pipeline.get_dataset = lambda: dataset
    args = SimpleNamespace(mode='scattered', operators=['by_facet'], target_set=None,
                           target_seed=1, target_samples_per_file=100)
    kwargs = worker_kwargs('rl/A3C_2_actors/A3C.py', 'PipelineEnvironment', args, {})
    kwargs['target_items'] = [1]
    mira = PipelineEnvironment(pipeline, **kwargs)
    baseline = PipelineEnvironment(pipeline, target_items=[1], operators=args.operators, episode_steps=12)
    np.testing.assert_array_equal(mira.reset(), baseline.reset())
    mira.datasets = baseline.datasets = [dataset]
    np.testing.assert_array_equal(mira.get_set_state()[0], baseline.get_set_state()[0])
    assert mira.set_state_dim == baseline.set_state_dim == 40
    assert len(mira.statistic_indices) == 30
    assert max(mira.statistic_indices) < mira.set_state_dim


@pytest.mark.parametrize('method', ['mira', 'mira_no_ext'])
def test_covertype_worker_uses_shared_candidates_and_observation(method):
    from covertype_rl.full_a3c import CovertypeDualActorEnvironment
    from covertype_rl.actions import build_action_space
    from covertype_rl.fixed_sets import load_fixed_universe
    from covertype_rl.candidates import build_candidate_view
    universe = load_fixed_universe(ROOT / 'covertype-exploration/preprocessed/by_distribution_path100k_seed1')
    actions = build_action_space(n_continuous=10, n_bins=10)
    args = SimpleNamespace(baseline=method, steps=12, seed=1, candidate_slots=10)
    kwargs = worker_kwargs('covertype-exploration/covertype_rl/full_a3c.py',
                          'CovertypeDualActorEnvironment', args, {'agent_id': 0})
    env = CovertypeDualActorEnvironment(universe, actions, **kwargs)
    state = env.reset()
    assert state.shape == (840,)
    for step in range(12):
        expected, action_ids = build_candidate_view(universe, actions, env.base_env.current_set_id)
        assert env.candidate_set_ids == expected
        np.testing.assert_array_equal(state, np.concatenate([
            universe.state_for_set(i) if i >= 0 else np.zeros(84, np.float32) for i in expected]))
        slot = next(i for i, ids in enumerate(action_ids) if ids)
        state, _, _, done, _, _ = env.step(slot, action_ids[slot][0])
        assert done == (step == 11)
    assert 0 < len(env.statistic_indices) < env.set_state_dim


@pytest.mark.parametrize('schema', ['cyber', 'flights'])
@pytest.mark.parametrize('dataset', [1, 2, 3, 4])
def test_atena_observations_actions_and_task_rewards_match(schema, dataset):
    from mira.env import make_env as mira_env
    from atena_baselines.env import make_env as baseline_env
    cli = load_file('mira_cli_test', 'ATENA_dataset/MIRA/run.py')
    baseline_cli = load_file('baseline_cli_test', 'ATENA_dataset/run_atena_baselines.py')
    args = cli.parse_args(['--schema', schema, '--dataset_number', str(dataset)])
    other = baseline_cli.parse_args(['--method', 'pure_a3c', '--schema', schema, '--dataset_number', str(dataset)])
    left, right = mira_env(schema, dataset, 0, args), baseline_env(schema, dataset, 0, other)
    assert left.state_dim == right.state_dim
    assert [repr(a) for a in left.action_space.actions] == [repr(a) for a in right.action_space.actions]
    np.testing.assert_array_equal(left.reset(), right.reset())
    for step in range(12):
        legal = np.flatnonzero(right.legal_action_mask())
        action = int(legal[(step * 17 + 5) % len(legal)])
        a, b = left.step(action), right.step(action)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1] == b[1]
        assert a[2] == b[2]
    assert max(left.statistic_indices) < left.state_dim


@pytest.mark.parametrize('method, expected', [('mira', 6.25), ('mira_no_ext', 0.25)])
def test_covertype_policy_reward_uses_external_and_directional_terms(method, expected):
    from covertype_rl.full_a3c import _training_reward
    args = SimpleNamespace(w_ext=2., alpha=.5, w_mira=1., w_int=999.)
    assert _training_reward(method, 3., 12345., 0., 0., 0., .5, args) == expected


@pytest.mark.parametrize('method, expected', [('mira', [6., 8.]), ('mira_no_ext', [0., 0.])])
def test_representation_reward_matches_external_task(method, expected):
    from revision_support.mira_task import replay_task_reward
    args = SimpleNamespace(baseline=method, w_ext=2., alpha=.5, w_int=.2)
    actual = replay_task_reward(np.array([3., 4.]), args)
    np.testing.assert_array_equal(actual.reshape(-1), expected)


def test_atena_final_policy_matches_last_update_and_reconstructs(tmp_path, monkeypatch):
    from mira import train
    from mira.models import PolicyValueNet
    from mira.env import make_env
    from recalculate_baseline_corpus_metrics.runtime import reconstruct_mira_session
    cli = load_file('mira_train_cli_test', 'ATENA_dataset/MIRA/run.py')
    args = cli.parse_args(['--schema', 'cyber', '--dataset_number', '1', '--workers', '1',
        '--steps', '24', '--eval_interval', '1', '--hidden', '8', '--mira_hidden', '8',
        '--mira_latent_dim', '4', '--mira_min_replay_size', '2', '--mira_phi_batch_size', '8',
        '--output_dir', str(tmp_path / 'runs')])
    snapshots = []
    original = train._a3c_update

    def capture(model, optimizer, batch, runtime_args):
        result = original(model, optimizer, batch, runtime_args)
        snapshots.append([w.copy() for w in model.get_weights()])
        return result

    monkeypatch.setattr(train, '_a3c_update', capture)
    train.run_experiment(args)
    directory = tmp_path / 'runs/MIRA/cyber1/seed0'
    env = make_env('cyber', 1, 777, args)
    restored = PolicyValueNet(env.state_dim, env.action_dim, hidden=8, latent_dim=4)
    restored.load_weights(str(directory / 'policy.weights.h5'))
    assert len(snapshots) == 2
    for expected, actual in zip(snapshots[-1], restored.get_weights()):
        np.testing.assert_array_equal(actual, expected)
    actions = reconstruct_mira_session(directory)
    assert [repr(a) for a in actions] == json.loads((directory / 'actions_steps24.json').read_text())
    report = json.loads((directory / 'representation_training.json').read_text())
    assert report['representation_updates'] > 0
