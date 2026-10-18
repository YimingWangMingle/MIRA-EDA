import ast
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_all_formal_methods_receive_the_same_training_and_target_seeds():
    from verify import load_runner, parser_from_source
    for job in load_runner(ROOT).build_jobs():
        args = parser_from_source(ROOT, job['dataset'], job['method']).parse_args(job['command'][2:])
        assert args.seed == job['seed']
        assert args.target_set == f"fixed_seed_{job['seed']}"
        assert (args.episodes, args.steps, args.workers) == (1000, 250, 14)


@pytest.mark.parametrize('module', ['rl', 'baselines.pure_a3c', 'baselines.dora', 'baselines.atena', 'baselines.atena_no_ext'])
def test_galaxy_model_and_worker_seed_before_initialization(module):
    source = ROOT / (module.replace('.', '/') + '/A3C_2_actors/A3C.py')
    tree = ast.parse(source.read_text(encoding='utf-8'))
    for name in ('ParameterServer', 'WorkerAgent'):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        calls = [n for n in ast.walk(init) if isinstance(n, ast.Call)]
        seeding = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'seed_training']
        assert len(seeding) == 1
        models = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id in ('SetActor', 'PipelineEnvironment')]
        assert seeding[0].lineno < min(n.lineno for n in models)
        assert len(seeding[0].args) == (1 if name == 'ParameterServer' else 2)


def test_training_seed_controls_python_numpy_and_tensorflow():
    import random
    import tensorflow as tf
    from revision_support.training_runtime import seed_training
    def draw():
        return random.random(), np.random.random(), float(tf.random.uniform(()))
    seed_training(7, 3)
    first = draw()
    seed_training(7, 3)
    assert draw() == first
    seed_training(7, 4)
    assert draw() != first


@pytest.mark.parametrize('module', ['baselines.dora', 'baselines.atena', 'baselines.atena_no_ext'])
def test_galaxy_baselines_share_observations_masks_and_task_reward(module):
    from app.pipelines.dataset import Dataset
    from app.pipelines.predicateitem import PredicateItem
    from rl.A3C_2_actors.pipeline_environment import PipelineEnvironment
    other = importlib.import_module(module + '.A3C_2_actors.pipeline_environment').PipelineEnvironment
    data = pd.DataFrame({'x': ['0', '0', '1'], 'galaxies.objID': [1, 2, 3]})
    pipeline = SimpleNamespace(exploration_columns=['x'], ordered_dimensions={'x': ['0', '1']},
        discrete_categories_count=10, initial_collection=data)
    root = Dataset(data=data)
    pipeline.get_dataset = lambda: root.copy()
    kwargs = dict(target_items=[1, 3], operators=['by_facet', 'by_superset', 'by_neighbors', 'by_distribution'], episode_steps=12)
    left, right = PipelineEnvironment(pipeline, **kwargs), other(pipeline, **kwargs)
    np.testing.assert_array_equal(left.reset(), right.reset())
    child = Dataset(data=data.iloc[:2].copy())
    child.predicate.append(PredicateItem('x', '==', '0'))
    for sample in (child, child, root):
        left.datasets, right.datasets = [sample.copy()], [sample.copy()]
        a, b = left.get_set_state(), right.get_set_state()
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y)
        np.testing.assert_array_equal(left.fix_possible_operation_action_probs(0, np.ones(4)),
                                      right.fix_possible_operation_action_probs(0, np.ones(4)))
