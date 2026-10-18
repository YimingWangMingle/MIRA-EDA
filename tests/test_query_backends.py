import importlib
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ATENA_dataset/ATENA-A-EDA/benchmark'))


def backends():
    spec = importlib.util.find_spec('revision_support.query_backends')
    assert spec is not None, 'Shared pure query backends are missing'
    return importlib.import_module('revision_support.query_backends')


def test_display_backend_shares_filtered_region_but_preserves_distinct_groups():
    from atena.simulation.display import DisplayCalculator
    from atena.simulation.state import EnvStateTuple, AggregationTuple
    from atena.simulation.actions import AggregationFunction
    df = pd.DataFrame(dict(a=['x', 'x', 'y'], b=['p', 'q', 'q'], n=[1., 2., 3.]))
    calculator = DisplayCalculator(SimpleNamespace(dataset_df=df, columns=list(df), numeric_columns=['n']))
    states = [EnvStateTuple.create_empty_state().append_grouping(column).append_aggregations(
        AggregationTuple('n', AggregationFunction.COUNT)) for column in ['a', 'b']]
    expected = [calculator.calculate_display(state) for state in states]
    backend = backends().DisplayBackend(calculator)
    values, shared = backend.evaluate_many(states)
    assert shared == 1
    for actual, reference in zip(values, expected):
        assert actual[0] == reference[0]
        for got, want in zip(actual[1], reference[1]):
            pd.testing.assert_frame_equal(got, want)
    assert not values[0][1][1].equals(values[1][1][1])


def test_galaxy_predicate_batch_preserves_input_scope_and_reuses_prefix():
    from app.pipelines.pipeline import Pipeline
    from app.pipelines.dataset import Dataset
    from app.pipelines.predicateitem import PredicateItem, PredicateItemGroup
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.initial_collection = pd.DataFrame(dict(a=[1, 1, 2, 1], b=[1, 2, 2, 2]))
    pipeline.ordered_dimensions = {}
    queries = []
    for value in (1, 2):
        predicate = PredicateItemGroup()
        predicate.append(PredicateItem('a', '==', 1))
        predicate.append(PredicateItem('b', '==', value))
        queries.append(Dataset(data=pipeline.initial_collection.iloc[:3].copy(), predicate=predicate))
    expected = [pipeline.query_predicate(item.copy(), item.predicate).copy() for item in queries]
    batch = backends().GalaxyPredicateBatch()
    actual = [batch.filter(pipeline, item.data, item.predicate) for item in queries]
    for got, want in zip(actual, expected):
        pd.testing.assert_frame_equal(got, want)
    assert batch.shared_intermediates >= 1
    assert list(actual[1].index) == [1]


def test_static_summaries_do_not_survive_changed_data():
    from app.pipelines.dataset import Dataset
    module = backends()
    dataset = Dataset(data=pd.DataFrame(dict(a=pd.Categorical(['x', 'x', 'y']))))
    original = module.galaxy_summary(dataset, ['a'])
    assert original['a']['distinct'] == 2
    dataset.data = dataset.data.iloc[:1].copy()
    current = module.galaxy_summary(dataset, ['a'])
    assert current['a']['distinct'] == 1
    assert current['a']['entropy'] == 0
    assert current['a']['counts'].sum() == 1
