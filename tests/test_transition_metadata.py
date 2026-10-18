from types import SimpleNamespace
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'ATENA_dataset/ATENA-A-EDA/benchmark'))


def test_atena_identity_distinguishes_filter_history_after_back():
    from atena.simulation.actions import FilterAction, FilterOperator, BackAction
    from atena.simulation.state import EnvStateTuple, FilteringTuple
    from revision_support.transition_metadata import atena_reference_identity
    empty = EnvStateTuple.create_empty_state()
    def env(term):
        filtered = empty.append_filtering(FilteringTuple('a', term, FilterOperator.EQUAL))
        simulation = SimpleNamespace(states_stack=[empty], states_history=[empty, filtered, empty])
        return SimpleNamespace(schema_name='flights', dataset_name=1, step_index=2,
            simulator=SimpleNamespace(simulation_state=simulation),
            actions=[FilterAction('a', FilterOperator.EQUAL, term), BackAction()])
    left, right = env('x'), env('y')
    assert atena_reference_identity(left) != atena_reference_identity(right)
    assert atena_reference_identity(left) == atena_reference_identity(deepcopy(left))


def test_galaxy_identity_keeps_multiline_predicates_and_metadata_is_read_only():
    from app.pipelines.dataset import Dataset
    from app.pipelines.predicateitem import PredicateItem, PredicateItemGroup
    from revision_support.transition_metadata import dual_reference_metadata
    def env(value):
        predicate = PredicateItemGroup()
        predicate.append(PredicateItem('text', '==', value))
        dataset = Dataset(predicate=predicate)
        return SimpleNamespace(datasets=[dataset], input_set=dataset, step_count=3,
            set_action_space=SimpleNamespace(n=10), operation_action_space=SimpleNamespace(n=2),
            action_manager=SimpleNamespace(fix_possible_operation_action_probs=lambda d, p: [1, 0]),
            state_encoder=SimpleNamespace(found_items_with_ratio={}))
    left, right = env('a\nx'), env('a\ny')
    mask, key = dual_reference_metadata(left)
    assert key != dual_reference_metadata(right)[1]
    assert mask.shape == (20,)
    np.testing.assert_array_equal(np.flatnonzero(mask), [0])
    assert left.step_count == 3
    assert left.state_encoder.found_items_with_ratio == {}


def test_covertype_dead_end_matches_existing_noop_fallback_until_horizon():
    from revision_support.transition_metadata import dual_reference_metadata
    env = SimpleNamespace(set_action_dim=2, operation_action_dim=3,
        candidate_action_ids=[[], []], candidate_set_ids=[-1, -1], step_count=4,
        base_env=SimpleNamespace(current_set_id=7, found_items_with_ratio={},
                                 valid_actions=lambda: np.arange(3)))
    masks, _ = dual_reference_metadata(env)


    assert masks.all()
