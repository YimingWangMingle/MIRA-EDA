import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def reference_api():
    from revision_support import reference_policy
    return reference_policy


def test_wasserstein_uses_optimal_matching_not_independent_pairs():
    api = reference_api()
    points = np.array([[[0.], [10.]], [[10.], [0.]], [[2.], [12.]]])
    np.testing.assert_allclose(api.empirical_wasserstein(points, [1, 2, 0]).ravel(), [0., 2., 2.])


def test_uniform_reference_reuses_complete_state_samples_and_masks_invalid_actions():
    api = reference_api()
    masks = np.array([[1, 0, 1, 1], [1, 0, 1, 1], [0, 0, 0, 0]], bool)
    actions = api.sample_reference_actions(masks, [b'a', b'a', b't'], [False, False, True],
                                         12000, np.random.default_rng(7))
    np.testing.assert_array_equal(actions[0], actions[1])
    assert np.all(actions[2] == -1)
    counts = np.bincount(actions[0], minlength=4)
    assert counts[1] == 0
    assert np.max(np.abs(counts[[0, 2, 3]] - 4000)) < 180


def test_reference_rejects_empty_nonterminal_and_inconsistent_state_identity():
    api = reference_api()
    with pytest.raises(ValueError, match='nonterminal'):
        api.sample_reference_actions([[0, 0]], [b'a'], [False], 4, np.random.default_rng(0))
    with pytest.raises(ValueError, match='identity'):
        api.sample_reference_actions([[1, 0], [0, 1]], [b'a', b'a'], [False, False],
                                     4, np.random.default_rng(0))


def test_absorbing_point_mass_has_zero_self_distance():
    api = reference_api()
    points = np.array([[[3.], [3.], [3.], [3.]], [[1.], [2.], [4.], [5.]]])
    np.testing.assert_allclose(api.empirical_wasserstein(points, [0, 0]).ravel(), [0., 1.5])
