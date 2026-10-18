import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revision_support.training_runtime import configure_training_threads

configure_training_threads()


def module(task_feedback=True, variant='learned'):
    from revision_support.mira_dual import DualActionMIRA
    return DualActionMIRA(3, 2, 2, latent_dim=2, dense_units=8,
                          statistic_indices=[0, 1], task_reward_enabled=task_feedback,
                          encoder_variant=variant)


def test_reference_predictions_share_actions_for_reward_and_dynamics():
    import tensorflow as tf
    m = module()
    states = np.array([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.]], np.float32)
    masks = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]], bool)
    m.dynamics = lambda x, training=False: x[:, -2:]
    m.reward_predictor = lambda x, training=False: tf.reduce_sum(x[:, -2:] * [2., 7.], axis=1, keepdims=True)
    predictions, rewards, ids = m.reference_predictions(states, masks, [b'a', b'a', b't'],
                                                        [False, False, True], np.random.default_rng(0))
    np.testing.assert_array_equal(ids[:2], np.ones((2, 4), dtype=int))
    np.testing.assert_allclose(predictions[:2], np.tile([0., 1.], (2, 4, 1)))
    np.testing.assert_allclose(rewards.ravel(), [7., 7., 0.])
    np.testing.assert_allclose(predictions[2], np.tile(m.embed(states[2:])[0], (4, 1)), atol=1e-6)


def test_reward_free_reference_is_exactly_zero():
    m = module(False)
    assert m.reward_predictor is None
    _, rewards, _ = m.reference_predictions(np.ones((2, 3)), np.ones((2, 4)),
                                           [b'a', b'b'], [False, False], np.random.default_rng(1))
    np.testing.assert_array_equal(rewards, np.zeros((2, 1)))


def test_reference_identity_rejects_different_descriptors():
    m = module()
    with pytest.raises(ValueError, match='descriptor'):
        m.reference_predictions([[1, 0, 0], [0, 1, 0]], np.ones((2, 4)),
                                [b'same', b'same'], [False, False])


@pytest.mark.parametrize('variant', ['learned', 'frozen_random', 'vae'])
def test_training_updates_auxiliary_models_and_preserves_episode_reward_encoder(variant):
    m = module(variant=variant)
    states = np.random.default_rng(5).normal(size=(8, 3)).astype(np.float32)
    m.begin_episode()
    frozen = m.reward_embed(states).copy()
    before = m.get_weights()
    stats = m.train_batch(states, m.action_features([0] * 8, [1] * 8),
                          np.ones(8), states[::-1].copy(), valid_action_masks=np.ones((8, 4)),
                          state_keys=[str(i) for i in range(8)], terminals=np.zeros(8, bool))
    assert all(np.isfinite(v) for v in stats.values())
    assert stats['reward_prediction_loss'] > 0
    for key in ['dynamics', 'reward_predictor']:
        assert any(not np.array_equal(a, b) for a, b in zip(before[key], m.get_weights()[key]))
    if variant == 'frozen_random':
        for a, b in zip(before['encoder'], m.get_weights()['encoder']):
            np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(m.reward_embed(states), frozen)


def test_auxiliary_losses_do_not_train_encoder_and_checkpoint_roundtrip():
    m = module()
    states = np.random.default_rng(4).normal(size=(8, 3)).astype(np.float32)
    before = m.get_weights()['encoder']
    m.train_batch(states, m.action_features([0] * 8, [0] * 8), np.ones(8), -states,
                  phi_weight=0., valid_action_masks=np.ones((8, 4)))
    for a, b in zip(before, m.get_weights()['encoder']):
        np.testing.assert_array_equal(a, b)
    other = module()
    other.set_weights(m.get_weights())
    np.testing.assert_array_equal(m.embed(states), other.embed(states))
    with pytest.raises(ValueError, match='version'):
        other.set_weights(dict(m.get_weights(), core_version='mira_grounded_latent_v2'))


def test_replay_records_absorbing_successor_without_erasing_final_transition_reward():
    from revision_support.mira_learner import RepresentationLearner
    m = module()
    learner = RepresentationLearner(m, capacity=16, batch_size=8, min_size=16)
    try:
        learner.submit([[1., 0., 0.]], [[0, 1]], [6.], [[0., 1., 0.]],
                       valid_action_masks=[[0, 1, 0, 0]], state_keys=[b's'],
                       next_terminals=[True], next_state_keys=[b't'])
        learner.queue.join()
        learner.check()
        assert learner.size == 2
        np.testing.assert_array_equal(learner.arrays[2][:2].ravel(), [6., 0.])
        np.testing.assert_array_equal(learner.arrays[5][:2], [False, True])
        assert not learner.arrays[4][1].any()
    finally:
        learner.close()
