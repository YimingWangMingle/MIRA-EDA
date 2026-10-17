import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input

from .inference import predict
from .representations import RepresentationVariants

CORE_VERSION = 'mira_grounded_latent_v2'


def finite_states(values, dimension):
    result = np.asarray(values, dtype=np.float32).reshape((-1, int(dimension)))
    if not np.isfinite(result).all():
        raise ValueError('MIRA descriptor contains nonfinite values')
    return result


def normalize_direction(direction, latent_dim):
    value = np.asarray(direction, dtype=np.float32).reshape(-1)
    if value.size != int(latent_dim) or not np.isfinite(value).all():
        raise ValueError('Invalid MIRA direction')
    norm = float(np.linalg.norm(value))
    if norm <= 0.:
        raise ValueError('MIRA direction must be nonzero')
    return value / norm


def sample_direction(rng, latent_dim):
    return normalize_direction(rng.normal(size=int(latent_dim)), latent_dim)


class RunningMoments:

    def __init__(self, state=None):
        state = state or {}
        self.count = int(state.get('count', 0))
        self.mean = float(state.get('mean', 0.))
        self.m2 = float(state.get('m2', 0.))

    @property
    def variance(self):
        return max(0., self.m2 / self.count) if self.count > 1 else 1.

    def state(self):
        return dict(count=self.count, mean=self.mean, m2=self.m2)

    def update(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite directional reward')
        if values.size:
            mean = float(values.mean())
            self.merge(dict(count=len(values), mean=mean, m2=float(np.sum((values-mean)**2))))

    def merge(self, state):
        n = int(state['count'])
        if not n:
            return
        delta = float(state['mean']) - self.mean
        total = self.count + n
        self.m2 += float(state['m2']) + delta*delta*self.count*n/total
        self.mean += delta*n/total
        self.count = total

    def normalize(self, value, clip):
        result = (float(value)-self.mean)/(np.sqrt(self.variance)+1e-6)
        return float(np.clip(result, -clip, clip)) if clip > 0 else float(result)


def grounded_target(rewards, statistics, pairs, successor_distance, gamma, lambda_ana):
    rewards = tf.reshape(tf.convert_to_tensor(rewards, tf.float32), (-1, 1))
    statistics = tf.convert_to_tensor(statistics, tf.float32)
    pairs = tf.convert_to_tensor(pairs, tf.int32)
    reward_difference = tf.abs(rewards-tf.gather(rewards, pairs))
    analytical = tf.norm(statistics-tf.gather(statistics, pairs), axis=1, keepdims=True)
    target = reward_difference + float(lambda_ana)*analytical + float(gamma)*successor_distance
    return tf.stop_gradient(target), dict(reward_difference=reward_difference, analytical=analytical)


class MIRACore(RepresentationVariants):
    def __init__(self, state_dim, action_dim, latent_dim=16, hidden=256, lr=3e-4,
                 target_tau=.01, statistic_indices=None, lambda_ana=1., mc_samples=4,
                 encoder_variant='learned', encoder_pca_path=None, encoder_vae_kl=.001):
        self.state_dim, self.action_dim = int(state_dim), int(action_dim)
        self.latent_dim, self.dense_units = int(latent_dim), int(hidden)
        self.target_tau = float(target_tau)
        self.lambda_ana, self.mc_samples = float(lambda_ana), int(mc_samples)
        if not np.isfinite(self.lambda_ana) or self.lambda_ana <= 0 or self.mc_samples < 1:
            raise ValueError('MIRA requires lambda_ana > 0 and mc_samples >= 1')
        if not np.isfinite(self.target_tau) or not 0. < self.target_tau <= 1.:
            raise ValueError('MIRA requires 0 < target_tau <= 1')
        if statistic_indices is None or len(statistic_indices) == 0:
            raise ValueError('MIRA requires explicit label-free statistic_indices')
        self.statistic_indices = np.asarray(statistic_indices, dtype=np.int32)
        if np.any(self.statistic_indices < 0) or np.any(self.statistic_indices >= self.state_dim):
            raise ValueError('Statistic index outside descriptor')
        self.encoder = self._create_encoder()
        self.target_encoder = self._create_encoder()
        self.target_encoder.set_weights(self.encoder.get_weights())
        self.configure_representation(encoder_variant, encoder_pca_path, encoder_vae_kl, hidden)
        self.dynamics = tf.keras.Sequential([
            Input((self.latent_dim+self.action_dim,), dtype=tf.float32),
            Dense(hidden, activation='relu', dtype=tf.float32),
            Dense(hidden, activation='relu', dtype=tf.float32),
            Dense(2*self.latent_dim, dtype=tf.float32)])
        self.encoder_opt = tf.keras.optimizers.Adam(float(lr))
        self.dynamics_opt = tf.keras.optimizers.Adam(float(lr))
        self.reward_moments = RunningMoments()
        self.episode_moments = RunningMoments()
        self.reward_encoder = None
        self.version = 0

    def _create_encoder(self):
        return tf.keras.Sequential([Input((self.state_dim,), dtype=tf.float32),
            Dense(self.dense_units, activation='relu', dtype=tf.float32),
            Dense(self.dense_units, activation='relu', dtype=tf.float32),
            Dense(self.latent_dim, dtype=tf.float32)])

    def analytical_summary(self, states):
        return finite_states(states, self.state_dim)[:, self.statistic_indices]

    def embed(self, states):
        return predict(self.encoder, finite_states(states, self.state_dim))

    def begin_episode(self, moments=None):

        if self.reward_encoder is None:
            self.reward_encoder = tf.keras.models.clone_model(self.target_encoder)
            self.reward_encoder.trainable = False
        self.reward_encoder.set_weights(self.target_encoder.get_weights())
        self.reward_encoder_version = self.version
        if moments is not None:
            self.reward_moments = RunningMoments(moments)
        self.episode_moments = RunningMoments()

    def reward_embed(self, states):
        if self.reward_encoder is None:
            self.begin_episode()
        return predict(self.reward_encoder, finite_states(states, self.state_dim))

    def raw_bonus(self, state, next_state, direction):
        embedding = self.reward_embed([state, next_state])
        delta = embedding[1]-embedding[0]
        return float(np.dot(delta, normalize_direction(direction, self.latent_dim)))

    def compute_bonus(self, state, next_state, direction, clip_value=1.):
        raw = self.raw_bonus(state, next_state, direction)
        normalized = self.reward_moments.normalize(raw, float(clip_value))
        self.reward_moments.update([raw])
        self.episode_moments.update([raw])
        self.last_raw_bonus, self.last_normalized_bonus = raw, normalized
        return normalized

    def distribution(self, embedding, actions, training=False):
        output = self.dynamics(tf.concat([embedding, actions], axis=1), training=training)
        mean, log_std = tf.split(output, 2, axis=1)
        return mean, tf.clip_by_value(log_std, -5., 2.)

    def successor_discrepancy(self, mean, log_std, pairs, samples=None):
        count = int(samples or self.mc_samples)
        shape = tf.concat([[count], tf.shape(mean)], axis=0)
        left = mean[None] + tf.exp(log_std)[None]*tf.random.normal(shape)

        right = tf.gather(mean, pairs)[None] + tf.exp(tf.gather(log_std, pairs))[None]*tf.random.normal(shape)
        return tf.reduce_mean(tf.norm(left-right, axis=2), axis=0)[:, None]

    def train_batch(self, states, action_features, task_rewards, next_states,
                    pair_indices=None, gamma=.99, phi_weight=1., dyn_weight=1.):
        states = tf.convert_to_tensor(finite_states(states, self.state_dim))
        following = tf.convert_to_tensor(finite_states(next_states, self.state_dim))
        actions = tf.convert_to_tensor(action_features, tf.float32)
        n = int(states.shape[0])
        if n < 2:
            raise ValueError('Representation learner requires at least two transitions')
        pairs = (tf.random.shuffle(tf.range(n)) if pair_indices is None
                 else tf.convert_to_tensor(pair_indices, tf.int32))
        current_latent = tf.stop_gradient(self.encoder(states, training=False))
        next_latent = tf.stop_gradient(self.encoder(following, training=False))
        with tf.GradientTape() as tape:
            mean, log_std = self.distribution(current_latent, actions, training=True)
            nll = .5*tf.square((next_latent-mean)*tf.exp(-log_std)) + log_std + .5*np.log(2*np.pi)
            dyn_loss = tf.reduce_mean(nll)
            dyn_objective = float(dyn_weight)*dyn_loss
        grads = tape.gradient(dyn_objective, self.dynamics.trainable_variables)
        self._apply(self.dynamics_opt, grads, self.dynamics.trainable_variables)

        mean, log_std = self.distribution(current_latent, actions)
        successor = self.successor_discrepancy(mean, log_std, pairs)
        statistics = tf.gather(states, self.statistic_indices, axis=1)
        target, components = grounded_target(task_rewards, statistics, pairs, successor, gamma, self.lambda_ana)
        with tf.GradientTape() as tape:
            phi = self.encoder(states, training=True)
            squared = tf.reduce_sum(tf.square(phi-tf.gather(phi, pairs)), axis=1, keepdims=True)
            distance = tf.sqrt(squared+1e-12)-1e-6
            metric_loss = tf.reduce_mean(tf.square(distance-target))
            phi_loss = self.representation_loss(states, phi, metric_loss)
            objective = float(phi_weight)*phi_loss
        if not self.representation_frozen:
            grads = tape.gradient(objective, self.representation_variables)
            self._apply(self.encoder_opt, grads, self.representation_variables)
        self.soft_update_target()
        self.version += 1
        return dict(phi_loss=float(phi_loss.numpy()), dyn_loss=float(dyn_loss.numpy()),
            prediction_error=float(tf.reduce_mean(tf.norm(next_latent-mean, axis=1)).numpy()),
            mean_d_phi=float(tf.reduce_mean(distance).numpy()), mean_metric_target=float(tf.reduce_mean(target).numpy()),
            analytical_distance=float(tf.reduce_mean(components['analytical']).numpy()),
            task_reward_difference=float(tf.reduce_mean(components['reward_difference']).numpy()),
            successor_distance=float(tf.reduce_mean(successor).numpy()), representation_updates=self.version)

    @staticmethod
    def _apply(optimizer, gradients, variables):
        pairs = [(g, v) for g, v in zip(gradients, variables) if g is not None]
        for gradient, _ in pairs:
            tf.debugging.assert_all_finite(gradient, 'Nonfinite MIRA gradient')
        if pairs:
            clipped, _ = tf.clip_by_global_norm([g for g, _ in pairs], 40.)
            optimizer.apply_gradients(zip(clipped, [v for _, v in pairs]))

    def soft_update_target(self):
        if not self.representation_frozen:
            tau = self.target_tau
            self.target_encoder.set_weights([tau*a+(1-tau)*b for a, b in
                zip(self.encoder.get_weights(), self.target_encoder.get_weights())])

    def get_weights(self):
        return dict(self.representation_metadata(), core_version=CORE_VERSION,
            encoder=self.encoder.get_weights(), target_encoder=self.target_encoder.get_weights(),
            dynamics=self.dynamics.get_weights(), version=self.version,
            statistic_indices=self.statistic_indices.tolist(), lambda_ana=self.lambda_ana, mc_samples=self.mc_samples)

    def set_weights(self, weights):
        if weights.get('core_version') != CORE_VERSION:
            raise ValueError('MIRA checkpoint version differs; old models require retraining')
        if list(weights['statistic_indices']) != self.statistic_indices.tolist():
            raise ValueError('MIRA descriptor schema differs')
        if weights['lambda_ana'] != self.lambda_ana or weights['mc_samples'] != self.mc_samples:
            raise ValueError('MIRA core configuration differs')
        self.restore_representation_metadata(weights)
        self.encoder.set_weights(weights['encoder'])
        self.target_encoder.set_weights(weights['target_encoder'])
        self.dynamics.set_weights(weights['dynamics'])
        self.version = int(weights['version'])

    def save_models(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.save_representation_metadata(directory)
        self.encoder.save(directory/'mira_encoder')
        self.target_encoder.save(directory/'mira_target_encoder')
        self.dynamics.save(directory/'mira_dynamics')
        (directory/'mira_core.json').write_text(json.dumps(dict(core_version=CORE_VERSION,
            statistic_indices=self.statistic_indices.tolist(), lambda_ana=self.lambda_ana,
            mc_samples=self.mc_samples, log_std_bounds=[-5., 2.], version=self.version,
            reward_normalization='running population moments, epsilon=1e-6; normalize before update',
            reward_moments=self.reward_moments.state()), indent=2), encoding='utf8')
