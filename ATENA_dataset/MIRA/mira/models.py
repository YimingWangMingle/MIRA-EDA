import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input


def masked_probs(probs, mask):
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=np.float64).reshape(-1)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.maximum(probs, 0.0) * mask
    total = float(probs.sum())
    if total <= 0.0:
        valid = np.flatnonzero(mask > 0)
        fixed = np.zeros_like(probs, dtype=np.float64)
        if valid.size == 0:
            fixed[:] = 1.0 / max(len(fixed), 1)
        else:
            fixed[valid] = 1.0 / valid.size
        return fixed
    return probs / total


class PolicyValueNet(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden=256, latent_dim=0):
        super().__init__()
        self.direction_dim = int(latent_dim)
        self.d1 = Dense(hidden, activation="relu")
        self.d2 = Dense(hidden, activation="relu")
        self.policy = Dense(action_dim, activation="softmax")
        self.value = Dense(1, activation="linear")
        self.build((None, int(state_dim)+self.direction_dim))

    def call(self, states, training=False):
        x = self.d1(states)
        x = self.d2(x)
        return self.policy(x), self.value(x)


from revision_support.mira_core import MIRACore, sample_direction, normalize_direction


class MiraMetricModule(MIRACore):
    def __init__(self, state_dim, action_dim, latent_dim=16, hidden=256, lr=3e-4,
                 tau=.01, statistic_indices=None, lambda_ana=1., mc_samples=4):
        super().__init__(state_dim,action_dim,latent_dim,hidden,lr,tau,
                         statistic_indices,lambda_ana,mc_samples)

    def encode_actions(self, actions):
        actions = np.asarray(actions,dtype=np.int64).reshape(-1)
        if np.any(actions<0) or np.any(actions>=self.action_dim):
            raise ValueError('ATENA replay action outside schema')
        return np.eye(self.action_dim,dtype=np.float32)[actions]
