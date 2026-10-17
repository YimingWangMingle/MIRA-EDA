import json
from pathlib import Path
import numpy as np
from .rnd import RandomNetworkDistillation


def select_bonus(kind, directional, replacement):

    return directional() if kind == 'directional' else replacement()


class ExplorationReward:
    def __init__(self, kind, state_dim, seed=0, lr=3e-4, knn_k=5, knn_capacity=2048):
        if kind not in ('rnd', 'count', 'knn', 'random'):
            raise ValueError('Expected rnd, count, knn or random')
        self.kind, self.state_dim = kind, int(state_dim)
        self.knn_k, self.knn_capacity = int(knn_k), int(knn_capacity)
        if kind == 'knn' and not 1 <= self.knn_k <= self.knn_capacity:
            raise ValueError('kNN requires 1 <= k <= capacity')
        self.knn_states = np.empty((0, self.state_dim), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.counts = {}
        self.rnd = RandomNetworkDistillation(state_dim, seed=seed, lr=lr) if kind == 'rnd' else None

    def observe(self, states, embed=None):
        states = np.asarray(states, dtype=np.float32).reshape((-1, self.state_dim))
        if not len(states) or not np.isfinite(states).all():
            raise ValueError('Reward replacement requires finite observations')
        if self.rnd is not None:
            return self.rnd.observe(states)
        if self.kind == 'random':
            return self.rng.uniform(-1., 1., len(states)).astype(np.float32)
        if self.kind == 'knn':
            if not callable(embed):
                raise ValueError('kNN requires the current MIRA encoder')
            observations = np.concatenate([self.knn_states, states], axis=0)


            vectors = np.asarray(embed(observations), dtype=np.float64)
            if (vectors.ndim != 2 or len(vectors) != len(observations)
                    or vectors.shape[1] == 0 or not np.isfinite(vectors).all()):
                raise ValueError('kNN encoder must return finite row-aligned embeddings')
            values = []
            for index in range(len(self.knn_states), len(observations)):
                history = vectors[max(0, index-self.knn_capacity):index]
                if not len(history):
                    values.append(0.)
                    continue
                distances = np.linalg.norm(history-vectors[index], axis=1)
                k = min(self.knn_k, len(history))
                values.append(np.log1p(np.partition(distances, k-1)[:k].mean()))
            self.knn_states = observations[-self.knn_capacity:].copy()
            return np.asarray(values, dtype=np.float32)
        values = []
        for state in states:
            state = state.astype('<f4', copy=True)
            state[state == 0] = 0.
            key = state.tobytes().hex()
            count = self.counts.get(key, 0) + 1
            self.counts[key] = count
            values.append(1. / np.sqrt(count))
        return np.asarray(values, dtype=np.float32)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'replacement_reward.json').write_text(json.dumps(dict(
            kind=self.kind, state_dim=self.state_dim, counts=self.counts,
            knn_k=self.knn_k, knn_capacity=self.knn_capacity,
            rng_state=self.rng.bit_generator.state)), encoding='utf-8')
        if self.kind == 'knn':
            np.savez(directory / 'replacement_knn.npz', states=self.knn_states)
        if self.rnd is not None:
            self.rnd.save(directory / 'replacement_rnd.npz')

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        saved = json.loads((directory / 'replacement_reward.json').read_text(encoding='utf-8'))
        model = cls(saved['kind'], saved['state_dim'],
                    knn_k=saved.get('knn_k', 5), knn_capacity=saved.get('knn_capacity', 2048))
        model.counts = saved['counts']
        if 'rng_state' in saved:
            model.rng.bit_generator.state = saved['rng_state']
        if model.kind == 'knn':
            with np.load(directory / 'replacement_knn.npz', allow_pickle=False) as fitted:
                states = np.asarray(fitted['states'], dtype=np.float32)
            if (states.ndim != 2 or states.shape[1] != model.state_dim
                    or len(states) > model.knn_capacity or not np.isfinite(states).all()):
                raise ValueError('Invalid kNN history checkpoint')
            model.knn_states = states.copy()
        if model.rnd is not None:
            model.rnd = RandomNetworkDistillation.load(directory / 'replacement_rnd.npz')
        return model


def make_reward_replacement(args, state_dim):
    kind = getattr(args, 'exploration_reward', 'directional')
    if kind == 'directional':
        return None
    return ExplorationReward(kind, state_dim, seed=int(getattr(args, 'seed', 0) or 0),
                             lr=float(getattr(args, 'replacement_rnd_lr', 3e-4)),
                             knn_k=int(getattr(args, 'replacement_knn_k', 5)),
                             knn_capacity=int(getattr(args, 'replacement_knn_capacity', 2048)))
