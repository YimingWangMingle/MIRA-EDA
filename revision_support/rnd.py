"""RND with a fixed random MLP target and an Adam-trained MLP predictor.

Reward is mean squared feature error BEFORE updating the predictor. Inputs
use a fixed signed log1p transform clipped to [-5, 5], divided by 5; no
target labels, rewards, learned representation, or running statistics enter
this transform. Both networks are tanh MLPs with linear feature outputs.
"""

from pathlib import Path

import numpy as np


class RandomNetworkDistillation:
    def __init__(self, state_dim, seed=0, hidden=128, features=64, lr=3e-4):
        self.state_dim, self.hidden, self.features = map(int, (state_dim, hidden, features))
        self.lr = float(lr)
        rng = np.random.default_rng(seed)
        def network():
            return [
                rng.normal(0, 1 / np.sqrt(self.state_dim), (self.state_dim, self.hidden)),
                np.zeros(self.hidden),
                rng.normal(0, 1 / np.sqrt(self.hidden), (self.hidden, self.features)),
                np.zeros(self.features),
            ]
        self.target = network()
        self.predictor = network()
        self.m = [np.zeros_like(x) for x in self.predictor]
        self.v = [np.zeros_like(x) for x in self.predictor]
        self.updates = 0

    def _inputs(self, states):
        states = np.asarray(states, dtype=np.float64)
        if states.ndim == 1:
            states = states[None, :]
        if states.ndim != 2 or states.shape[1] != self.state_dim or not len(states):
            raise ValueError("RND expects a nonempty batch with the configured state dimension")
        if not np.isfinite(states).all():
            raise ValueError("RND states must be finite")
        return np.clip(np.sign(states) * np.log1p(np.abs(states)), -5, 5) / 5

    @staticmethod
    def _forward(x, weights):
        hidden = np.tanh(x @ weights[0] + weights[1])
        return hidden @ weights[2] + weights[3], hidden

    def errors(self, states):
        x = self._inputs(states)
        target, _ = self._forward(x, self.target)
        predicted, _ = self._forward(x, self.predictor)
        return np.mean((predicted - target) ** 2, axis=1)

    def observe(self, states):
        x = self._inputs(states)
        target, _ = self._forward(x, self.target)
        predicted, hidden = self._forward(x, self.predictor)
        error = predicted - target
        rewards = np.mean(error ** 2, axis=1)
        grad_output = 2 * error / error.size
        grad_hidden = (grad_output @ self.predictor[2].T) * (1 - hidden ** 2)
        grads = [x.T @ grad_hidden, grad_hidden.sum(axis=0),
                 hidden.T @ grad_output, grad_output.sum(axis=0)]
        self.updates += 1
        for i, grad in enumerate(grads):
            self.m[i] = .9 * self.m[i] + .1 * grad
            self.v[i] = .999 * self.v[i] + .001 * grad ** 2
            m = self.m[i] / (1 - .9 ** self.updates)
            v = self.v[i] / (1 - .999 ** self.updates)
            self.predictor[i] -= self.lr * m / (np.sqrt(v) + 1e-8)
        return rewards

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"{name}_{i}": x for name in ("target", "predictor", "m", "v")
                  for i, x in enumerate(getattr(self, name))}
        np.savez(path, state_dim=self.state_dim, hidden=self.hidden,
                 features=self.features, lr=self.lr, updates=self.updates, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            model = cls(int(saved["state_dim"]), hidden=int(saved["hidden"]),
                        features=int(saved["features"]), lr=float(saved["lr"]))
            for name in ("target", "predictor", "m", "v"):
                setattr(model, name, [saved[f"{name}_{i}"].copy() for i in range(4)])
            model.updates = int(saved["updates"])
        return model
