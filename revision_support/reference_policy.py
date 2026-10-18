import hashlib

import numpy as np
from scipy.optimize import linear_sum_assignment

CORE_VERSION = 'mira_reference_wasserstein_v3'


def state_identity(*parts):
    return hashlib.sha256(repr(parts).encode('utf-8')).hexdigest()


def sample_reference_actions(masks, state_keys, terminals, count, rng):
    masks = np.asarray(masks, dtype=bool)
    terminals = np.asarray(terminals, dtype=bool).reshape(-1)
    if masks.ndim != 2 or len(masks) != len(terminals) or len(state_keys) != len(masks) or count < 1:
        raise ValueError('Invalid reference-policy batch')
    result = np.full((len(masks), int(count)), -1, dtype=np.int64)
    seen = {}
    for i, key in enumerate(state_keys):
        key = key.item() if isinstance(key, np.generic) else key
        if key in seen:
            previous = seen[key]
            if terminals[i] != terminals[previous] or not np.array_equal(masks[i], masks[previous]):
                raise ValueError('Inconsistent legal actions for the same complete-state identity')
            result[i] = result[previous]
            continue
        seen[key] = i
        if terminals[i]:
            continue
        valid = np.flatnonzero(masks[i])
        if not valid.size:
            raise ValueError('A nonterminal reference state must have valid actions')
        result[i] = rng.choice(valid, size=int(count), replace=True)
    return result


def empirical_wasserstein(successors, pairs):
    points = np.asarray(successors, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1)
    if points.ndim != 3 or points.shape[1] < 1 or len(pairs) != len(points):
        raise ValueError('Expected equal-weight successor samples [states, K, latent]')
    if not np.isfinite(points).all() or np.any(pairs < 0) or np.any(pairs >= len(points)):
        raise ValueError('Invalid Wasserstein samples/pairs')
    distances = np.zeros((len(points), 1), dtype=np.float32)
    for i, j in enumerate(pairs):
        if i == j or np.array_equal(points[i], points[j]):
            continue
        cost = np.linalg.norm(points[i, :, None, :] - points[j, None, :, :], axis=-1)
        rows, cols = linear_sum_assignment(cost)
        distances[i, 0] = cost[rows, cols].mean()
    return distances
