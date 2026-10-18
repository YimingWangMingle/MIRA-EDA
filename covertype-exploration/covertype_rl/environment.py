import hashlib
import sys
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from revision_support.resources import timed_query


class FixedSetEnvironment:
    def __init__(self, universe, actions, episode_steps=250, seed=0):
        self.universe = universe
        self.actions = actions
        self.episode_steps = int(episode_steps)
        self.rng = np.random.default_rng(seed)
        self.root_set_id = int(universe.metadata.get("root_set_id", 0))
        self.current_set_id = self.root_set_id
        self.previous_state = None
        self.previous_set_size = 0
        self.visited_cells = {}
        self.found_items_with_ratio = {}
        self.target_ratio = 0.1
        self.target_max_reward = 100.0
        self.target_set_size = int(universe.target_set_size)
        self.reward_multiplier = (
            self.target_max_reward / (self.target_set_size * self.target_ratio)
            if self.target_set_size > 0
            else 0.0
        )

    def reset(self):
        self.current_set_id = self.root_set_id
        self.previous_state = None
        self.visited_cells = {}
        self.found_items_with_ratio = {}
        state = self.universe.state_for_set(self.current_set_id)
        self.previous_state = state
        self.previous_set_size = self.universe.size_for_set(self.current_set_id)
        return state

    @timed_query("covertype_fixed", "step")
    def step(self, action_id):
        action_id = int(action_id)
        previous_set_id = int(self.current_set_id)
        next_set_id = self.universe.next_set(previous_set_id, action_id)
        self.current_set_id = int(next_set_id)
        valid = int(next_set_id != previous_set_id)
        metrics = self._metrics(next_set_id, valid=valid, get_reward=True)
        self.previous_state = metrics["state"]
        self.previous_set_size = metrics["set_size"]
        return metrics["state"], metrics, int(next_set_id), bool(valid)

    def valid_actions(self, set_id=None):
        set_id = self.current_set_id if set_id is None else int(set_id)
        row = np.asarray(self.universe.graph[int(set_id)], dtype=np.int64)
        valid = np.flatnonzero(row != int(set_id)).astype(np.int64)
        if valid.size == 0:
            return np.arange(self.universe.action_dim, dtype=np.int64)
        return valid

    @timed_query("covertype_fixed", "random_valid_set_step")
    def random_valid_set_step(self, rng):
        previous_set_id = int(self.current_set_id)
        next_set_id = int(rng.integers(0, self.universe.n_sets))
        self.current_set_id = next_set_id
        valid = int(next_set_id != previous_set_id)
        metrics = self._metrics(next_set_id, valid=valid, get_reward=True)
        self.previous_state = metrics["state"]
        self.previous_set_size = metrics["set_size"]
        return metrics["state"], metrics, int(next_set_id), bool(valid)

    def _metrics(self, set_id, valid, get_reward=True):
        from revision_support.query_service import service_available
        if service_available():
            from revision_support.query_clients import covertype_query
            result = covertype_query(self.universe, set_id)
            state, set_size = result['state'], int(result['set_size'])

            target_items = np.intersect1d(result['indices'], self.universe.targets_for_set(self.root_set_id), assume_unique=True)
        else:
            state = self.universe.state_for_set(set_id)
            set_size = self.universe.size_for_set(set_id)
            target_items = self.universe.targets_for_set(set_id)
        extrinsic = self._extrinsic(target_items, set_size, get_reward=get_reward)
        interestingness = self._interestingness(state, set_size)
        familiarity = extrinsic
        counter_curiosity = self._counter_curiosity(state) if valid else 0.0
        return {
            "state": state,
            "extrinsic_reward": float(extrinsic),
            "target_hits": int(len(target_items)),
            "interestingness": float(interestingness),
            "familiarity": float(familiarity),
            "counter_curiosity": float(counter_curiosity),
            "set_size": int(set_size),
            "sample_size": int(min(set_size, 500)),
            "valid": int(valid),
        }

    def _extrinsic(self, target_items, set_size, get_reward=True):
        if not get_reward or set_size <= 0 or self.target_set_size <= 0 or len(target_items) == 0:
            return 0.0

        target_found = set(map(int, np.asarray(target_items, dtype=np.int64).tolist()))
        reward_set_size_ratio = (
            float(len(target_found)) / float(set_size)
        ) * (
            float(self.target_set_size) / float(set_size)
        )

        reward = 0.0
        new_targets = target_found - set(self.found_items_with_ratio.keys())
        if new_targets:
            reward += len(new_targets) * reward_set_size_ratio * self.reward_multiplier
            for item in new_targets:
                self.found_items_with_ratio[int(item)] = reward_set_size_ratio

        old_targets = target_found - new_targets
        better_targets = [
            item
            for item in old_targets
            if self.found_items_with_ratio.get(int(item), 0.0) < reward_set_size_ratio
        ]
        if better_targets:
            reward += len(better_targets) * reward_set_size_ratio * self.reward_multiplier
            for item in better_targets:
                self.found_items_with_ratio[int(item)] = reward_set_size_ratio

        return float(reward)

    def _interestingness(self, state, set_size):
        if self.previous_state is None:
            return 0.0
        current_parts = (
            np.asarray(state[20:27], dtype=np.float64),
            np.asarray(state[27:31], dtype=np.float64),
            np.asarray(state[31:71], dtype=np.float64),
        )
        previous_parts = (
            np.asarray(self.previous_state[20:27], dtype=np.float64),
            np.asarray(self.previous_state[27:31], dtype=np.float64),
            np.asarray(self.previous_state[31:71], dtype=np.float64),
        )
        divergences = [
            _js_divergence(current, previous)
            for current, previous in zip(current_parts, previous_parts)
            if current.sum() > 0.0 and previous.sum() > 0.0
        ]
        if not divergences:
            return 0.0
        min_size = min(int(set_size), int(self.previous_set_size))
        max_size = max(int(set_size), int(self.previous_set_size))
        ratio = float(min_size) / float(max_size) if max_size > 0 else 0.0
        compactness_weight = max(0.0, 1.0 - abs(ratio - 0.5) * 2.0)
        return float(np.mean(divergences) * compactness_weight)

    def _counter_curiosity(self, state):
        cell = tuple(np.rint(np.asarray(state[:3], dtype=np.float32) * 2.0).astype(int).tolist())
        count = self.visited_cells.get(cell, 0) + 1
        self.visited_cells[cell] = count
        return 100.0 / float(count)


@dataclass
class Constraints:
    cont_bins: tuple
    cover: int = -1
    wilderness: int = -1
    soil: int = -1

    @classmethod
    def empty(cls, n_continuous):
        return cls(tuple([-1] * n_continuous), -1, -1, -1)

    def key(self):
        return self.cont_bins + (self.cover, self.wilderness, self.soil)


class CovertypeRLEnvironment:
    def __init__(self, data, target_items, actions, episode_steps=250, min_set_size=10, sample_size=500, seed=0):
        self.data = data
        self.actions = actions
        self.episode_steps = int(episode_steps)
        self.min_set_size = int(min_set_size)
        self.sample_size = int(sample_size)
        self.rng = np.random.default_rng(seed)
        self.constraints = Constraints.empty(data.continuous_bins.shape[1])
        self.previous_state_stats = None
        self.visited_cells = {}
        self.indexes = self._build_indexes()
        self.index_cache = OrderedDict()
        self.max_cache_size = 20000
        self.target_items = set(map(int, target_items or []))
        self.target_ratio = 0.1
        self.target_max_reward = 100.0
        self.target_set_size = len(self.target_items)
        self.reward_multiplier = (
            self.target_max_reward / (self.target_set_size * self.target_ratio)
            if self.target_set_size > 0
            else 0.0
        )
        self.found_items_with_ratio = {}
        self.target_mask = np.zeros(data.n_rows, dtype=bool)
        if self.target_items:
            self.target_mask[np.asarray(sorted(self.target_items), dtype=np.int64)] = True

    def reset(self):
        self.constraints = Constraints.empty(self.data.continuous_bins.shape[1])
        self.previous_state_stats = None
        self.visited_cells = {}
        self.found_items_with_ratio = {}
        return self._observe(self.constraints)

    def step(self, action_id):
        action = self.actions[int(action_id)]
        previous_constraints = self.constraints
        candidate_constraints = self._apply_action(previous_constraints, action)
        indices = self._indices_for(candidate_constraints)
        valid = indices.size >= self.min_set_size
        if valid:
            self.constraints = candidate_constraints
        else:
            indices = self._indices_for(previous_constraints)

        metrics = self._metrics(indices, valid=valid, get_reward=valid)
        self.previous_state_stats = metrics["state_core"]
        return metrics["state"], metrics, self.set_id(self.constraints), valid

    def random_valid_set_step(self, rng, max_tries=80):

        fallback_constraints = self.constraints
        fallback_indices = self._indices_for(fallback_constraints)

        for _ in range(int(max_tries)):
            candidate_constraints = self._sample_random_constraints(rng)
            indices = self._indices_for(candidate_constraints)
            if indices.size >= self.min_set_size:
                self.constraints = candidate_constraints
                metrics = self._metrics(indices, valid=True, get_reward=True)
                self.previous_state_stats = metrics["state_core"]
                return metrics["state"], metrics, self.set_id(self.constraints), True

        metrics = self._metrics(fallback_indices, valid=False, get_reward=False)
        self.previous_state_stats = metrics["state_core"]
        return metrics["state"], metrics, self.set_id(self.constraints), False

    def _apply_action(self, constraints, action):
        cont = list(constraints.cont_bins)
        cover = constraints.cover
        wilderness = constraints.wilderness
        soil = constraints.soil

        if action.op == "by_facet_cont":
            cont[action.feature] = int(action.value)
        elif action.op == "by_superset_cont":
            cont[action.feature] = -1
        elif action.op == "by_neighbors_cont":
            current = cont[action.feature]
            if current < 0:
                current = int(np.rint(self.data.continuous_bins[:, action.feature].mean()))
            cont[action.feature] = int(np.clip(current + action.delta, 0, self.data.n_bins - 1))
        elif action.op == "by_facet_cover":
            cover = int(action.value)
        elif action.op == "by_superset_cover":
            cover = -1
        elif action.op == "by_facet_wilderness":
            wilderness = int(action.value)
        elif action.op == "by_superset_wilderness":
            wilderness = -1
        elif action.op == "by_facet_soil":
            soil = int(action.value)
        elif action.op == "by_superset_soil":
            soil = -1
        elif action.op == "by_distribution":
            return self._apply_distribution_action(cont, cover, wilderness, soil, action.delta)

        return Constraints(tuple(cont), cover, wilderness, soil)

    def _apply_distribution_action(self, cont, cover, wilderness, soil, delta):
        active_cont = [idx for idx, value in enumerate(cont) if int(value) >= 0]
        if len(active_cont) <= 1:
            return Constraints(tuple(cont), cover, wilderness, soil)
        candidate = list(cont)
        for idx in active_cont:
            next_value = int(candidate[idx]) + int(delta)
            if next_value < 0 or next_value >= self.data.n_bins:
                return Constraints(tuple(cont), cover, wilderness, soil)
            candidate[idx] = int(next_value)
        return Constraints(tuple(candidate), cover, wilderness, soil)

    def _sample_random_constraints(self, rng):
        cont = [-1] * self.data.continuous_bins.shape[1]

        active_cont = int(rng.choice([1, 2, 3], p=[0.45, 0.40, 0.15]))
        active_cont = min(active_cont, len(cont))
        features = rng.choice(len(cont), size=active_cont, replace=False)
        for feature in features:
            cont[int(feature)] = int(rng.integers(0, self.data.n_bins))

        cover = -1
        wilderness = -1
        soil = -1
        category_group = int(rng.choice([0, 1, 2, 3], p=[0.30, 0.25, 0.20, 0.25]))
        if category_group == 1:
            cover = int(rng.integers(1, 8))
        elif category_group == 2:
            wilderness = int(rng.integers(1, 5))
        elif category_group == 3:
            soil = int(rng.integers(1, 41))

        return Constraints(tuple(cont), cover, wilderness, soil)

    def _observe(self, constraints):
        indices = self._indices_for(constraints)
        metrics = self._metrics(indices, valid=True, get_reward=False)
        self.previous_state_stats = metrics["state_core"]
        return metrics["state"]

    def _metrics(self, indices, valid, get_reward=True):
        sampled = self._sample_indices(indices)
        state_core = self._state_core(sampled)
        state = self._state_vector(state_core)
        extrinsic = self._extrinsic(indices, get_reward=get_reward)
        target_hits = int(np.count_nonzero(self.target_mask[indices])) if indices.size else 0
        interestingness = self._interestingness(state_core)
        familiarity = extrinsic
        counter_curiosity = self._counter_curiosity(state_core)
        if not valid:
            counter_curiosity = 0.0
        return {
            "state": state,
            "state_core": state_core,
            "extrinsic_reward": float(extrinsic),
            "target_hits": target_hits,
            "interestingness": float(interestingness),
            "familiarity": float(familiarity),
            "counter_curiosity": float(counter_curiosity),
            "set_size": int(indices.size),
            "sample_size": int(sampled.size),
            "valid": int(valid),
        }

    def _build_indexes(self):
        indexes = {"cont": [], "cover": {}, "wilderness": {}, "soil": {}}
        for feature in range(self.data.continuous_bins.shape[1]):
            feature_indexes = {}
            for value in range(self.data.n_bins):
                feature_indexes[value] = np.flatnonzero(self.data.continuous_bins[:, feature] == value).astype(np.int64)
            indexes["cont"].append(feature_indexes)
        for value in range(1, 8):
            indexes["cover"][value] = np.flatnonzero(self.data.cover_type == value).astype(np.int64)
        for value in range(1, 5):
            indexes["wilderness"][value] = np.flatnonzero(self.data.wilderness == value).astype(np.int64)
        for value in range(1, 41):
            indexes["soil"][value] = np.flatnonzero(self.data.soil == value).astype(np.int64)
        return indexes

    def _indices_for(self, constraints):
        key = constraints.key()
        cached = self.index_cache.get(key)
        if cached is not None:
            self.index_cache.move_to_end(key)
            return cached

        parts = []
        for feature, value in enumerate(constraints.cont_bins):
            if value >= 0:
                parts.append(self.indexes["cont"][feature][int(value)])
        if constraints.cover > 0:
            parts.append(self.indexes["cover"][constraints.cover])
        if constraints.wilderness > 0:
            parts.append(self.indexes["wilderness"][constraints.wilderness])
        if constraints.soil > 0:
            parts.append(self.indexes["soil"][constraints.soil])

        if not parts:
            result = np.arange(self.data.n_rows, dtype=np.int64)
        else:
            parts.sort(key=len)
            result = parts[0]
            for part in parts[1:]:
                result = np.intersect1d(result, part, assume_unique=True)
                if result.size == 0:
                    break

        self.index_cache[key] = result
        if len(self.index_cache) > self.max_cache_size:
            self.index_cache.popitem(last=False)
        return result

    def _sample_indices(self, indices):
        if indices.size <= self.sample_size:
            return indices
        return self.rng.choice(indices, size=self.sample_size, replace=False)

    def _state_core(self, indices):
        if indices.size == 0:
            cont_mean = np.zeros(self.data.continuous_norm.shape[1], dtype=np.float32)
            cont_std = np.zeros_like(cont_mean)
            cover_hist = np.zeros(7, dtype=np.float32)
            wilderness_hist = np.zeros(4, dtype=np.float32)
            soil_hist = np.zeros(40, dtype=np.float32)
        else:
            cont = self.data.continuous_norm[indices]
            cont_mean = cont.mean(axis=0).astype(np.float32)
            cont_std = cont.std(axis=0).astype(np.float32)
            cover_hist = _hist(self.data.cover_type[indices], 7)
            wilderness_hist = _hist(self.data.wilderness[indices], 4)
            soil_hist = _hist(self.data.soil[indices], 40)
        return {
            "cont_mean": cont_mean,
            "cont_std": cont_std,
            "cover_hist": cover_hist,
            "wilderness_hist": wilderness_hist,
            "soil_hist": soil_hist,
        }

    def _state_vector(self, core):
        constraint_vec = np.array(self.constraints.key(), dtype=np.float32)
        constraint_vec = np.where(constraint_vec < 0, -1.0, constraint_vec / 10.0)
        return np.concatenate(
            [
                core["cont_mean"],
                core["cont_std"],
                core["cover_hist"],
                core["wilderness_hist"],
                core["soil_hist"],
                constraint_vec,
            ]
        ).astype(np.float32)

    def _extrinsic(self, indices, get_reward=True):
        if not get_reward or indices.size == 0 or not self.target_items:
            return 0.0

        target_found = set(map(int, indices[self.target_mask[indices]].tolist()))
        if not target_found:
            return 0.0

        reward_set_size_ratio = (
            float(len(target_found)) / float(indices.size)
        ) * (
            float(self.target_set_size) / float(indices.size)
        )

        reward = 0.0
        known_items = set(map(int, self.found_items_with_ratio.keys()))
        new_targets = target_found - known_items
        if new_targets:
            reward += len(new_targets) * reward_set_size_ratio * self.reward_multiplier
            for item in new_targets:
                self.found_items_with_ratio[int(item)] = reward_set_size_ratio

        old_targets = target_found - new_targets
        better_targets = [
            item
            for item in old_targets
            if self.found_items_with_ratio.get(int(item), 0.0) < reward_set_size_ratio
        ]
        if better_targets:
            reward += len(better_targets) * reward_set_size_ratio * self.reward_multiplier
            for item in better_targets:
                self.found_items_with_ratio[int(item)] = reward_set_size_ratio

        return float(reward)

    def _interestingness(self, core):
        if self.previous_state_stats is None:
            return 0.0
        current_parts = (
            core["cover_hist"],
            core["wilderness_hist"],
            core["soil_hist"],
        )
        previous_parts = (
            self.previous_state_stats["cover_hist"],
            self.previous_state_stats["wilderness_hist"],
            self.previous_state_stats["soil_hist"],
        )
        divergences = [
            _js_divergence(current, previous)
            for current, previous in zip(current_parts, previous_parts)
            if np.asarray(current).sum() > 0.0 and np.asarray(previous).sum() > 0.0
        ]
        if not divergences:
            return 0.0
        return float(np.mean(divergences))

    def _counter_curiosity(self, core):
        cell = tuple(np.rint(core["cont_mean"][:3] * 2.0).astype(int).tolist())
        count = self.visited_cells.get(cell, 0) + 1
        self.visited_cells[cell] = count
        return 100.0 / float(count)

    def set_id(self, constraints):
        digest = hashlib.md5(str(constraints.key()).encode("utf-8")).hexdigest()
        return int(digest[:12], 16)


def _hist(values, labels):
    counts = np.bincount(np.asarray(values, dtype=np.int16), minlength=labels + 1)[1 : labels + 1].astype(np.float32)
    return counts / max(float(counts.sum()), 1.0)


def _js_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _kl(p, q):
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-12))))
