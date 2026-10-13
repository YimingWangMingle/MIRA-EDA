import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .actions import Action


PREPROCESS_SCHEMA_VERSION = "fixed_set_universe_v1"


@dataclass(frozen=True)
class Primitive:
    kind: str
    feature: int
    value: int
    indices: np.ndarray


@dataclass
class FixedSetUniverse:
    root_dir: Path
    metadata: dict
    constraints: np.ndarray
    set_states: np.ndarray
    set_sizes: np.ndarray
    graph: np.ndarray
    target_offsets: np.ndarray
    target_items: np.ndarray

    @property
    def n_sets(self):
        return int(self.set_states.shape[0])

    @property
    def state_dim(self):
        return int(self.set_states.shape[1])

    @property
    def action_dim(self):
        return int(self.graph.shape[1])

    @property
    def target_set_size(self):
        return int(self.metadata.get("target_set_size", 0))

    def targets_for_set(self, set_index):
        start = int(self.target_offsets[int(set_index)])
        end = int(self.target_offsets[int(set_index) + 1])
        return self.target_items[start:end]

    def state_for_set(self, set_index):
        return np.asarray(self.set_states[int(set_index)], dtype=np.float32)

    def size_for_set(self, set_index):
        return int(self.set_sizes[int(set_index)])

    def next_set(self, set_index, action_id):
        return int(self.graph[int(set_index), int(action_id)])


def load_fixed_universe(universe_dir):
    universe_dir = Path(universe_dir)
    with open(universe_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return FixedSetUniverse(
        root_dir=universe_dir,
        metadata=metadata,
        constraints=np.load(universe_dir / "constraints.npy", mmap_mode="r"),
        set_states=np.load(universe_dir / "set_states.npy", mmap_mode="r"),
        set_sizes=np.load(universe_dir / "set_sizes.npy", mmap_mode="r"),
        graph=np.load(universe_dir / "set_graph.npy", mmap_mode="r"),
        target_offsets=np.load(universe_dir / "target_offsets.npy", mmap_mode="r"),
        target_items=np.load(universe_dir / "target_items.npy", mmap_mode="r"),
    )


def ensure_fixed_universe(
    data,
    actions,
    target_items,
    target_path,
    root_dir,
    n_sets=50000,
    seed=1,
    min_set_size=10,
    max_set_size=200000,
    preprocess_name=None,
    force=False,
    csv_path=None,
    allow_build=False,
):
    from .cache_contract import validate_cache, certify_cache
    root_dir = Path(root_dir)
    preprocessed_dir = root_dir / "preprocessed"

    name = preprocess_name or f"fixed_sets_seed{seed}_n{n_sets}_min{min_set_size}"
    universe_dir = preprocessed_dir / name
    metadata_path = universe_dir / "metadata.json"

    csv_path = Path(csv_path) if csv_path is not None else root_dir / 'covertype.csv'
    expected = dict(seed=int(seed), requested_sets=int(n_sets), min_set_size=int(min_set_size),
                    max_set_size=int(max_set_size), n_bins=int(data.n_bins))
    if force and not allow_build:
        raise ValueError('Rebuild only through preprocess_fixed_sets.py, never during training')
    if metadata_path.exists() and not force:
        universe = validate_cache(universe_dir, csv_path, target_path,
                                  [action.label for action in actions], expected)
        if sorted(map(int, universe.targets_for_set(0))) != sorted(set(map(int, target_items))):
            raise ValueError('Requested target items differ from the certified cache')
        return universe, universe_dir
    if not allow_build:
        raise FileNotFoundError('Missing cache; run preprocess_fixed_sets.py explicitly: ' + str(universe_dir))

    universe_dir.mkdir(parents=True, exist_ok=True)
    target_regions = _load_target_region_constraints(target_path)
    target_items = sorted({int(item) for item in target_items})
    build_fixed_universe(
        data=data,
        actions=actions,
        target_items=target_items,
        target_regions=target_regions,
        output_dir=universe_dir,
        n_sets=int(n_sets),
        seed=int(seed),
        min_set_size=int(min_set_size),
        max_set_size=int(max_set_size),
    )
    validation_path = universe_dir / 'validation.json'
    if validation_path.exists():
        validation_path.unlink()
    universe = certify_cache(universe_dir, csv_path, target_path, actions, expected)
    return universe, universe_dir


def build_fixed_universe(
    data,
    actions,
    target_items,
    target_regions,
    output_dir,
    n_sets=50000,
    seed=1,
    min_set_size=10,
    max_set_size=200000,
):
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    primitives = _build_primitives(data)
    target_mask = np.zeros(data.n_rows, dtype=bool)
    if target_items:
        target_mask[np.asarray(target_items, dtype=np.int64)] = True

    constraints = []
    indices_by_key = {}

    def add_constraint(constraint, indices, allow_large=False):
        if len(constraints) >= int(n_sets):
            return
        key = _constraint_key(constraint)
        if key in indices_by_key:
            return
        size = int(indices.size)
        if size < int(min_set_size):
            return
        if not allow_large and size > int(max_set_size):
            return
        indices_by_key[key] = np.asarray(indices, dtype=np.int64)
        constraints.append(np.asarray(constraint, dtype=np.int16))

    root = np.full(data.continuous_bins.shape[1] + 3, -1, dtype=np.int16)
    add_constraint(root, np.arange(data.n_rows, dtype=np.int64), allow_large=True)

    # Make target regions explicitly reachable.
    for constraint in target_regions:
        normalized = _normalize_constraint(constraint, data.continuous_bins.shape[1])
        for relaxed in _relaxed_constraints(normalized):
            add_constraint(relaxed, _indices_for_constraint(data, relaxed), allow_large=True)

    # All one-predicate sets are part of the fixed universe.
    for primitive in primitives:
        constraint = root.copy()
        _apply_primitive_to_constraint(constraint, primitive)
        add_constraint(constraint, primitive.indices, allow_large=True)

    # Add all valid two-predicate sets.
    for left, right in itertools.combinations(primitives, 2):
        merged = _merge_primitives(root, [left, right])
        if merged is None:
            continue
        indices = _intersect_sorted(left.indices, right.indices)
        add_constraint(merged, indices)
        if len(constraints) >= int(n_sets):
            break

    # Fill the remainder with deterministic sampled three-predicate sets.
    attempts = 0
    max_attempts = max(int(n_sets) * 40, 100000)
    primitive_count = len(primitives)
    while len(constraints) < int(n_sets) and attempts < max_attempts:
        attempts += 1
        combo = rng.choice(primitive_count, size=3, replace=False)
        chosen = [primitives[int(idx)] for idx in combo]
        merged = _merge_primitives(root, chosen)
        if merged is None:
            continue
        indices = chosen[0].indices
        for primitive in chosen[1:]:
            indices = _intersect_sorted(indices, primitive.indices)
            if indices.size < int(min_set_size):
                break
        add_constraint(merged, indices)

    constraints = np.asarray(constraints, dtype=np.int16)
    set_sizes = np.zeros(constraints.shape[0], dtype=np.int32)
    set_states = []
    target_offsets = [0]
    target_hits_flat = []

    for constraint in constraints:
        indices = indices_by_key[_constraint_key(constraint)]
        set_sizes[len(set_states)] = int(indices.size)
        set_states.append(_state_for_indices(data, constraint, indices))
        hits = indices[target_mask[indices]]
        target_hits_flat.extend(map(int, hits.tolist()))
        target_offsets.append(len(target_hits_flat))

    set_states = np.asarray(set_states, dtype=np.float32)
    target_offsets = np.asarray(target_offsets, dtype=np.int64)
    target_hits_flat = np.asarray(target_hits_flat, dtype=np.int64)
    graph = _build_graph(constraints, actions, n_bins=data.n_bins)

    np.save(output_dir / "constraints.npy", constraints)
    np.save(output_dir / "set_sizes.npy", set_sizes)
    np.save(output_dir / "set_states.npy", set_states)
    np.save(output_dir / "target_offsets.npy", target_offsets)
    np.save(output_dir / "target_items.npy", target_hits_flat)
    np.save(output_dir / "set_graph.npy", graph)

    metadata = {
        "schema": PREPROCESS_SCHEMA_VERSION,
        "seed": int(seed),
        "requested_sets": int(n_sets),
        "n_sets": int(constraints.shape[0]),
        "min_set_size": int(min_set_size),
        "max_set_size": int(max_set_size),
        "target_set_size": int(len(target_items)),
        "state_dim": int(set_states.shape[1]),
        "action_dim": int(len(actions)),
        "action_labels": [action.label for action in actions],
        "root_set_id": 0,
        "note": "Fixed Covertype set universe. Runtime environment never creates new sets.",
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _build_primitives(data):
    primitives = []
    for feature in range(data.continuous_bins.shape[1]):
        for value in range(data.n_bins):
            indices = np.flatnonzero(data.continuous_bins[:, feature] == value).astype(np.int64)
            primitives.append(Primitive("cont", int(feature), int(value), indices))
    for value in range(1, 8):
        indices = np.flatnonzero(data.cover_type == value).astype(np.int64)
        primitives.append(Primitive("cover", -1, int(value), indices))
    for value in range(1, 5):
        indices = np.flatnonzero(data.wilderness == value).astype(np.int64)
        primitives.append(Primitive("wilderness", -1, int(value), indices))
    for value in range(1, 41):
        indices = np.flatnonzero(data.soil == value).astype(np.int64)
        primitives.append(Primitive("soil", -1, int(value), indices))
    return primitives


def _merge_primitives(root, primitives):
    constraint = root.copy()
    for primitive in primitives:
        if not _apply_primitive_to_constraint(constraint, primitive):
            return None
    return constraint


def _apply_primitive_to_constraint(constraint, primitive):
    if primitive.kind == "cont":
        current = int(constraint[int(primitive.feature)])
        if current >= 0 and current != int(primitive.value):
            return False
        constraint[int(primitive.feature)] = int(primitive.value)
        return True
    offset = constraint.shape[0] - 3
    slot = {"cover": offset, "wilderness": offset + 1, "soil": offset + 2}[primitive.kind]
    current = int(constraint[slot])
    if current >= 0 and current != int(primitive.value):
        return False
    constraint[slot] = int(primitive.value)
    return True


def _indices_for_constraint(data, constraint):
    parts = []
    n_cont = data.continuous_bins.shape[1]
    for feature in range(n_cont):
        value = int(constraint[feature])
        if value >= 0:
            parts.append(np.flatnonzero(data.continuous_bins[:, feature] == value).astype(np.int64))
    cover = int(constraint[n_cont])
    wilderness = int(constraint[n_cont + 1])
    soil = int(constraint[n_cont + 2])
    if cover > 0:
        parts.append(np.flatnonzero(data.cover_type == cover).astype(np.int64))
    if wilderness > 0:
        parts.append(np.flatnonzero(data.wilderness == wilderness).astype(np.int64))
    if soil > 0:
        parts.append(np.flatnonzero(data.soil == soil).astype(np.int64))
    if not parts:
        return np.arange(data.n_rows, dtype=np.int64)
    parts.sort(key=len)
    result = parts[0]
    for part in parts[1:]:
        result = _intersect_sorted(result, part)
        if result.size == 0:
            break
    return result


def _intersect_sorted(left, right):
    if left.size == 0 or right.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.intersect1d(left, right, assume_unique=True)


def _state_for_indices(data, constraint, indices):
    if indices.size == 0:
        cont_mean = np.zeros(data.continuous_norm.shape[1], dtype=np.float32)
        cont_std = np.zeros_like(cont_mean)
        cover_hist = np.zeros(7, dtype=np.float32)
        wilderness_hist = np.zeros(4, dtype=np.float32)
        soil_hist = np.zeros(40, dtype=np.float32)
    else:
        cont = data.continuous_norm[indices]
        cont_mean = cont.mean(axis=0).astype(np.float32)
        cont_std = cont.std(axis=0).astype(np.float32)
        cover_hist = _hist(data.cover_type[indices], 7)
        wilderness_hist = _hist(data.wilderness[indices], 4)
        soil_hist = _hist(data.soil[indices], 40)
    constraint_vec = np.asarray(constraint, dtype=np.float32)
    constraint_vec = np.where(constraint_vec < 0, -1.0, constraint_vec / 10.0)
    return np.concatenate([cont_mean, cont_std, cover_hist, wilderness_hist, soil_hist, constraint_vec]).astype(np.float32)


def _hist(values, labels):
    counts = np.bincount(np.asarray(values, dtype=np.int16), minlength=labels + 1)[1 : labels + 1].astype(np.float32)
    return counts / max(float(counts.sum()), 1.0)


def _build_graph(constraints, actions, n_bins=10):
    key_to_idx = {_constraint_key(constraint): idx for idx, constraint in enumerate(constraints)}
    graph = np.zeros((constraints.shape[0], len(actions)), dtype=np.int32)
    for set_idx, constraint in enumerate(constraints):
        for action_idx, action in enumerate(actions):
            candidate = _apply_action_to_constraint(constraint, action, n_bins=n_bins)
            graph[set_idx, action_idx] = int(key_to_idx.get(_constraint_key(candidate), set_idx))
    return graph


def _apply_action_to_constraint(constraint, action, n_bins=10):
    candidate = np.asarray(constraint, dtype=np.int16).copy()
    n_cont = candidate.shape[0] - 3
    cover_idx = n_cont
    wilderness_idx = n_cont + 1
    soil_idx = n_cont + 2
    if action.op == "by_facet_cont":
        candidate[int(action.feature)] = int(action.value)
    elif action.op == "by_superset_cont":
        candidate[int(action.feature)] = -1
    elif action.op == "by_neighbors_cont":
        current = int(candidate[int(action.feature)])
        if current < 0:
            current = 5
        candidate[int(action.feature)] = int(np.clip(current + int(action.delta), 0, 9))
    elif action.op == "by_facet_cover":
        candidate[cover_idx] = int(action.value)
    elif action.op == "by_superset_cover":
        candidate[cover_idx] = -1
    elif action.op == "by_facet_wilderness":
        candidate[wilderness_idx] = int(action.value)
    elif action.op == "by_superset_wilderness":
        candidate[wilderness_idx] = -1
    elif action.op == "by_facet_soil":
        candidate[soil_idx] = int(action.value)
    elif action.op == "by_superset_soil":
        candidate[soil_idx] = -1
    elif action.op == "by_distribution":
        candidate = _apply_distribution_to_constraint(candidate, action, n_cont, n_bins=n_bins)
    return candidate


def _apply_distribution_to_constraint(candidate, action, n_cont, n_bins=10):
    active_cont = [idx for idx in range(n_cont) if int(candidate[idx]) >= 0]
    if len(active_cont) <= 1:
        return candidate
    moved = np.asarray(candidate, dtype=np.int16).copy()
    for idx in active_cont:
        next_value = int(moved[idx]) + int(action.delta)
        if next_value < 0 or next_value >= int(n_bins):
            return candidate
        moved[idx] = int(next_value)
    return moved


def _load_target_region_constraints(target_path):
    if not target_path:
        return []
    path = Path(target_path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    regions = []
    for item in payload.get("metadata", []):
        constraints = item.get("constraints")
        if constraints:
            regions.append(constraints)
    return regions


def _normalize_constraint(payload, n_continuous):
    constraint = np.full(n_continuous + 3, -1, dtype=np.int16)
    for key, value in payload.get("continuous_bins", {}).items():
        constraint[int(key)] = int(value)
    constraint[n_continuous] = int(payload.get("cover_type", -1))
    constraint[n_continuous + 1] = int(payload.get("wilderness", -1))
    constraint[n_continuous + 2] = int(payload.get("soil", -1))
    return constraint


def _relaxed_constraints(constraint):
    active = np.flatnonzero(np.asarray(constraint) >= 0)
    variants = [np.asarray(constraint, dtype=np.int16)]
    for drop_idx in active:
        variant = np.asarray(constraint, dtype=np.int16).copy()
        variant[int(drop_idx)] = -1
        variants.append(variant)
    return variants


def _constraint_key(constraint):
    return tuple(int(value) for value in np.asarray(constraint).tolist())
