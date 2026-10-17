import json
import re
from pathlib import Path

import numpy as np


TARGET_SCHEMA_VERSION = "clustered_predicate_regions_v2"
PRIORITY_CONTINUOUS_FEATURES = [0, 5, 9, 3, 1, 2, 6, 7, 8, 4]


def resolve_target_set(data, root_dir, target_set, target_seed=None, target_size=1000):
    targets_dir = Path(root_dir) / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    target_path = Path(target_set)
    if not target_path.suffix:
        target_path = targets_dir / f"{target_set}.json"
    elif not target_path.is_absolute():
        target_path = Path(root_dir) / target_path

    if target_path.exists():
        return load_target_items(target_path), target_path

    seed = int(target_seed) if target_seed is not None else _seed_from_name(target_path.stem)
    target_items, metadata = generate_sparse_targets(data, seed=seed, target_size=target_size)
    with open(target_path, "w") as f:
        json.dump(
            {
                "dataset": "covertype",
                "schema": TARGET_SCHEMA_VERSION,
                "seed": seed,
                "target_size": len(target_items),
                "target_items": target_items,
                "metadata": metadata,
            },
            f,
            indent=1,
        )
    return target_items, target_path


def load_target_items(path):
    with open(path) as f:
        payload = json.load(f)
    items = payload.get("target_items", payload)
    return sorted({int(item) for item in items})


def generate_sparse_targets(data, seed=1, target_size=1000, regions=8):

    rng = np.random.default_rng(seed)
    selected = set()
    metadata = []
    per_region = max(1, int(target_size) // int(regions))
    attempts = 0
    max_attempts = int(regions) * 60
    used_region_keys = set()

    while len(metadata) < int(regions) and attempts < max_attempts:
        attempts += 1
        anchor = int(rng.integers(0, data.n_rows))
        constraints, pool = _predicate_region_for_anchor(data, anchor, rng, per_region)
        if pool.size == 0:
            continue

        region_key = _region_key(constraints)
        if region_key in used_region_keys:
            continue
        used_region_keys.add(region_key)

        remaining = max(0, int(target_size) - len(selected))
        if remaining <= 0:
            break
        take = min(per_region, remaining, pool.size)
        anchor_vec = data.continuous_norm[anchor]
        dist = np.sum((data.continuous_norm[pool] - anchor_vec) ** 2, axis=1)
        nearest = pool[np.argpartition(dist, take - 1)[:take]]
        selected.update(map(int, nearest))

        metadata.append(
            {
                "region": len(metadata),
                "anchor": anchor,
                "schema": TARGET_SCHEMA_VERSION,
                "cover_type": int(data.cover_type[anchor]),
                "wilderness": int(data.wilderness[anchor]),
                "soil": int(data.soil[anchor]),
                "constraints": constraints,
                "pool_size": int(pool.size),
                "selected": int(take),
            }
        )

    if len(selected) < int(target_size):
        selected.update(_fill_remaining_targets(data, selected, rng, int(target_size)))

    return sorted(selected), metadata


def _predicate_region_for_anchor(data, anchor, rng, per_region):
    min_pool = max(int(per_region), 30)
    constraints = {
        "cover_type": int(data.cover_type[anchor]),
        "wilderness": int(data.wilderness[anchor]),
        "continuous_bins": {},
        "soil": -1,
    }
    pool = _pool_for_constraints(data, constraints)

    feature_order = list(PRIORITY_CONTINUOUS_FEATURES)
    rng.shuffle(feature_order)
    for feature in feature_order:
        candidate = {
            **constraints,
            "continuous_bins": dict(constraints["continuous_bins"]),
        }
        candidate["continuous_bins"][int(feature)] = int(data.continuous_bins[anchor, feature])
        candidate_pool = _pool_for_constraints(data, candidate)
        if candidate_pool.size >= min_pool:
            constraints = candidate
            pool = candidate_pool
        if len(constraints["continuous_bins"]) >= 3:
            break

    candidate = {
        **constraints,
        "continuous_bins": dict(constraints["continuous_bins"]),
        "soil": int(data.soil[anchor]),
    }
    candidate_pool = _pool_for_constraints(data, candidate)
    if candidate_pool.size >= min_pool:
        constraints = candidate
        pool = candidate_pool

    return constraints, pool


def _pool_for_constraints(data, constraints):
    mask = np.ones(data.n_rows, dtype=bool)
    cover_type = int(constraints.get("cover_type", -1))
    wilderness = int(constraints.get("wilderness", -1))
    soil = int(constraints.get("soil", -1))
    if cover_type > 0:
        mask &= data.cover_type == cover_type
    if wilderness > 0:
        mask &= data.wilderness == wilderness
    if soil > 0:
        mask &= data.soil == soil
    for feature, value in constraints.get("continuous_bins", {}).items():
        mask &= data.continuous_bins[:, int(feature)] == int(value)
    return np.flatnonzero(mask).astype(np.int64)


def _fill_remaining_targets(data, selected, rng, target_size):
    remaining_count = target_size - len(selected)
    if remaining_count <= 0:
        return []
    remaining = np.setdiff1d(
        np.arange(data.n_rows, dtype=np.int64),
        np.asarray(sorted(selected), dtype=np.int64),
        assume_unique=True,
    )
    take = min(int(remaining_count), remaining.size)
    return list(map(int, rng.choice(remaining, size=take, replace=False)))


def _region_key(constraints):
    continuous = tuple(sorted((int(k), int(v)) for k, v in constraints.get("continuous_bins", {}).items()))
    return (
        int(constraints.get("cover_type", -1)),
        int(constraints.get("wilderness", -1)),
        int(constraints.get("soil", -1)),
        continuous,
    )


def _seed_from_name(name):
    match = re.search(r"seed[_-]?(\d+)", name)
    return int(match.group(1)) if match else 1
