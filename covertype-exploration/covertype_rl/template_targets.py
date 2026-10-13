"""Reproducible seed-1-style targets; not the recovered historical search."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .targets import TARGET_SCHEMA_VERSION, _predicate_region_for_anchor, _region_key


def write_new_json(path, payload):
    """Never replace an existing target or generation record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=1, ensure_ascii=True)
        stream.write("\n")


def generate_template_targets(
    data, reference, seed, candidates_per_region=1024, excluded_constraints=()
):
    """Match reference region structure and size, without model feedback.

    A fixed candidate budget avoids selecting the search effort after seeing
    training performance. Distinct regions may overlap across target sets, but
    each generated target contains exactly the sum of its regional quotas.
    """
    if candidates_per_region < 1:
        raise ValueError("candidates_per_region must be positive")
    rng = np.random.default_rng(int(seed))
    selected = set()
    metadata = []
    excluded = {_region_key(c) for c in excluded_constraints}
    groups = {}
    for region_index, ref in enumerate(reference["metadata"]):
        rc = ref["constraints"]
        quota = int(ref["selected"])
        if quota < 1 or int(ref["pool_size"]) < quota:
            raise ValueError(f"Invalid quota for region {region_index}")
        group_key = (int(rc["cover_type"]), int(rc["wilderness"]))
        if group_key not in groups:
            rows = np.flatnonzero(
                (data.cover_type == group_key[0])
                & (data.wilderness == group_key[1])
            )
            view = SimpleNamespace(
                n_rows=len(rows),
                cover_type=data.cover_type[rows],
                wilderness=data.wilderness[rows],
                soil=data.soil[rows],
                continuous_bins=data.continuous_bins[rows],
            )
            groups[group_key] = (rows, view)
        rows, view = groups[group_key]
        if len(rows) < quota:
            raise ValueError(f"Insufficient candidates for region {region_index}")
        best = None
        valid_candidates = 0
        for attempt in range(int(candidates_per_region)):
            anchor_local = int(rng.integers(len(rows)))
            constraints, pool_local = _predicate_region_for_anchor(
                view, anchor_local, rng, quota
            )
            key = _region_key(constraints)
            if (
                len(pool_local) < quota
                or len(constraints["continuous_bins"]) != len(rc["continuous_bins"])
                or key in excluded
            ):
                continue
            score = abs(float(np.log(len(pool_local) / float(ref["pool_size"]))))
            score += 0.05 * (
                (int(constraints["soil"]) > 0) != (int(rc.get("soil", -1)) > 0)
            )
            if best is not None and score >= best[0]:
                continue
            anchor = int(rows[anchor_local])
            pool = rows[pool_local]
            distances = np.sum(
                (data.continuous_norm[pool] - data.continuous_norm[anchor]) ** 2,
                axis=1,
            )
            order = np.lexsort((pool, distances))
            targets = list(map(int, pool[order[:quota]]))
            if selected.intersection(targets):
                continue
            valid_candidates += 1
            canonical = {
                "cover_type": int(constraints["cover_type"]),
                "wilderness": int(constraints["wilderness"]),
                "soil": int(constraints["soil"]),
                "continuous_bins": {
                    str(k): int(v) for k, v in constraints["continuous_bins"].items()
                },
            }
            record = {
                "region": region_index,
                "reference_region": int(ref["region"]),
                "anchor": anchor,
                "schema": TARGET_SCHEMA_VERSION,
                "cover_type": int(data.cover_type[anchor]),
                "wilderness": int(data.wilderness[anchor]),
                "soil": int(data.soil[anchor]),
                "constraints": canonical,
                "pool_size": len(pool),
                "selected": quota,
                "template_distance": score,
                "reference_pool_size": int(ref["pool_size"]),
                "selected_candidate_index": attempt,
            }
            best = (score, targets, record)
        if best is None:
            raise ValueError(
                f"No valid candidate for region {region_index} after "
                f"{candidates_per_region} attempts; no random fill was used."
            )
        _, targets, record = best
        record["improving_candidates"] = valid_candidates
        selected.update(targets)
        excluded.add(_region_key(record["constraints"]))
        metadata.append(record)
    expected = sum(int(m["selected"]) for m in reference["metadata"])
    if len(selected) != expected:
        raise ValueError("Regional quotas did not produce the exact target count")
    return {
        "dataset": "covertype",
        "schema": TARGET_SCHEMA_VERSION,
        "seed": int(seed),
        "target_size": len(selected),
        "target_items": sorted(selected),
        "metadata": metadata,
        "generation": {
            "method": "seed1_like_region_template_search_v20260918",
            "reference_target": "fixed_seed_1",
            "candidates_per_region": int(candidates_per_region),
            "score": "abs(log(pool_size/reference_pool_size)) + 0.05*soil_presence_mismatch",
            "matching": "same cover_type, wilderness and continuous-constraint count",
            "row_selection": "nearest normalized continuous vectors; row-ID tie break",
            "excluded_region_count": len({_region_key(c) for c in excluded_constraints}),
            "historical_generator_recovered": False,
            "selection_uses_model_performance": False,
        },
    }
