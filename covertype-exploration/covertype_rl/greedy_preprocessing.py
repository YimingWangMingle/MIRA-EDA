from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class OfficialGreedyInput:
    seed: int
    target_set: str
    preprocess_name: str


@dataclass(frozen=True)
class ValidatedGreedyPreprocessing:
    universe_dir: Path
    metadata: dict


OFFICIAL_GREEDY_INPUTS = {
    seed: OfficialGreedyInput(
        seed=seed,
        target_set=f"fixed_seed_{seed}",
        preprocess_name=f"by_distribution_path100k_seed{seed}",
    )
    for seed in (1, 2, 3)
}

REQUIRED_FILES = (
    "metadata.json",
    "constraints.npy",
    "set_states.npy",
    "set_sizes.npy",
    "set_graph.npy",
    "target_offsets.npy",
    "target_items.npy",
)

EXPECTED_METADATA = {
    "schema": "fixed_set_universe_v1",
    "requested_sets": 100_000,
    "n_sets": 100_000,
    "min_set_size": 10,
    "max_set_size": 200_000,
    "target_set_size": 1_000,
    "state_dim": 84,
    "action_dim": 202,
    "root_set_id": 0,
}


def official_greedy_input(seed: int) -> OfficialGreedyInput:

    try:
        return OFFICIAL_GREEDY_INPUTS[int(seed)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported official Covertype Greedy seed: {seed!r}") from exc


def validate_official_greedy_preprocessing(
    root_dir: str | Path,
    *,
    seed: int,
    target_set: str,
    preprocess_name: str,
    action_labels: Sequence[str],
) -> ValidatedGreedyPreprocessing:

    official = official_greedy_input(seed)
    if target_set != official.target_set:
        raise ValueError(
            f"Target set mismatch for seed {official.seed}: expected "
            f"{official.target_set!r}, got {target_set!r}"
        )
    if preprocess_name != official.preprocess_name:
        raise ValueError(
            f"Preprocessing mismatch for seed {official.seed}: expected "
            f"{official.preprocess_name!r}, got {preprocess_name!r}"
        )

    universe_dir = Path(root_dir) / "preprocessed" / official.preprocess_name
    missing = [name for name in REQUIRED_FILES if not (universe_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Official Covertype preprocessing is incomplete at {universe_dir}: "
            f"missing {', '.join(missing)}"
        )

    metadata_path = universe_dir / "metadata.json"
    try:
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid metadata from {metadata_path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata at {metadata_path} must be a JSON object")

    expected = {**EXPECTED_METADATA, "seed": official.seed}
    mismatches = [
        f"{key}={metadata.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"Metadata mismatch at {metadata_path}: " + "; ".join(mismatches)
        )

    runtime_labels = list(action_labels)
    cached_labels = metadata.get("action_labels")
    if cached_labels != runtime_labels:
        raise ValueError(
            f"Action labels in {metadata_path} do not match the runtime action space"
        )

    return ValidatedGreedyPreprocessing(
        universe_dir=universe_dir.resolve(),
        metadata=dict(metadata),
    )
