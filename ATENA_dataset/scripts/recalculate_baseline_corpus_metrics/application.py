import csv
import logging
from pathlib import Path

import pandas as pd

from . import (
    EXPECTED_DATASETS,
    METRICS,
    reconstruct_random_sessions,
)
from .orchestrator import (
    calculate_deterministic,
    calculate_random,
    calculate_recomputed_deterministic,
    calculate_recomputed_random,
)
from .policy import NumpyPolicyValueNet
from .runtime import (
    reconstruct_greedy_session,
    reconstruct_mira_session,
    reconstruct_official_session,
    reconstruct_policy_session,
    write_outputs,
)


METHOD_ORDER = ("official_atena", "random", "pure_a3c", "dora", "greedy", "MIRA")
BENCHMARK_COMMIT = "8428c48011dbf2f7f04f3ffded917038e4670657"


def recalculate_all(results_dir, seed=0):
    results_dir = Path(results_dir)
    _enable_legacy_string_behavior()
    deterministic_sessions = {
        "official_atena": _reconstruct_official(results_dir, seed),
        "greedy": _reconstruct_greedy(results_dir, seed),
        "dora": _reconstruct_policy(results_dir, "dora", seed),
        "pure_a3c": _reconstruct_policy(results_dir, "pure_a3c", seed),
        "MIRA": _reconstruct_mira(results_dir, seed),
    }
    detail_rows = []
    summary_by_method = {}
    for method in ("official_atena", "greedy", "dora", "pure_a3c", "MIRA"):
        detail, summary = calculate_recomputed_deterministic(
            method,
            deterministic_sessions[method],
            single_metric=_single_metric,
            corpus_metric=_corpus_metric,
            seed=seed,
        )
        detail_rows.extend(detail)
        summary_by_method[method] = summary
    random_sessions = {
        key: reconstruct_random_sessions(
            _result_dir(results_dir, "random", key, seed)
        )
        for key in EXPECTED_DATASETS
    }
    random_detail, random_summary = calculate_recomputed_random(
        random_sessions,
        single_metric=_single_metric,
        corpus_metric=_corpus_metric,
        seed=seed,
    )
    detail_rows.extend(random_detail)
    summary_by_method["random"] = random_summary
    detail_rows.sort(key=lambda row: (METHOD_ORDER.index(row["method"]), row["schema"], row["dataset"]))
    summary_rows = [summary_by_method[method] for method in METHOD_ORDER]
    manifest = {
        "validation_passed": True,
        "methods": list(METHOD_ORDER),
        "schemas": ["cyber", "flights"],
        "datasets_per_schema": [1, 2, 3, 4],
        "seed": int(seed),
        "episode_length": 12,
        "random_k": 16,
        "benchmark_commit": BENCHMARK_COMMIT,
        "t_bleu_aggregation": "official corpus T-BLEU over all 8 datasets",
        "precision_aggregation": "official evaluator mean over all 8 reconstructed final sessions",
        "eda_sim_aggregation": "official evaluator mean over all 8 reconstructed final sessions",
        "legacy_string_behavior": True,
        "metric_source": "official evaluator recomputation from reconstructed final sessions",
        "policy_inference": "NumPy Dense/ReLU/Softmax from final Keras HDF5 weights; no checkpoint selection",
        "saved_eval_metrics_role": "audit only; not used in final table values",
        "results_dir": str(results_dir.resolve()),
    }
    for row in summary_rows:
        row["seed"] = int(seed)
    paths = write_outputs(results_dir / "corpus" / f"seed{seed}", detail_rows, summary_rows, manifest)
    return detail_rows, summary_rows, manifest, paths


def _load_all_saved_rows(results_dir, seed):
    all_rows = {}
    for method in METHOD_ORDER:
        method_rows = {}
        for key in EXPECTED_DATASETS:
            if method == "MIRA":
                path = _mira_result_dir(results_dir, key, seed) / "eval_metrics.csv"
            elif method == "official_atena":
                path = (
                    results_dir
                    / "official_atena_eval"
                    / key[0]
                    / f"dataset{key[1]}"
                    / f"seed{seed}"
                    / "eval_metrics.csv"
                )
            else:
                path = _result_dir(results_dir, method, key, seed) / "eval_metrics.csv"
            rows = _read_metric_rows(path)
            expected_count = 16 if method == "random" else None
            if expected_count is not None and len(rows) != expected_count:
                raise ValueError(f"{path} expected {expected_count} rows, got {len(rows)}")
            method_rows[key] = rows
        all_rows[method] = method_rows
    return all_rows


def _read_metric_rows(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no metric rows")
    for row in rows:
        for metric in METRICS:
            row[metric] = float(row[metric])
    return rows


def _reconstruct_official(results_dir, seed):
    return {
        key: reconstruct_official_session(
            results_dir,
            key[0],
            key[1],
            seed=seed,
        )
        for key in EXPECTED_DATASETS
    }


def _reconstruct_greedy(results_dir, seed):
    return {
        key: reconstruct_greedy_session(
            _result_dir(results_dir, "greedy", key, seed)
        )
        for key in EXPECTED_DATASETS
    }


def _reconstruct_policy(results_dir, method, seed):
    return {
        key: reconstruct_policy_session(
            _result_dir(results_dir, method, key, seed),
            model_factory=NumpyPolicyValueNet,
        )
        for key in EXPECTED_DATASETS
    }


def _reconstruct_mira(results_dir, seed):
    return {
        key: reconstruct_mira_session(
            _mira_result_dir(results_dir, key, seed),
            model_factory=NumpyPolicyValueNet,
        )
        for key in EXPECTED_DATASETS
    }


def _mira_result_dir(results_dir, key, seed):
    results_dir = Path(results_dir)
    direct = results_dir / "MIRA" / f"{key[0]}{key[1]}" / f"seed{seed}"
    legacy = results_dir / "MIRA" / "mira" / f"{key[0]}{key[1]}" / f"seed{seed}"
    if direct.is_dir():
        return direct
    if legacy.is_dir():
        return legacy
    raise FileNotFoundError(f"MIRA result directory not found: {direct} or {legacy}")


def _single_metric(key, actions):
    return _official_metric({key: actions})


def _corpus_metric(sessions):
    instances = _official_instances(sessions)
    return {
        f"T-BLEU-{n}": _tree_bleu(n, instances)
        for n in (1, 2, 3)
    }


def _tree_bleu(n, instances):
    from atena.evaluation.metrics import DisplaysTreeBleuMetric

    return float(DisplaysTreeBleuMetric(n, instances).compute())


def _official_metric(sessions):
    from atena.evaluation.metrics import get_dataframe_all_eval_metrics

    instances = _official_instances(sessions)
    frame = get_dataframe_all_eval_metrics(instances)
    return {metric: float(frame.iloc[0][metric]) for metric in METRICS}


def _official_instances(sessions):
    from atena.evaluation.metrics import EvalInstance
    from atena.simulation.dataset import DatasetMeta
    from atena_baselines.env import dataset_enum

    instances = []
    for key in sorted(sessions):
        schema_name, dataset_name = dataset_enum(*key)
        instances.append(
            EvalInstance(DatasetMeta(schema_name, dataset_name), sessions[key])
        )
    return instances


def _result_dir(results_dir, method, key, seed):
    return results_dir / method / f"{key[0]}{key[1]}" / f"seed{seed}"


def _enable_legacy_string_behavior():
    try:
        pd.set_option("future.infer_string", False)
    except (KeyError, ValueError):
        pass
    logging.getLogger("atena.simulation.display").setLevel(logging.ERROR)
