from . import (
    EXPECTED_DATASETS,
    METRICS,
    mean_metric_rows,
    validate_dataset_keys,
    validate_metric_row,
)
from .engine import (
    apply_saved_scalar_metrics,
    deterministic_summary,
    ordered_sessions,
    random_summary,
)


def calculate_deterministic(
    method,
    sessions,
    saved_by_key,
    single_metric,
    corpus_metric,
    seed=0,
):
    ordered_sessions(sessions)
    validate_dataset_keys(saved_by_key)
    detail = []
    saved_rows = []
    for key in sorted(EXPECTED_DATASETS):
        expected = saved_by_key[key]
        observed = single_metric(key, sessions[key])
        validate_metric_row(expected, observed, f"{method}/{key[0]}{key[1]}")
        row = _detail_row(method, key, expected, seed)
        detail.append(row)
        saved_rows.append(expected)
    summary = {
        "method": method,
        **deterministic_summary(saved_rows, corpus_metric(sessions)),
    }
    return detail, summary


def calculate_recomputed_deterministic(
    method,
    sessions,
    single_metric,
    corpus_metric,
    seed=0,
):
    ordered_sessions(sessions)
    detail = [
        _detail_row(method, key, single_metric(key, sessions[key]), seed)
        for key in sorted(EXPECTED_DATASETS)
    ]
    scalar_metrics = mean_metric_rows(detail)
    corpus = corpus_metric(sessions)
    summary = {
        "method": method,
        "Precision": scalar_metrics["Precision"],
        "T-BLEU-1": float(corpus["T-BLEU-1"]),
        "T-BLEU-2": float(corpus["T-BLEU-2"]),
        "T-BLEU-3": float(corpus["T-BLEU-3"]),
        "EDA-Sim": scalar_metrics["EDA-Sim"],
    }
    return detail, summary


def calculate_random(
    sessions_by_key,
    saved_by_key,
    single_metric,
    corpus_metric,
    seed=0,
):
    validate_dataset_keys(sessions_by_key)
    validate_dataset_keys(saved_by_key)
    counts = {len(sessions_by_key[key]) for key in EXPECTED_DATASETS}
    saved_counts = {len(saved_by_key[key]) for key in EXPECTED_DATASETS}
    if counts != {16} or saved_counts != {16}:
        raise ValueError(
            f"random requires 16 sessions and rows per dataset; "
            f"sessions={sorted(counts)} saved={sorted(saved_counts)}"
        )
    detail = []
    for key in sorted(EXPECTED_DATASETS):
        detail.append(_detail_row("random", key, mean_metric_rows(saved_by_key[key]), seed))
    corpus_rows = []
    for episode in range(16):
        episode_sessions = {
            key: sessions_by_key[key][episode]
            for key in EXPECTED_DATASETS
        }
        ordered_sessions(episode_sessions)
        saved_episode_rows = []
        for key in sorted(EXPECTED_DATASETS):
            expected = saved_by_key[key][episode]
            observed = single_metric(key, episode_sessions[key])
            validate_metric_row(
                expected,
                observed,
                f"random/{key[0]}{key[1]}/episode{episode}",
            )
            saved_episode_rows.append(expected)
        corpus_rows.append(
            apply_saved_scalar_metrics(
                corpus_metric(episode_sessions),
                saved_episode_rows,
            )
        )
    summary = {"method": "random", **random_summary(corpus_rows)}
    return detail, summary


def calculate_recomputed_random(
    sessions_by_key,
    single_metric,
    corpus_metric,
    seed=0,
):
    validate_dataset_keys(sessions_by_key)
    counts = {len(sessions_by_key[key]) for key in EXPECTED_DATASETS}
    if counts != {16}:
        raise ValueError(
            f"random requires 16 sessions per dataset; sessions={sorted(counts)}"
        )

    observed_by_key = {key: [] for key in EXPECTED_DATASETS}
    corpus_rows = []
    for episode in range(16):
        episode_sessions = {
            key: sessions_by_key[key][episode]
            for key in EXPECTED_DATASETS
        }
        ordered_sessions(episode_sessions)
        episode_rows = []
        for key in sorted(EXPECTED_DATASETS):
            observed = single_metric(key, episode_sessions[key])
            observed_by_key[key].append(observed)
            episode_rows.append(observed)
        scalar_metrics = mean_metric_rows(episode_rows)
        corpus = corpus_metric(episode_sessions)
        corpus_rows.append(
            {
                "Precision": scalar_metrics["Precision"],
                "T-BLEU-1": float(corpus["T-BLEU-1"]),
                "T-BLEU-2": float(corpus["T-BLEU-2"]),
                "T-BLEU-3": float(corpus["T-BLEU-3"]),
                "EDA-Sim": scalar_metrics["EDA-Sim"],
            }
        )

    detail = [
        _detail_row("random", key, mean_metric_rows(observed_by_key[key]), seed)
        for key in sorted(EXPECTED_DATASETS)
    ]
    summary = {
        "method": "random",
        **mean_metric_rows(corpus_rows),
    }
    return detail, summary


def _detail_row(method, key, metrics, seed=0):
    return {
        "method": method,
        "schema": key[0],
        "dataset": key[1],
        "seed": int(seed),
        **{name: float(metrics[name]) for name in METRICS},
    }
