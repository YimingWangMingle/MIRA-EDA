import csv
import time
import json
from pathlib import Path

import numpy as np

from .env import dataset_enum, make_env
from atena_common.costs import session_cost_row, write_session


METRIC_NAMES = (
    "Precision",
    "T-BLEU-1",
    "T-BLEU-2",
    "T-BLEU-3",
    "EDA-Sim",
)


def official_metrics(schema, dataset_number, actions):
    from atena.evaluation.metrics import EvalInstance, get_dataframe_all_eval_metrics
    from atena.simulation.dataset import DatasetMeta

    schema_name, dataset_name = dataset_enum(schema, dataset_number)
    frame = get_dataframe_all_eval_metrics([
        EvalInstance(DatasetMeta(schema_name, dataset_name), actions)
    ])
    row = frame.iloc[0].to_dict()
    return {name: float(row[name]) for name in METRIC_NAMES}


def write_metrics_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def masked_probs(probs, mask):
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=np.float64).reshape(-1)
    if probs.shape != mask.shape:
        raise ValueError("action mask shape must match probabilities")
    if not np.all(np.logical_or(mask == 0.0, mask == 1.0)):
        raise ValueError("action mask must be binary")
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.maximum(probs, 0.0) * mask
    total = float(probs.sum())
    if total <= 0.0:
        valid = np.flatnonzero(mask > 0)
        fixed = np.zeros_like(probs, dtype=np.float64)
        if valid.size == 0:
            fixed[:] = 1.0 / max(len(fixed), 1)
        else:
            fixed[valid] = 1.0 / valid.size
        return fixed
    return probs / total


def evaluate_policy(
    args,
    model,
    result_dir,
    steps,
    action_filename_prefix="actions_steps",
):
    started = time.perf_counter()
    env = make_env(args.schema, args.dataset_number, args.seed + 777, args)
    setup_seconds = time.perf_counter() - started
    started = time.perf_counter()
    state = env.reset()
    from revision_support.mira_core import sample_direction
    direction = sample_direction(np.random.default_rng(int(args.seed)+777), args.mira_latent_dim)
    done = False
    total_reward = 0.0
    while not done:
        state_array = np.asarray(state, dtype=np.float32).reshape(1, -1)
        policy_state = np.concatenate([state_array[0], direction]).reshape(1, -1)
        probs, _ = model(policy_state, training=False)
        policy = masked_probs(probs.numpy()[0], env.legal_action_mask())
        action = int(np.argmax(policy))
        state, reward, done, _ = env.step(action)
        total_reward += float(reward)

    metrics = session_cost_row(env, started, lambda: official_metrics(args.schema, args.dataset_number, env.actions))
    row = {
        "method": args.method,
        "schema": args.schema,
        "dataset": int(args.dataset_number),
        "seed": int(args.seed),
        "steps": int(steps),
        "episode_reward": float(total_reward),
        "common_task_reward": float(total_reward),
        "session_setup_seconds": setup_seconds,
        **metrics,
    }
    write_json(
        Path(result_dir) / "{}{}.json".format(action_filename_prefix, int(steps)),
        [repr(action) for action in env.actions],
    )
    write_session(Path(result_dir) / "session_steps{}.json".format(int(steps)), env.actions)
    return row
