import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


METRICS = ("Precision", "T-BLEU-1", "T-BLEU-2", "T-BLEU-3", "EDA-Sim")
EXPECTED_DATASETS = {
    (schema, dataset)
    for schema in ("cyber", "flights")
    for dataset in range(1, 5)
}


def mean_metric_rows(rows):
    if not rows:
        raise ValueError("cannot average zero metric rows")
    return {
        name: float(np.mean([float(row[name]) for row in rows]))
        for name in METRICS
    }


def validate_dataset_keys(keys):
    keys = set(keys)
    if keys != EXPECTED_DATASETS:
        missing = sorted(EXPECTED_DATASETS - keys)
        extra = sorted(keys - EXPECTED_DATASETS)
        raise ValueError(
            f"missing dataset keys={missing}; extra dataset keys={extra}"
        )


def validate_metric_row(expected, observed, context):
    tolerances = {name: 1e-10 for name in METRICS}
    tolerances["EDA-Sim"] = 2e-2
    for name in METRICS:
        if not np.isclose(
            float(expected[name]),
            float(observed[name]),
            rtol=0.0,
            atol=tolerances[name],
        ):
            raise ValueError(
                f"{context}: {name} expected={expected[name]} "
                f"observed={observed[name]}"
            )


def load_result_config(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SimpleNamespace(**payload)


def _masked_probs(probs, mask):
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=np.float64).reshape(-1)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.maximum(probs, 0.0) * mask
    total = float(probs.sum())
    if total <= 0.0:
        valid = np.flatnonzero(mask > 0)
        if valid.size == 0:
            raise ValueError("environment has no legal action")
        probs = np.zeros_like(probs)
        probs[valid] = 1.0 / valid.size
        return probs
    return probs / total


def rollout_argmax(env, model, direction=None):
    state = env.reset()
    done = False
    while not done:
        policy_state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        if direction is not None:
            policy_state = np.concatenate([policy_state, np.asarray(direction, dtype=np.float32).reshape(1, -1)], axis=1)
        probs, _ = model(policy_state, training=False)
        policy = _masked_probs(probs.numpy()[0], env.legal_action_mask())
        state, _, done, _ = env.step(int(np.argmax(policy)))
    if len(env.actions) != 12:
        raise ValueError(f"expected 12 actions, got {len(env.actions)}")
    return list(env.actions)


def _default_env_factory(schema, dataset, seed, args, reward_mode):
    from atena_baselines.env import make_env

    return make_env(schema, dataset, seed, args, reward_mode=reward_mode)


def reconstruct_random_sessions(result_dir, env_factory=None):
    result_dir = Path(result_dir)
    args = load_result_config(result_dir / "config.json")
    env_factory = env_factory or _default_env_factory
    sessions = []
    for episode in range(int(args.random_eval_episodes)):
        env = env_factory(
            args.schema,
            args.dataset_number,
            args.seed + episode,
            args,
            "official_compound",
        )
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(env.sample_legal_action())
        if len(env.actions) != 12:
            raise ValueError(
                f"random episode {episode} produced {len(env.actions)} actions"
            )
        sessions.append(list(env.actions))
    return sessions
