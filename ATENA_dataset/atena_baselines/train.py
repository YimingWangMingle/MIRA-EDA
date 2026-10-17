import csv
import time
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from atena_common.costs import QueryCosts, session_cost_row, write_costs, write_session, average_random_rows

import numpy as np

from .env import make_env
from .evaluate import official_metrics, write_json, write_metrics_csv
from .rollout_pool import PersistentRolloutExecutor
from .selection import attach_selection_features


METHODS = {"random", "greedy", "dora", "pure_a3c"}

FORMAL_METRICS_FILE = "eval_metrics.csv"
FORMAL_FINAL_FILE = "final_metrics.json"
FORMAL_POLICY_FILE = "policy.weights.h5"
FORMAL_ACTION_PREFIX = "actions_steps"
TRAIN_LOG_FIELDS = (
    "update",
    "steps",
    "episode_reward",
    "episode_task_reward",
    "policy_loss",
    "value_loss",
    "entropy",
)


def run_greedy(*args, **kwargs):
    from .greedy import run_greedy as greedy_runner

    return greedy_runner(*args, **kwargs)


def method_reward_mode(method: str) -> str:
    return "dora" if method == "dora" else "official_compound"


def formal_final_row(metrics_rows):
    return dict(metrics_rows[-1]) if metrics_rows else None


def run_experiment(args):
    training_started = time.perf_counter()
    if args.method not in METHODS:
        raise ValueError(f"Unknown method {args.method}. Choices: {sorted(METHODS)}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    result_dir = (
        Path(args.output_dir)
        / args.method
        / f"{args.schema}{args.dataset_number}"
        / f"seed{args.seed}"
    )
    result_dir.mkdir(parents=True, exist_ok=False)
    write_json(result_dir / "config.json", vars(args))

    if args.method == "greedy":
        return run_greedy(args, result_dir)
    if args.method == "random":
        return run_random(args, result_dir)

    probe_env = make_env(
        args.schema,
        args.dataset_number,
        args.seed,
        args,
        reward_mode=method_reward_mode(args.method),
    )
    if args.method == "dora":
        write_json(
            result_dir / "dora_target_set.json",
            {
                "schema": args.schema,
                "dataset": int(args.dataset_number),
                "seed": int(args.seed),
                "target_seed": int(getattr(args, "dora_target_seed", 0)),
                "target_size": len(probe_env.dora_target_indices),
                "curiosity_ratio": float(getattr(args, "dora_curiosity_ratio", 0.25)),
                "actions": probe_env.dora_target_actions_repr(),
            },
        )

    import tensorflow as tf
    from .models import PolicyValueNet

    tf.random.set_seed(args.seed)
    model = PolicyValueNet(probe_env.state_dim, probe_env.action_dim, hidden=args.hidden)
    optimizer = tf.keras.optimizers.Adam(float(args.lr))
    training_queries = QueryCosts()

    metrics_rows = []
    train_log_path = result_dir / "train_log.csv"
    _init_train_log(train_log_path)
    steps_done = 0
    update_idx = 0

    with PersistentRolloutExecutor(args.workers, rollout_worker) as executor:
        while steps_done < args.steps:
            rollout_count = min(
                args.workers,
                max(
                    1,
                    (args.steps - steps_done + args.episode_length - 1)
                    // args.episode_length,
                ),
            )
            policy_weights = model.get_weights()
            worker_args = []
            for worker_id in range(int(rollout_count)):
                worker_seed = int(
                    args.seed * 100000 + update_idx * args.workers + worker_id
                )
                worker_args.append((args, policy_weights, worker_seed))

            rollouts = executor.map(worker_args)
            for rollout in rollouts:
                training_queries.add(rollout["query_costs"])
            batch = _merge_rollouts(rollouts)
            steps_done += int(len(batch["rewards"]))
            update_idx += 1

            update_stats = _a3c_update(model, optimizer, batch, args)
            _append_train_log(
                train_log_path,
                {
                    "update": update_idx,
                    "steps": steps_done,
                    "episode_reward": float(
                        np.mean([rollout["episode_reward"] for rollout in rollouts])
                    ),
                    "episode_task_reward": float(
                        np.mean(
                            [rollout["episode_task_reward"] for rollout in rollouts]
                        )
                    ),
                    **update_stats,
                },
            )

            if update_idx % max(1, args.eval_interval) == 0 or steps_done >= args.steps:
                eval_row = evaluate_policy(args, model, result_dir, steps_done)
                metrics_rows.append(eval_row)
                write_metrics_csv(result_dir / FORMAL_METRICS_FILE, metrics_rows)
                model.save_weights(str(result_dir / FORMAL_POLICY_FILE))

    if metrics_rows:
        write_json(result_dir / FORMAL_FINAL_FILE, formal_final_row(metrics_rows))
    write_costs(result_dir, metrics_rows, training_started, steps_done, training_queries)
    return metrics_rows


def run_random(args, result_dir):
    rows = []
    for episode in range(max(1, args.random_eval_episodes)):
        started = time.perf_counter()
        env = make_env(args.schema, args.dataset_number, args.seed + episode, args,
                       reward_mode=method_reward_mode(args.method))
        setup_seconds = time.perf_counter() - started
        started = time.perf_counter()
        env.reset()
        done = False
        total = 0.0
        while not done:
            _, reward, done, _ = env.step(env.sample_legal_action())
            total += float(reward)
        row = dict(method=args.method, schema=args.schema, dataset=args.dataset_number,
                   seed=args.seed, steps=0, episode=episode, episode_reward=total,
                   common_task_reward=total, session_setup_seconds=setup_seconds,
                   **session_cost_row(env, started, lambda: official_metrics(args.schema, args.dataset_number, env.actions)))
        write_session(Path(result_dir) / f"session_random_{episode}.json", env.actions)
        rows.append(row)
    write_metrics_csv(result_dir / FORMAL_METRICS_FILE, rows)
    write_json(result_dir / FORMAL_FINAL_FILE, average_random_rows(rows))
    write_costs(result_dir, rows)
    return rows


def rollout_worker(payload):
    args, policy_weights, worker_seed = payload
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import tensorflow as tf
    from .models import PolicyValueNet

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    tf.random.set_seed(worker_seed)

    env = make_env(
        args.schema,
        args.dataset_number,
        worker_seed,
        args,
        reward_mode=method_reward_mode(args.method),
    )
    model = PolicyValueNet(env.state_dim, env.action_dim, hidden=args.hidden)
    model.set_weights(policy_weights)

    state = env.reset()
    done = False
    states = []
    actions = []
    rewards = []
    values = []
    next_states = []
    masks = []
    total_reward = 0.0
    task_rewards = []

    while not done:
        probs, value = model(
            np.asarray(state, dtype=np.float32).reshape(1, -1), training=False
        )
        mask = env.legal_action_mask()
        policy = _masked_probs(probs.numpy()[0], mask)
        action = int(np.random.choice(env.action_dim, p=policy))
        next_state, reward, done, info = env.step(action)
        task_rewards.append(float(info["common_task_reward"]))

        states.append(state)
        actions.append(action)
        rewards.append(float(reward))
        values.append(float(value.numpy()[0, 0]))
        next_states.append(next_state)
        masks.append(mask)
        total_reward += float(reward)
        state = next_state

    rewards_array = np.asarray(rewards, dtype=np.float32)
    advantages, returns = _gae(
        rewards_array,
        np.asarray(values + [0.0], dtype=np.float32),
        args.gamma,
        args.gae_lambda,
    )
    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": rewards_array,
        "task_rewards": np.asarray(task_rewards, dtype=np.float32),
        "values": np.asarray(values + [0.0], dtype=np.float32),
        "returns": returns.astype(np.float32),
        "advantages": advantages.astype(np.float32),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "masks": np.asarray(masks, dtype=np.float32),
        "episode_reward": float(total_reward),
        "episode_task_reward": float(sum(task_rewards)),
        "query_costs": env.query_costs.snapshot(),
    }


def evaluate_policy(args, model, result_dir: Path, steps: int):
    started = time.perf_counter()
    env = make_env(
        args.schema,
        args.dataset_number,
        args.seed + 777,
        args,
        reward_mode=method_reward_mode(args.method),
    )
    setup_seconds = time.perf_counter() - started
    started = time.perf_counter()
    state = env.reset()
    done = False
    total = 0.0
    common_total = 0.0
    while not done:
        probs, _ = model(
            np.asarray(state, dtype=np.float32).reshape(1, -1), training=False
        )
        action = int(np.argmax(_masked_probs(probs.numpy()[0], env.legal_action_mask())))
        state, reward, done, info = env.step(action)
        common_total += float(info["common_task_reward"])
        total += float(reward)

    actions_repr = [repr(action) for action in env.actions]
    row = {
        "method": args.method,
        "schema": args.schema,
        "dataset": args.dataset_number,
        "seed": args.seed,
        "steps": int(steps),
        "episode_reward": float(total),
        "common_task_reward": common_total,
        "session_setup_seconds": setup_seconds,
        **session_cost_row(env, started, lambda: official_metrics(args.schema, args.dataset_number, env.actions)),
    }
    row = attach_selection_features(row, actions_repr)
    write_json(result_dir / f"{FORMAL_ACTION_PREFIX}{steps}.json", actions_repr)
    write_session(result_dir / f"session_steps{steps}.json", env.actions)
    return row


def _masked_probs(probs, mask):
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=np.float64).reshape(-1)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.maximum(probs, 0.0) * mask
    total = float(probs.sum())
    if total <= 0.0:
        valid = np.flatnonzero(mask > 0)
        fallback = np.zeros_like(probs, dtype=np.float64)
        if valid.size == 0:
            fallback[:] = 1.0 / max(len(fallback), 1)
        else:
            fallback[valid] = 1.0 / valid.size
        return fallback
    return probs / total


def _merge_rollouts(rollouts):
    keys = (
        "states",
        "actions",
        "rewards",
        "task_rewards",
        "returns",
        "advantages",
        "next_states",
        "masks",
    )
    return {
        key: np.concatenate([rollout[key] for rollout in rollouts], axis=0)
        for key in keys
    }


def _a3c_update(model, optimizer, batch, args):
    import tensorflow as tf

    states = tf.convert_to_tensor(batch["states"], dtype=tf.float32)
    actions = tf.convert_to_tensor(batch["actions"], dtype=tf.int32)
    returns = tf.convert_to_tensor(batch["returns"].reshape(-1, 1), dtype=tf.float32)
    advantages = batch["advantages"].astype(np.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages_tensor = tf.convert_to_tensor(
        advantages.reshape(-1, 1), dtype=tf.float32
    )
    masks = tf.convert_to_tensor(batch["masks"], dtype=tf.float32)

    with tf.GradientTape() as tape:
        probs, values = model(states, training=True)
        probs = probs * masks
        probs = probs / tf.maximum(tf.reduce_sum(probs, axis=1, keepdims=True), 1e-8)
        action_one_hot = tf.one_hot(actions, probs.shape[-1])
        action_probs = tf.reduce_sum(probs * action_one_hot, axis=1, keepdims=True)
        logps = tf.math.log(tf.maximum(action_probs, 1e-12))
        policy_loss = -tf.reduce_mean(logps * advantages_tensor)
        value_loss = tf.reduce_mean(tf.square(returns - values))
        entropy = -tf.reduce_mean(
            tf.reduce_sum(probs * tf.math.log(tf.maximum(probs, 1e-12)), axis=1)
        )
        loss = (
            policy_loss
            + float(args.value_coef) * value_loss
            - float(args.entropy_coef) * entropy
        )

    grads = tape.gradient(loss, model.trainable_variables)
    valid_pairs = [
        (gradient, variable)
        for gradient, variable in zip(grads, model.trainable_variables)
        if gradient is not None
    ]
    if float(getattr(args, "max_grad_norm", 0.0)) > 0 and valid_pairs:
        clipped, _ = tf.clip_by_global_norm(
            [gradient for gradient, _ in valid_pairs], float(args.max_grad_norm)
        )
        optimizer.apply_gradients(
            [(gradient, pair[1]) for gradient, pair in zip(clipped, valid_pairs)]
        )
    elif valid_pairs:
        optimizer.apply_gradients(valid_pairs)

    return {
        "policy_loss": float(policy_loss.numpy()),
        "value_loss": float(value_loss.numpy()),
        "entropy": float(entropy.numpy()),
    }


def _gae(rewards, values, gamma, lam):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for index in reversed(range(len(rewards))):
        delta = rewards[index] + float(gamma) * values[index + 1] - values[index]
        gae = delta + float(gamma) * float(lam) * gae
        advantages[index] = gae
    return advantages, advantages + values[:-1]


def _init_train_log(path: Path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=TRAIN_LOG_FIELDS).writeheader()


def _append_train_log(path: Path, row: Dict[str, float]):
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=TRAIN_LOG_FIELDS).writerow(row)


def namespace_from_args(args):
    return SimpleNamespace(**vars(args))
