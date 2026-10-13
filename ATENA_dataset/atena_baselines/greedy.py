import logging
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from atena_common.costs import session_cost_row, write_costs, write_session
from .env import make_env
from .evaluate import official_metrics, write_json, write_metrics_csv


@contextmanager
def _suppress_display_warnings():
    display_logger = logging.getLogger("atena.simulation.display")
    original_level = display_logger.level
    display_logger.setLevel(max(original_level, logging.ERROR))
    try:
        yield
    finally:
        display_logger.setLevel(original_level)


def select_greedy_action(env) -> Tuple[int, float, int]:
    legal_indices = np.flatnonzero(env.legal_action_mask() > 0)
    if len(legal_indices) == 0:
        raise RuntimeError("Greedy found no legal actions in the current state")

    best_action = None
    best_reward = -math.inf
    evaluated = 0
    with _suppress_display_warnings():
        for action_index in legal_indices:
            action_index = int(action_index)
            reward, _ = env.preview_step(action_index)
            evaluated += 1
            if not math.isfinite(reward):
                continue
            if best_action is None or reward > best_reward:
                best_action = action_index
                best_reward = float(reward)

    if best_action is None:
        raise RuntimeError("Greedy found no legal candidate with a finite reward")
    return best_action, best_reward, evaluated


def run_greedy(args, result_dir: Path) -> List[Dict[str, object]]:
    started = time.perf_counter()
    env = make_env(
        args.schema,
        args.dataset_number,
        args.seed,
        args,
        reward_mode="official_compound",
    )
    setup_seconds = time.perf_counter() - started
    started = time.perf_counter()
    env.reset()
    done = False
    episode_reward = 0.0
    candidate_evaluations = 0
    while not done:
        action_index, preview_reward, evaluated = select_greedy_action(env)
        _, committed_reward, done, _ = env.step(action_index)
        if not math.isclose(preview_reward, committed_reward, rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError(
                f"Greedy preview reward {preview_reward} did not match committed reward {committed_reward}"
            )
        episode_reward += float(committed_reward)
        candidate_evaluations += int(evaluated)

    with _suppress_display_warnings():
        metrics = session_cost_row(env, started, lambda: official_metrics(args.schema, args.dataset_number, env.actions))
    runtime_seconds = setup_seconds + time.perf_counter() - started
    actions_repr = [repr(action) for action in env.actions]
    row = {
        "method": "greedy",
        "schema": args.schema,
        "dataset": int(args.dataset_number),
        "seed": int(args.seed),
        "steps": 0,
        "episode": 0,
        "episode_reward": float(episode_reward),
        "common_task_reward": float(episode_reward),
        "session_setup_seconds": setup_seconds,
        "runtime_seconds": float(runtime_seconds),
        "candidate_evaluations": int(candidate_evaluations),
        **metrics,
    }
    write_json(result_dir / "actions_steps0.json", actions_repr)
    write_metrics_csv(result_dir / "eval_metrics.csv", [row])
    write_json(result_dir / "final_metrics.json", row)
    write_session(result_dir / "session_steps0.json", env.actions)
    write_costs(result_dir, [row])
    return [row]
