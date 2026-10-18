import argparse
import csv
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path
from types import SimpleNamespace

COVERTYPE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
for import_root in (COVERTYPE_ROOT, REPOSITORY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from revision_support.covertype_defaults import configure_covertype_runtime
from revision_support.query_service import shared_query_run, set_query_report_path
from revision_support.query_config import add_query_arguments
from revision_support.query_clients import register_covertype_backend

configure_covertype_runtime()

import numpy as np
from covertype_rl.candidates import build_candidate_view

from baselines.greedy_eda.covertype import (
    WorkerExplorationMemory,
    choose_count_balanced_candidate,
)
from baselines.greedy_eda.policy import Candidate, choose_operation
from revision_support.resources import flush_events


REWARD_HEADER = [
    "episode",
    "extrinsic_reward",
    "mean_step_extrinsic",
    "target_hits",
    "interestingness",
    "familiarity",
    "counter_curiosity",
    "coherency",
    "diversity",
    "total_reward",
    "sets_viewed",
    "cumulative_unique_sets_viewed",
    "target_efficiency",
    "cumulative_extrinsic_reward",
    "cumulative_target_efficiency",
    "valid_steps",
]

TRACE_HEADER = [
    "episode",
    "step",
    "set_id",
    "action_id",
    "operator",
    "parameter",
    "step_extrinsic_reward",
    "step_interestingness",
    "step_counter_curiosity",
    "step_coherency",
    "step_diversity",
    "step_total_reward",
    "set_size",
    "sample_size",
    "valid",
]

OPERATOR_FAMILIES = (
    "by_facet",
    "by_superset",
    "by_neighbors",
    "by_distribution",
)

METHOD_CONFIG = {
    "method": "greedy_eda",
    "target_blind": True,
    "uses_extrinsic_for_selection": False,
    "interestingness_weight": 1.0,
    "coherency_weight": 1.0,
    "diversity_weight": 1.0,
    "diversity_eta": 0.1,
    "selection_mode": "count_balanced_top_k",
    "cross_episode_memory": True,
    "memory_scope": "worker",
    "rank_weighting": "linear",
}


def _js_divergence(left, right):
    left = np.nan_to_num(np.asarray(left, dtype=np.float64), nan=0.0)
    right = np.nan_to_num(np.asarray(right, dtype=np.float64), nan=0.0)
    left_sum = float(left.sum())
    right_sum = float(right.sum())
    if left_sum <= 0.0 or right_sum <= 0.0:
        return 0.0
    left = left / left_sum
    right = right / right_sum
    midpoint = 0.5 * (left + right)

    def kl_divergence(values, reference):
        mask = values > 0.0
        return float(
            np.sum(values[mask] * np.log(values[mask] / reference[mask]))
        )

    return 0.5 * kl_divergence(left, midpoint) + 0.5 * kl_divergence(
        right, midpoint
    )


def candidate_interestingness(
    current_state,
    candidate_state,
    current_size,
    candidate_size,
):
    current_state = np.asarray(current_state, dtype=np.float64)
    candidate_state = np.asarray(candidate_state, dtype=np.float64)
    current_parts = (
        current_state[20:27],
        current_state[27:31],
        current_state[31:71],
    )
    candidate_parts = (
        candidate_state[20:27],
        candidate_state[27:31],
        candidate_state[31:71],
    )
    divergences = [
        _js_divergence(candidate, current)
        for candidate, current in zip(candidate_parts, current_parts)
        if float(np.asarray(candidate).sum()) > 0.0
        and float(np.asarray(current).sum()) > 0.0
    ]
    if not divergences:
        return 0.0
    minimum = min(int(candidate_size), int(current_size))
    maximum = max(int(candidate_size), int(current_size))
    ratio = float(minimum) / float(maximum) if maximum > 0 else 0.0
    compactness = max(0.0, 1.0 - abs(ratio - 0.5) * 2.0)
    return float(np.mean(divergences) * compactness)


def visible_candidates(env, interestingness_by_slot):
    slots = []
    candidates = []
    for slot, set_id in enumerate(env.candidate_set_ids):
        if int(set_id) < 0:
            continue
        slots.append(slot)
        candidates.append(
            Candidate(
                candidate_id=int(set_id),
                state=env.universe.state_for_set(int(set_id)),
                interestingness=float(interestingness_by_slot[slot]),
            )
        )
    return slots, candidates


def execute_selected_step(env, action_id):
    return env.step(int(action_id))


def episode_ranges(total_episodes, workers):
    workers = max(1, min(int(workers), int(total_episodes)))
    base = int(total_episodes) // workers
    remainder = int(total_episodes) % workers
    ranges = []
    start = 1
    for worker_id in range(workers):
        count = base + (1 if worker_id < remainder else 0)
        end = start + count - 1
        ranges.append((worker_id, start, end))
        start = end + 1
    return ranges


def build_context(args):
    from covertype_rl.actions import build_action_space
    from covertype_rl.fixed_sets import load_fixed_universe
    from covertype_rl.greedy_preprocessing import validate_official_greedy_preprocessing
    from covertype_rl.targets import load_target_items
    from covertype_rl.cache_contract import validate_cache

    if getattr(args, '_validated_cache', None) is not None:
        directory, target_items, metadata = args._validated_cache
        return build_action_space(10, 10), load_fixed_universe(directory), Path(directory), target_items, metadata

    actions = build_action_space(n_continuous=10, n_bins=10)
    contract = validate_official_greedy_preprocessing(
        root_dir=COVERTYPE_ROOT,
        target_set=args.target_set,
        seed=args.seed,
        preprocess_name=args.preprocess_name,
        action_labels=[action.label for action in actions],
    )
    target_path = COVERTYPE_ROOT / "targets" / f"{args.target_set}.json"
    if not target_path.is_file():
        raise FileNotFoundError(f"Missing fixed target set: {target_path}")
    target_items = load_target_items(target_path)
    universe = validate_cache(contract.universe_dir, COVERTYPE_ROOT / 'covertype.csv', target_path,
                              [action.label for action in actions],
                              dict(seed=int(args.seed), requested_sets=100000, min_set_size=10,
                                   max_set_size=200000, n_bins=10))
    root_targets = sorted(
        map(int, universe.targets_for_set(int(universe.metadata["root_set_id"])))
    )
    if root_targets != target_items:
        raise ValueError(
            "Fixed preprocessing root targets do not match the fixed target JSON"
        )
    args._validated_cache = (str(contract.universe_dir), target_items, contract.metadata)
    return actions, universe, contract.universe_dir, target_items, contract.metadata


def run_worker(worker_args):
    from covertype_rl.environment import FixedSetEnvironment

    args, worker_id, start_episode, end_episode = worker_args
    random.seed(int(args.seed) + worker_id)
    np.random.seed(int(args.seed) + worker_id)
    rng = np.random.default_rng(int(args.seed) + worker_id)
    actions, universe, _universe_dir, target_items, _metadata = build_context(args)
    env = FixedSetEnvironment(
        universe,
        actions,
        episode_steps=args.steps,
        seed=int(args.seed) * 1000 + worker_id,
    )
    family_by_action = {
        action_id: str(action.family)
        for action_id, action in enumerate(actions)
    }

    reward_rows = []
    trace_rows = []
    state_rows_by_set = {}
    worker_memory = WorkerExplorationMemory()
    worker_memory.register_root(env.root_set_id)
    for episode in range(int(start_episode), int(end_episode) + 1):
        current_state = np.asarray(env.reset(), dtype=np.float32)
        current_set_id = int(env.current_set_id)
        current_size = int(universe.size_for_set(current_set_id))
        history_states = [current_state]
        visited_ids = {current_set_id}
        episode_set_ids = set()
        family_counts = {}
        action_counts = {}
        totals = {
            "extrinsic_reward": 0.0,
            "target_hits": 0,
            "interestingness": 0.0,
            "familiarity": 0.0,
            "counter_curiosity": 0.0,
            "coherency": 0.0,
            "diversity": 0.0,
            "total_reward": 0.0,
            "valid_steps": 0,
        }

        for step in range(1, int(args.steps) + 1):
            candidate_ids, candidate_action_ids = build_candidate_view(
                universe,
                actions,
                current_set_id,
                candidate_slots=args.candidate_slots,
            )
            interestingness_by_slot = [
                candidate_interestingness(
                    current_state,
                    universe.state_for_set(set_id),
                    current_size,
                    universe.size_for_set(set_id),
                )
                if int(set_id) >= 0
                else 0.0
                for set_id in candidate_ids
            ]
            view = SimpleNamespace(
                universe=universe,
                candidate_set_ids=candidate_ids,
            )
            source_slots, candidates = visible_candidates(
                view,
                interestingness_by_slot,
            )
            if candidates:
                selection = choose_count_balanced_candidate(
                    candidates,
                    current_state=current_state,
                    history_states=history_states,
                    memory=worker_memory,
                    rng=rng,
                    top_k=args.selection_top_k,
                )
                slot = source_slots[selection.index]
                legal_action_ids = candidate_action_ids[slot]
                coherency = float(selection.components["coherency"])
                diversity = float(selection.components["diversity"])
            else:
                legal_action_ids = env.valid_actions().astype(int).tolist()
                coherency = 0.0
                diversity = 0.0

            action_id = choose_operation(
                legal_action_ids,
                family_by_action,
                family_counts,
                action_counts,
                rng,
            )
            family = family_by_action[action_id]
            family_counts[family] = family_counts.get(family, 0) + 1
            action_counts[action_id] = action_counts.get(action_id, 0) + 1

            next_state, metrics, next_set_id, valid = execute_selected_step(
                env,
                action_id,
            )
            action = actions[action_id]
            next_state = np.asarray(next_state, dtype=np.float32)
            extrinsic = float(metrics["extrinsic_reward"]) if valid else 0.0
            target_hits = int(metrics["target_hits"]) if valid else 0
            interestingness = float(metrics["interestingness"]) if valid else 0.0
            familiarity = float(metrics["familiarity"]) if valid else 0.0
            counter_curiosity = (
                float(metrics["counter_curiosity"]) if valid else 0.0
            )

            totals["extrinsic_reward"] += extrinsic
            totals["target_hits"] += target_hits
            totals["interestingness"] += interestingness
            totals["familiarity"] += familiarity
            totals["counter_curiosity"] += counter_curiosity
            totals["coherency"] += coherency
            totals["diversity"] += diversity
            totals["total_reward"] += extrinsic
            totals["valid_steps"] += int(valid)
            episode_set_ids.add(int(next_set_id))

            trace_rows.append(
                {
                    "episode": int(episode),
                    "step": int(step),
                    "set_id": int(next_set_id),
                    "action_id": int(action_id),
                    "operator": str(action.family),
                    "parameter": str(action.label),
                    "step_extrinsic_reward": extrinsic,
                    "step_interestingness": interestingness,
                    "step_counter_curiosity": counter_curiosity,
                    "step_coherency": coherency,
                    "step_diversity": diversity,
                    "step_total_reward": extrinsic,
                    "set_size": int(metrics["set_size"]),
                    "sample_size": int(metrics["sample_size"]),
                    "valid": int(valid),
                }
            )
            state_rows_by_set.setdefault(
                int(next_set_id),
                next_state.astype(float).tolist(),
            )
            current_state = next_state
            current_set_id = int(next_set_id)
            current_size = int(metrics["set_size"])
            history_states.append(next_state)
            visited_ids.add(int(next_set_id))
            if valid:
                worker_memory.record_visit(next_set_id)

        reward_rows.append(
            {
                "episode": int(episode),
                "extrinsic_reward": float(totals["extrinsic_reward"]),
                "mean_step_extrinsic": float(totals["extrinsic_reward"])
                / max(int(args.steps), 1),
                "target_hits": int(totals["target_hits"]),
                "interestingness": float(totals["interestingness"]),
                "familiarity": float(totals["familiarity"]),
                "counter_curiosity": float(totals["counter_curiosity"]),
                "coherency": float(totals["coherency"]),
                "diversity": float(totals["diversity"]),
                "total_reward": float(totals["total_reward"]),
                "sets_viewed": len(episode_set_ids),
                "valid_steps": int(totals["valid_steps"]),
                "set_ids": sorted(episode_set_ids),
            }
        )
        flush_events()
        print(
            f"EP{episode} greedy_eda worker{worker_id} | "
            f"Ext_R: {totals['extrinsic_reward']:.6f} | "
            f"sets_viewed: {len(episode_set_ids)}",
            flush=True,
        )
    return reward_rows, trace_rows, state_rows_by_set, sorted(map(int, target_items))


def _output_prefix(args):
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = COVERTYPE_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / (args.output_prefix or f"greedy_eda_seed{args.seed}")


def _write_rewards(path, rows):
    global_sets = set()
    cumulative_ext = 0.0
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(REWARD_HEADER)
        for row in sorted(rows, key=lambda item: int(item["episode"])):
            set_ids = row["set_ids"]
            global_sets.update(int(set_id) for set_id in set_ids)
            cumulative_ext += float(row["extrinsic_reward"])
            sets_viewed = int(row["sets_viewed"])
            cumulative_sets = len(global_sets)
            writer.writerow(
                [
                    int(row["episode"]),
                    float(row["extrinsic_reward"]),
                    float(row["mean_step_extrinsic"]),
                    int(row["target_hits"]),
                    float(row["interestingness"]),
                    float(row["familiarity"]),
                    float(row["counter_curiosity"]),
                    float(row["coherency"]),
                    float(row["diversity"]),
                    float(row["total_reward"]),
                    sets_viewed,
                    cumulative_sets,
                    float(row["extrinsic_reward"]) / max(sets_viewed, 1),
                    cumulative_ext,
                    cumulative_ext / max(cumulative_sets, 1),
                    int(row["valid_steps"]),
                ]
            )


def _write_trace(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=TRACE_HEADER)
        writer.writeheader()
        writer.writerows(
            sorted(rows, key=lambda item: (item["episode"], item["step"]))
        )


def _write_states(path, states_by_set):
    items = sorted((int(set_id), state) for set_id, state in states_by_set.items())
    state_dim = len(items[0][1]) if items else 0
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["set_id"] + [f"state_{index}" for index in range(state_dim)])
        for set_id, state in items:
            writer.writerow([set_id] + list(state))


@shared_query_run
def run(args):
    args.baseline = "greedy_eda"
    from covertype_rl.data import load_covertype
    register_covertype_backend(load_covertype(COVERTYPE_ROOT / 'covertype.csv', n_bins=10))
    set_query_report_path(Path(f"{_output_prefix(args)}_query_scheduler.json"))
    if int(args.selection_top_k) < 1 or int(args.selection_top_k) > int(args.candidate_slots):
        raise ValueError("selection_top_k must be between 1 and candidate_slots")
    _actions, _universe, universe_dir, _targets, preprocessing_metadata = build_context(args)
    tasks = [
        (args, worker_id, start, end)
        for worker_id, start, end in episode_ranges(args.episodes, args.workers)
    ]
    if len(tasks) == 1:
        results = [run_worker(tasks[0])]
    else:
        with mp.Pool(processes=len(tasks)) as pool:
            results = pool.map(run_worker, tasks)

    reward_rows = []
    trace_rows = []
    states_by_set = {}
    target_items = []
    for worker_rewards, worker_trace, worker_states, worker_targets in results:
        reward_rows.extend(worker_rewards)
        trace_rows.extend(worker_trace)
        for set_id, state in worker_states.items():
            states_by_set.setdefault(int(set_id), state)
        if worker_targets and not target_items:
            target_items = worker_targets
    episodes = sorted(int(row["episode"]) for row in reward_rows)
    if episodes != list(range(1, int(args.episodes) + 1)):
        raise ValueError("Greedy EDA worker merge produced missing or duplicate episodes")
    expected_trace_rows = int(args.episodes) * int(args.steps)
    if len(trace_rows) != expected_trace_rows:
        raise ValueError(
            f"Greedy EDA trace row mismatch: expected {expected_trace_rows}, "
            f"got {len(trace_rows)}"
        )
    trajectories = {}
    for row in trace_rows:
        trajectories.setdefault(int(row["episode"]), []).append(row)
    trajectory_ids = set()
    for episode, rows in trajectories.items():
        ordered = sorted(rows, key=lambda item: int(item["step"]))
        steps = [int(item["step"]) for item in ordered]
        if steps != list(range(1, int(args.steps) + 1)):
            raise ValueError(f"Greedy EDA episode {episode} has incomplete steps")
        trajectory_ids.add(tuple(int(item["set_id"]) for item in ordered))
    ordered_rewards = sorted(reward_rows, key=lambda item: int(item["episode"]))
    first_sets = set(map(int, ordered_rewards[0]["set_ids"]))
    all_sets = set()
    for row in ordered_rewards:
        all_sets.update(map(int, row["set_ids"]))
    if int(args.episodes) > 1 and len(trajectory_ids) < 2:
        raise ValueError("Greedy EDA produced identical episode trajectories")
    if int(args.episodes) > 1 and len(all_sets) <= len(first_sets):
        raise ValueError("Greedy EDA cumulative unique coverage did not grow")

    prefix = _output_prefix(args)
    reward_path = Path(f"{prefix}_greedy_eda_rewards.csv")
    trace_path = Path(f"{prefix}_greedy_eda_exploration_trace.csv")
    states_path = Path(f"{prefix}_greedy_eda_visited_set_states.csv")
    config_path = Path(f"{prefix}_greedy_eda_config.json")
    target_path = Path(f"{prefix}_target_items.json")
    _write_rewards(reward_path, reward_rows)
    _write_trace(trace_path, trace_rows)
    _write_states(states_path, states_by_set)
    config = dict(vars(args))
    config.update(METHOD_CONFIG)
    config["fixed_universe_dir"] = str(universe_dir.resolve())
    config["fixed_universe_sets"] = int(preprocessing_metadata["n_sets"])
    config["preprocessing_metadata"] = preprocessing_metadata
    with open(config_path, "w", encoding="utf-8") as output:
        json.dump(config, output, indent=2, sort_keys=True)
    if target_items:
        with open(target_path, "w", encoding="utf-8") as output:
            json.dump(target_items, output, indent=1)
    paths = {
        "rewards": str(reward_path),
        "trace": str(trace_path),
        "states": str(states_path),
        "config": str(config_path),
        "target_items": str(target_path) if target_items else "",
    }
    for label, path in paths.items():
        if path:
            print(f"Saved {label} to {path}")
    return paths


def build_parser():
    from revision_support.covertype_defaults import CovertypeArgumentParser
    parser = CovertypeArgumentParser(
        description=''
    )
    parser.add_argument("--baseline", choices=["greedy_eda"], default="greedy_eda")
    parser.add_argument("--target_set", default=None)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--preprocess_name", default=None)
    parser.add_argument("--candidate_slots", type=int, default=10)
    parser.add_argument("--selection_top_k", type=int, default=3)
    parser.add_argument("--output_prefix", default=None)
    parser.add_argument("--output_dir", default="outputs/GreedyEDA_count_balanced")
    add_query_arguments(parser)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
