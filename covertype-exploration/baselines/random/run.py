import argparse
import csv
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from revision_support.query_service import shared_query_run, set_query_report_path, service_available
from revision_support.query_config import add_query_arguments
from revision_support.query_clients import register_covertype_backend
from revision_support.candidate_pruning import sample_family_action as sample_random_operator_action
from revision_support.covertype_defaults import configure_covertype_runtime

configure_covertype_runtime()

import numpy as np

from covertype_rl.actions import build_action_space
from covertype_rl.candidates import build_candidate_view
from covertype_rl.data import load_covertype
from covertype_rl.environment import FixedSetEnvironment
from revision_support.resources import flush_events
from covertype_rl.fixed_sets import ensure_fixed_universe
from covertype_rl.targets import resolve_target_set


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

OPERATOR_FAMILIES = ("by_facet", "by_superset", "by_neighbors", "by_distribution")


def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


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
    if getattr(args, '_validated_cache', None) is not None:
        from covertype_rl.fixed_sets import load_fixed_universe
        directory, targets = args._validated_cache
        return build_action_space(10, args.n_bins), load_fixed_universe(directory), Path(directory), targets
    root_dir = ROOT
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = root_dir / csv_path

    data = load_covertype(csv_path, n_bins=args.n_bins)
    if service_available():
        register_covertype_backend(data)
    target_items, target_path = resolve_target_set(
        data,
        root_dir=root_dir,
        target_set=args.target_set,
        target_seed=args.target_seed,
        target_size=args.target_size,
    )
    actions = build_action_space(n_continuous=data.continuous.shape[1], n_bins=data.n_bins)
    universe, universe_dir = ensure_fixed_universe(
        data=data,
        actions=actions,
        target_items=target_items,
        target_path=target_path,
        root_dir=root_dir,
        n_sets=args.n_sets,
        seed=args.seed,
        min_set_size=args.min_set_size,
        max_set_size=args.max_set_size,
        preprocess_name=args.preprocess_name,
        force=args.force_preprocess,
        csv_path=csv_path,
    )
    args._validated_cache = (str(universe_dir), target_items)
    return actions, universe, universe_dir, target_items


def action_family(action):
    op = str(action.family)
    if op in OPERATOR_FAMILIES:
        return op
    raise ValueError(f"Unsupported random operator family for action: {action}")


def build_action_groups(actions):
    groups = {family: [] for family in OPERATOR_FAMILIES}
    for action_id, action in enumerate(actions):
        groups[action_family(action)].append(int(action_id))
    missing = [family for family, action_ids in groups.items() if not action_ids]
    if missing:
        raise ValueError(f"Missing action ids for random operator families: {missing}")
    return groups


def run_worker(worker_args):
    args, worker_id, start_episode, end_episode = worker_args
    random.seed(int(args.seed) + worker_id)
    np.random.seed(int(args.seed) + worker_id)
    rng = np.random.default_rng(int(args.seed) + worker_id)

    actions, universe, _universe_dir, _target_items = build_context(args)
    env = FixedSetEnvironment(
        universe,
        actions,
        episode_steps=args.steps,
        seed=int(args.seed) * 1000 + worker_id,
    )
    action_groups = build_action_groups(actions)

    reward_rows = []
    trace_rows = []
    state_rows_by_set = {}

    print(
        f"Worker{worker_id} random_operator action count: {universe.action_dim}, fixed set count: {universe.n_sets}",
        flush=True,
    )

    for episode in range(int(start_episode), int(end_episode) + 1):
        env.reset()
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
        episode_set_ids = set()

        for step in range(1, int(args.steps) + 1):
            _, candidate_actions = build_candidate_view(universe, actions, env.current_set_id)
            valid_ids = [i for group in candidate_actions for i in group] or env.valid_actions().tolist()
            action_id = sample_random_operator_action(action_groups, rng, valid_ids)
            found_items_before_step = dict(env.found_items_with_ratio)
            _state, metrics, set_id, valid = env.step(action_id)
            action = actions[action_id]
            if valid:
                extrinsic = float(metrics["extrinsic_reward"])
                target_hits = int(metrics["target_hits"])
                familiarity = float(metrics["familiarity"])
                counter_curiosity = float(metrics["counter_curiosity"])
            else:
                env.found_items_with_ratio = found_items_before_step
                extrinsic = 0.0
                target_hits = 0
                familiarity = 0.0
                counter_curiosity = 0.0
            interestingness = float(metrics["interestingness"]) if valid else 0.0
            total_reward = extrinsic

            totals["extrinsic_reward"] += extrinsic
            totals["target_hits"] += target_hits
            totals["interestingness"] += interestingness
            totals["familiarity"] += familiarity
            totals["counter_curiosity"] += counter_curiosity
            totals["total_reward"] += total_reward
            totals["valid_steps"] += int(valid)
            episode_set_ids.add(int(set_id))

            trace_rows.append(
                {
                    "episode": int(episode),
                    "step": int(step),
                    "set_id": int(set_id),
                    "action_id": int(action_id),
                    "operator": str(action.family),
                    "parameter": str(action.label),
                    "step_extrinsic_reward": extrinsic,
                    "step_interestingness": interestingness,
                    "step_counter_curiosity": counter_curiosity,
                    "step_coherency": 0.0,
                    "step_diversity": 0.0,
                    "step_total_reward": total_reward,
                    "set_size": int(metrics["set_size"]),
                    "sample_size": int(metrics["sample_size"]),
                    "valid": int(valid),
                }
            )

            if int(set_id) not in state_rows_by_set:
                state_rows_by_set[int(set_id)] = list(np.asarray(metrics["state"], dtype=np.float32))

        reward_rows.append(
            {
                "episode": int(episode),
                "extrinsic_reward": float(totals["extrinsic_reward"]),
                "mean_step_extrinsic": float(totals["extrinsic_reward"]) / max(int(args.steps), 1),
                "target_hits": int(totals["target_hits"]),
                "interestingness": float(totals["interestingness"]),
                "familiarity": float(totals["familiarity"]),
                "counter_curiosity": float(totals["counter_curiosity"]),
                "coherency": 0.0,
                "diversity": 0.0,
                "total_reward": float(totals["total_reward"]),
                "sets_viewed": int(len(episode_set_ids)),
                "valid_steps": int(totals["valid_steps"]),
                "set_ids": sorted(episode_set_ids),
            }
        )

        flush_events()
        print(
            f"EP{episode} random_operator worker{worker_id} | "
            f"Ext_R: {totals['extrinsic_reward']:.6f} | sets_viewed: {len(episode_set_ids)}",
            flush=True,
        )

    return reward_rows, trace_rows, state_rows_by_set, sorted(map(int, _target_items))


def output_prefix_path(args):
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or f"random_seed{args.seed}"
    return output_dir / prefix


def write_rewards(path, rows):
    ensure_parent_dir(path)
    global_sets_viewed = set()
    cumulative_extrinsic_reward = 0.0

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(REWARD_HEADER)
        for row in sorted(rows, key=lambda item: int(item["episode"])):
            global_sets_viewed.update(int(set_id) for set_id in row.pop("set_ids"))
            cumulative_extrinsic_reward += float(row["extrinsic_reward"])
            cumulative_unique_sets_viewed = len(global_sets_viewed)
            sets_viewed = int(row["sets_viewed"])
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
                    cumulative_unique_sets_viewed,
                    float(row["extrinsic_reward"]) / max(sets_viewed, 1),
                    cumulative_extrinsic_reward,
                    cumulative_extrinsic_reward / max(cumulative_unique_sets_viewed, 1),
                    int(row["valid_steps"]),
                ]
            )


def write_trace(path, rows):
    ensure_parent_dir(path)
    rows = sorted(rows, key=lambda item: (int(item["episode"]), int(item["step"])))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRACE_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def write_states(path, states_by_set):
    ensure_parent_dir(path)
    state_items = sorted((int(set_id), state) for set_id, state in states_by_set.items())
    state_dim = len(state_items[0][1]) if state_items else 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["set_id"] + [f"state_{idx}" for idx in range(state_dim)])
        for set_id, state in state_items:
            writer.writerow([set_id] + list(state))


@shared_query_run
def run(args):
    args.baseline = "random"

    args._validated_cache = None
    prefix = output_prefix_path(args)
    set_query_report_path(Path(f"{prefix}_query_scheduler.json"))
    Path(f"{prefix}_random_config.json").write_text(json.dumps(
        {k: v for k, v in vars(args).items() if not k.startswith('_')}, indent=2), encoding='utf-8')
    build_context(args)
    tasks = [
        (args, worker_id, start, end)
        for worker_id, start, end in episode_ranges(args.episodes, args.workers)
    ]

    if int(args.workers) == 1:
        results = [run_worker(tasks[0])]
    else:
        with mp.Pool(processes=len(tasks)) as pool:
            results = pool.map(run_worker, tasks)

    reward_rows = []
    trace_rows = []
    states_by_set = {}
    target_items = []
    for worker_rewards, worker_trace, worker_states, worker_target_items in results:
        reward_rows.extend(worker_rewards)
        trace_rows.extend(worker_trace)
        for set_id, state in worker_states.items():
            states_by_set.setdefault(int(set_id), state)
        if worker_target_items and not target_items:
            target_items = worker_target_items

    prefix = output_prefix_path(args)
    reward_path = Path(f"{prefix}_random_rewards.csv")
    trace_path = Path(f"{prefix}_random_exploration_trace.csv")
    states_path = Path(f"{prefix}_random_visited_set_states.csv")
    target_items_path = Path(f"{prefix}_target_items.json")

    write_rewards(reward_path, reward_rows)
    write_trace(trace_path, trace_rows)
    write_states(states_path, states_by_set)

    if target_items:
        ensure_parent_dir(target_items_path)
        with open(target_items_path, "w", encoding="utf-8") as f:
            json.dump(target_items, f, indent=1)

    print(f"Saved results to {reward_path}")
    print(f"Saved exploration trace to {trace_path}")
    print(f"Saved visited set states to {states_path}")
    return {
        "rewards": str(reward_path),
        "trace": str(trace_path),
        "states": str(states_path),
        "target_items": str(target_items_path) if target_items else "",
    }


def build_parser():
    from revision_support.covertype_defaults import CovertypeArgumentParser
    parser = CovertypeArgumentParser(
        description=''
    )
    parser.add_argument("--baseline", choices=["random"], default="random")
    parser.add_argument("--csv_path", default="covertype.csv")
    parser.add_argument("--target_set", default=None)
    parser.add_argument("--target_seed", type=int, default=None)
    parser.add_argument("--target_size", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--n_sets", type=int, default=100000)
    parser.add_argument("--min_set_size", type=int, default=10)
    parser.add_argument("--max_set_size", type=int, default=200000)
    parser.add_argument("--preprocess_name", default=None)
    parser.add_argument("--force_preprocess", action="store_true")
    parser.add_argument("--output_prefix", default=None)
    parser.add_argument("--output_dir", default="outputs/random")
    add_query_arguments(parser)
    return parser


def main():
    paths = run(build_parser().parse_args())
    print("Wrote:")
    for key, value in paths.items():
        if value:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
