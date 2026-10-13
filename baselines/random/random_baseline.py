import argparse
import csv
import json
import os
import random

import numpy as np

from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from rl.A3C_2_actors.action_manager import ActionManager
from rl.A3C_2_actors.state_encoder import StateEncoder
from rl.A3C_2_actors.target_set_generator import TargetSetGenerator
from revision_support.galaxy_execution import run_worker_batch


EXPLORATION_COLUMNS = [
    "galaxies.u",
    "galaxies.g",
    "galaxies.r",
    "galaxies.i",
    "galaxies.z",
    "galaxies.petroRad_r",
    "galaxies.redshift",
]

RAW_HEADER = [
    "episode",
    "extrinsic_reward",
    "interestingness",
    "sets_viewed",
    "total_reward",
]

CSV_HEADER = RAW_HEADER + [
    "cumulative_unique_sets_viewed",
    "target_efficiency",
    "cumulative_extrinsic_reward",
    "cumulative_target_efficiency",
]

TRACE_HEADER = [
    "episode",
    "agent_id",
    "step",
    "set_id",
    "step_extrinsic_reward",
    "step_interestingness",
    "operator",
    "parameter",
    "input_set_id",
    "operation_action",
    "z_source",
]

OPERATOR_FAMILIES = ("by_facet", "by_superset", "by_neighbors", "by_distribution")


def build_pipeline():
    return PipelineWithPrecalculatedSets(
        "sdss",
        ["galaxies"],
        data_folder="./app/data/",
        discrete_categories_count=10,
        min_set_size=10,
        exploration_columns=EXPLORATION_COLUMNS,
    )


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def get_precomputed_set_ids(pipeline):
    return np.asarray([int(set_id) for set_id in pipeline.groups.index], dtype=np.int64)


def build_action_manager(pipeline):
    return ActionManager(pipeline, operators=list(OPERATOR_FAMILIES))


def action_family(action_type):
    action_type = str(action_type)
    if action_type.startswith("by_facet"):
        return "by_facet"
    if action_type.startswith("by_superset"):
        return "by_superset"
    if action_type.startswith("by_neighbors"):
        return "by_neighbors"
    if action_type.startswith("by_distribution"):
        return "by_distribution"
    raise ValueError(f"Unsupported random operator family for action: {action_type}")


def build_action_groups(action_manager):
    groups = {family: [] for family in OPERATOR_FAMILIES}
    for action_id, action_type in enumerate(action_manager.set_action_types):
        groups[action_family(action_type)].append(int(action_id))
    missing = [family for family, action_ids in groups.items() if not action_ids]
    if missing:
        raise ValueError(f"Missing action ids for random operator families: {missing}")
    return groups


def sample_random_operator_action(action_groups, rng):
    family = str(rng.choice(OPERATOR_FAMILIES))
    return int(rng.choice(action_groups[family]))


def safe_set_id(dataset, default=-1):
    try:
        set_id = int(dataset.set_id)
    except (AttributeError, TypeError, ValueError):
        return default
    return set_id


def loggable_set_id(dataset):
    set_id = safe_set_id(dataset, default=-1)
    return set_id if set_id >= 0 else None


def dataset_has_predicate(dataset, attribute):
    return any(item.attribute == attribute for item in dataset.predicate.components)


def is_operator_valid(dataset, action_type):
    parts = str(action_type).split("-&-")
    operator = parts[0]
    parameter = parts[1] if len(parts) > 1 else None

    if operator == "by_facet":
        return parameter is not None and not dataset_has_predicate(dataset, parameter)
    if operator == "by_neighbors":
        return parameter is not None and dataset_has_predicate(dataset, parameter)
    if operator in {"by_superset", "by_distribution"}:
        return len(dataset.predicate.components) > 1
    return False


def execute_operator(pipeline, dataset, action_type):
    parts = str(action_type).split("-&-")
    operator = parts[0]
    parameter = parts[1] if len(parts) > 1 else None

    if operator == "by_superset":
        result = pipeline.by_superset(dataset)
    elif operator == "by_distribution":
        result = pipeline.by_distribution(dataset)
    elif operator == "by_facet":
        result = pipeline.by_facet(dataset, [parameter], pipeline.discrete_categories_count)
    elif operator == "by_neighbors":
        result = pipeline.by_neighbors(dataset, [parameter])
    else:
        result = []

    if result is None:
        return []
    return result if isinstance(result, list) else [result]


def build_state_encoder(pipeline, args):
    if args.target_set not in [None, "None"]:
        with open(f"./rl/targets/{args.target_set}.json") as f:
            target_items = set(int(x) for x in json.load(f))
        print(f"Loaded fixed target set '{args.target_set}' with {len(target_items)} objects.")
        return StateEncoder(pipeline, target_items=target_items, target_set_size=2000), target_items

    if args.mode == "scattered":
        target_items = set(
            int(x)
            for x in TargetSetGenerator.get_diverse_target_set(
                number_of_samples=args.target_samples_per_file,
                seed=args.target_seed,
            )
        )
        print(
            "Loaded random scattered target set with "
            f"{len(target_items)} objects. seed={args.target_seed} "
            f"samples_per_file={args.target_samples_per_file}"
        )
        return StateEncoder(pipeline, target_items=target_items, target_set_size=2000), target_items

    target_items = set()
    return StateEncoder(pipeline, target_items=target_items), target_items


def run_worker(worker_args):
    from revision_support.galaxy_results import score_returned_sets, log_returned_sets
    from revision_support.resources import flush_events
    args, worker_id, start_episode, end_episode = worker_args
    random.seed(args.seed + worker_id)
    np.random.seed(args.seed + worker_id)
    rng = np.random.default_rng(args.seed + worker_id)

    pipeline = build_pipeline()
    action_manager = build_action_manager(pipeline)
    action_groups = build_action_groups(action_manager)
    print(
        f"Worker{worker_id} random_operator action count: {len(action_manager.set_action_types)}, "
        f"precomputed set count: {len(get_precomputed_set_ids(pipeline))}"
    )

    state_encoder, target_items_set = build_state_encoder(pipeline, args)

    rows = []
    target_items = sorted(map(int, target_items_set)) if target_items_set else []
    trace_partial_path = f"{args.output_prefix}_{args.baseline}_worker{worker_id}_exploration_trace_partial.csv"
    state_partial_path = f"{args.output_prefix}_{args.baseline}_worker{worker_id}_visited_set_states_partial.csv"
    partial_path = f"{args.output_prefix}_{args.baseline}_worker{worker_id}_partial.csv"
    logged_state_ids = set()

    ensure_parent_dir(trace_partial_path)
    ensure_parent_dir(state_partial_path)
    ensure_parent_dir(partial_path)

    partial_file = None
    partial_writer = None
    if args.write_partial:
        partial_file = open(partial_path, "w", newline="")
        partial_writer = csv.writer(partial_file)
        partial_writer.writerow(RAW_HEADER)
        partial_file.flush()

    trace_file = open(trace_partial_path, "w", newline="")
    trace_writer = csv.writer(trace_file)
    trace_writer.writerow(TRACE_HEADER)

    state_file = open(state_partial_path, "w", newline="")
    state_writer = csv.writer(state_file)
    state_dim = len(state_encoder.set_description)
    state_writer.writerow(["set_id"] + [f"state_{i}" for i in range(state_dim)])

    try:
        for episode in range(start_episode, end_episode + 1):
            state_encoder.reset()
            current_dataset = pipeline.get_dataset()
            ep_ext = 0.0
            ep_int = 0.0
            episode_set_ids = set()

            for step in range(1, args.steps + 1):
                input_set_id = safe_set_id(current_dataset, default=-1)
                operation_action = sample_random_operator_action(action_groups, rng)
                action_type = action_manager.set_action_types[operation_action]
                action_parts = action_type.split("-&-")
                operator = action_parts[0]
                parameter = action_parts[1] if len(action_parts) > 1 else ""
                valid = is_operator_valid(current_dataset, action_type)

                selected_dataset = None
                output_datasets = []
                output_states = []
                if valid:
                    try:
                        output_datasets = [
                            dataset
                            for dataset in execute_operator(pipeline, current_dataset, action_type)
                            if dataset is not None and dataset.data is not None and not dataset.data.empty
                        ]
                    except Exception as exc:
                        raise RuntimeError(f'Galaxy Random operator failure: episode={episode}, step={step}') from exc
                    if output_datasets:
                        selected_dataset = output_datasets[int(rng.integers(0, len(output_datasets)))]
                    else:
                        valid = False

                if valid and selected_dataset is not None:
                    try:
                        output_states, extrinsic, interestingness = score_returned_sets(
                            state_encoder, output_datasets, current_dataset)
                        current_dataset = selected_dataset
                    except Exception as exc:
                        raise RuntimeError(f'Galaxy Random scoring failure: episode={episode}, step={step}') from exc
                else:
                    extrinsic = 0.0
                    interestingness = 0.0

                ep_ext += extrinsic
                ep_int += interestingness
                log_returned_sets(output_datasets, output_states, episode_set_ids, logged_state_ids,
                    state_writer, trace_writer, [episode, worker_id, step],
                    [operator, parameter, input_set_id, operation_action,
                     "random_operator" if valid else "random_operator_invalid"], extrinsic, interestingness)

            trace_file.flush()
            state_file.flush()

            row = {
                "episode": episode,
                "extrinsic_reward": ep_ext,
                "interestingness": ep_int,
                "sets_viewed": len(episode_set_ids),
                "total_reward": ep_ext,
                "set_ids": sorted(episode_set_ids),
            }
            rows.append(row)

            if partial_writer is not None:
                partial_writer.writerow([
                    row["episode"],
                    row["extrinsic_reward"],
                    row["interestingness"],
                    row["sets_viewed"],
                    row["total_reward"],
                ])
                partial_file.flush()

            print(
                f"EP{episode} random_operator worker{worker_id} | "
                f"Ext_R: {ep_ext:.3f} | Int: {ep_int:.3f} | sets_viewed: {len(episode_set_ids)}"
            )
    finally:
        flush_events()
        if partial_file is not None:
            partial_file.close()
        trace_file.close()
        state_file.close()

    return rows, target_items, trace_partial_path, state_partial_path


def episode_ranges(total_episodes, workers):
    workers = max(1, min(workers, total_episodes))
    base = total_episodes // workers
    remainder = total_episodes % workers
    ranges = []
    start = 1
    for worker_id in range(workers):
        count = base + (1 if worker_id < remainder else 0)
        end = start + count - 1
        ranges.append((worker_id, start, end))
        start = end + 1
    return ranges


def run(args):
    config_path = f"{args.output_prefix}_{args.baseline}_config.json"
    ensure_parent_dir(config_path)
    with open(config_path, 'w', encoding='utf-8') as output:
        json.dump(vars(args), output, indent=2)
    tasks = [
        (args, worker_id, start, end)
        for worker_id, start, end in episode_ranges(args.episodes, args.workers)
    ]

    results = run_worker_batch(run_worker, tasks, len(tasks))

    rows = []
    target_items = []
    trace_partial_paths = []
    state_partial_paths = []
    for worker_rows, worker_target_items, trace_partial_path, state_partial_path in results:
        rows.extend(worker_rows)
        if worker_target_items and not target_items:
            target_items = worker_target_items
        trace_partial_paths.append(trace_partial_path)
        state_partial_paths.append(state_partial_path)

    rows.sort(key=lambda row: row["episode"])

    target_items_path = f"{args.output_prefix}_target_items.json"
    if target_items:
        ensure_parent_dir(target_items_path)
        with open(target_items_path, "w") as f:
            json.dump(target_items, f, indent=1)

    csv_path = f"{args.output_prefix}_{args.baseline}_rewards.csv"
    ensure_parent_dir(csv_path)
    global_sets_viewed = set()
    cumulative_extrinsic_reward = 0.0
    output_rows = []

    for row in rows:
        global_sets_viewed.update(row["set_ids"])
        cumulative_extrinsic_reward += float(row["extrinsic_reward"])
        cumulative_unique_sets_viewed = len(global_sets_viewed)
        episode_sets_viewed = int(row["sets_viewed"])
        output_rows.append([
            row["episode"],
            row["extrinsic_reward"],
            row["interestingness"],
            episode_sets_viewed,
            row["total_reward"],
            cumulative_unique_sets_viewed,
            float(row["extrinsic_reward"]) / max(episode_sets_viewed, 1),
            cumulative_extrinsic_reward,
            cumulative_extrinsic_reward / max(cumulative_unique_sets_viewed, 1),
        ])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(output_rows)

    trace_output_path = f"{args.output_prefix}_{args.baseline}_exploration_trace.csv"
    ensure_parent_dir(trace_output_path)
    with open(trace_output_path, "w", newline="") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(TRACE_HEADER)
        for path in trace_partial_paths:
            if not os.path.exists(path):
                continue
            with open(path, newline="") as in_file:
                reader = csv.reader(in_file)
                next(reader, None)
                writer.writerows(reader)
            os.remove(path)

    state_output_path = f"{args.output_prefix}_{args.baseline}_visited_set_states.csv"
    ensure_parent_dir(state_output_path)
    global_state_ids = set()
    state_header_written = False
    with open(state_output_path, "w", newline="") as out_file:
        writer = csv.writer(out_file)
        for path in state_partial_paths:
            if not os.path.exists(path):
                continue
            with open(path, newline="") as in_file:
                reader = csv.reader(in_file)
                header = next(reader, None)
                if header is not None and not state_header_written:
                    writer.writerow(header)
                    state_header_written = True
                for state_row in reader:
                    if not state_row:
                        continue
                    try:
                        set_id = int(state_row[0])
                    except (TypeError, ValueError):
                        continue
                    if set_id in global_state_ids:
                        continue
                    global_state_ids.add(set_id)
                    writer.writerow(state_row)
            os.remove(path)

    print(f"Saved results to {csv_path}")
    print(f"Saved exploration trace to {trace_output_path}")
    print(f"Saved visited set states to {state_output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["random"], required=True)
    parser.add_argument("--target_set", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--mode", type=str, default="scattered")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_seed", type=int, default=None)
    parser.add_argument("--target_samples_per_file", type=int, default=100)
    parser.add_argument("--output_prefix", type=str, default="random_baseline")
    parser.add_argument("--write_partial", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
