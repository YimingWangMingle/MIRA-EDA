from revision_support.query_service import shared_query_run, set_query_report_path
from revision_support.query_config import add_query_arguments
import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np

from .policy import Candidate, choose_candidate, choose_operation
from revision_support.galaxy_execution import run_worker_batch


METHOD_CONFIG = {
    "method": "greedy_eda",
    "target_blind": True,
    "uses_extrinsic_for_selection": False,
    "interestingness_weight": 1.0,
    "coherency_weight": 1.0,
    "diversity_weight": 1.0,
    "diversity_eta": 0.1,
    "repeat_penalty": 1.0,
    "cross_episode_memory": False,
}


def _random_helpers():
    from baselines.random import random_baseline

    return random_baseline


def _dataset_nonempty(dataset):
    if dataset is None:
        return False
    data = getattr(dataset, "data", None)
    if data is None:
        return False
    if hasattr(data, "empty"):
        return not bool(data.empty)
    try:
        return len(data) > 0
    except TypeError:
        return False


def _safe_candidate_id(dataset, fallback):
    try:
        set_id = int(getattr(dataset, "set_id", fallback))
    except (TypeError, ValueError):
        return int(fallback)
    return set_id if set_id >= 0 else int(fallback)


def action_family(action_type):
    return str(action_type).split("-&-", 1)[0]


def legal_operation_ids(dataset, action_types, validity_fn):
    return [
        action_id
        for action_id, action_type in enumerate(action_types)
        if validity_fn(dataset, action_type)
    ]


def execute_selected_operation(pipeline, dataset, action_type, execute_fn):
    result = execute_fn(pipeline, dataset, action_type)
    if result is None:
        return []
    datasets = result if isinstance(result, list) else [result]
    return [dataset for dataset in datasets if _dataset_nonempty(dataset)]


def preview_candidates(datasets, parent_dataset, state_encoder):
    indices = []
    candidates = []
    for index, dataset in enumerate(datasets):
        if not _dataset_nonempty(dataset):
            continue
        state, _extrinsic, interestingness = state_encoder.encode_dataset(
            dataset,
            parent_dataset=parent_dataset,
            get_reward=False,
            get_interestingness=True,
        )
        indices.append(index)
        candidates.append(
            Candidate(
                candidate_id=_safe_candidate_id(dataset, -(index + 1)),
                state=np.asarray(state, dtype=np.float64),
                interestingness=float(interestingness),
            )
        )
    return indices, candidates


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


def _write_config(args):
    helpers = _random_helpers()
    path = f"{args.output_prefix}_{args.baseline}_config.json"
    helpers.ensure_parent_dir(path)
    config = dict(vars(args))
    config.update(METHOD_CONFIG)
    with open(path, "w", encoding="utf-8") as output:
        json.dump(config, output, indent=2, sort_keys=True)
    return path


def run_worker(worker_args):
    from revision_support.galaxy_results import score_returned_sets, log_returned_sets
    from revision_support.resources import flush_events
    helpers = _random_helpers()
    args, worker_id, start_episode, end_episode = worker_args
    random.seed(int(args.seed) + worker_id)
    np.random.seed(int(args.seed) + worker_id)
    rng = np.random.default_rng(int(args.seed) + worker_id)

    pipeline = helpers.build_pipeline()
    action_manager = helpers.build_action_manager(pipeline)
    state_encoder, target_items_set = helpers.build_state_encoder(pipeline, args)
    action_types = list(action_manager.set_action_types)
    family_by_action = {
        action_id: action_family(action_type)
        for action_id, action_type in enumerate(action_types)
    }

    rows = []
    target_items = sorted(map(int, target_items_set)) if target_items_set else []
    trace_partial_path = (
        f"{args.output_prefix}_{args.baseline}_worker{worker_id}_exploration_trace_partial.csv"
    )
    state_partial_path = (
        f"{args.output_prefix}_{args.baseline}_worker{worker_id}_visited_set_states_partial.csv"
    )
    partial_path = f"{args.output_prefix}_{args.baseline}_worker{worker_id}_partial.csv"
    for path in (trace_partial_path, state_partial_path, partial_path):
        helpers.ensure_parent_dir(path)

    trace_file = open(trace_partial_path, "w", newline="")
    trace_writer = csv.writer(trace_file)
    trace_writer.writerow(helpers.TRACE_HEADER)
    state_file = open(state_partial_path, "w", newline="")
    state_writer = csv.writer(state_file)
    state_dim = len(state_encoder.set_description)
    state_writer.writerow(["set_id"] + [f"state_{index}" for index in range(state_dim)])
    partial_file = None
    partial_writer = None
    if args.write_partial:
        partial_file = open(partial_path, "w", newline="")
        partial_writer = csv.writer(partial_file)
        partial_writer.writerow(helpers.RAW_HEADER)

    logged_state_ids = set()
    try:
        for episode in range(start_episode, end_episode + 1):
            state_encoder.reset()
            current_dataset = pipeline.get_dataset()
            current_state, _unused_ext, _unused_int = state_encoder.encode_dataset(
                current_dataset,
                get_reward=False,
            )
            history_states = [np.asarray(current_state, dtype=np.float64)]
            visited_ids = set()
            episode_set_ids = set()
            family_counts = {}
            action_counts = {}
            ep_ext = 0.0
            ep_int = 0.0

            for step in range(1, int(args.steps) + 1):
                input_set_id = helpers.safe_set_id(current_dataset, default=-1)
                legal_ids = np.flatnonzero(action_manager.fix_possible_operation_action_probs(
                    current_dataset, np.ones(len(action_types)))).tolist()
                operation_action = choose_operation(
                    legal_ids,
                    family_by_action,
                    family_counts,
                    action_counts,
                    rng,
                )
                action_type = action_types[operation_action]
                family = family_by_action[operation_action]
                action_parts = action_type.split("-&-", 1)
                parameter = action_parts[1] if len(action_parts) > 1 else ""
                family_counts[family] = family_counts.get(family, 0) + 1
                action_counts[operation_action] = action_counts.get(operation_action, 0) + 1

                try:
                    output_datasets = execute_selected_operation(
                        pipeline,
                        current_dataset,
                        action_type,
                        helpers.execute_operator,
                    )
                    source_indices, candidates = preview_candidates(
                        output_datasets,
                        current_dataset,
                        state_encoder,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Galaxy Greedy EDA candidate failure: episode={episode}, "
                        f"step={step}, input_set_id={input_set_id}, "
                        f"operation_action={operation_action}"
                    ) from exc

                selected_dataset = None
                selected_state = None
                extrinsic = 0.0
                interestingness = 0.0
                source = "greedy_eda_empty"
                output_states, extrinsic, interestingness = score_returned_sets(
                    state_encoder, output_datasets, current_dataset)
                if candidates:
                    selection = choose_candidate(
                        candidates,
                        current_state=current_state,
                        history_states=history_states,
                        visited_ids=visited_ids,
                        rng=rng,
                    )
                    selected_dataset = output_datasets[source_indices[selection.index]]
                    selected_state = output_states[source_indices[selection.index]]
                    current_dataset = selected_dataset
                    current_state = np.asarray(selected_state, dtype=np.float64)
                    history_states.append(current_state)
                    visited_ids.add(int(selection.candidate_id))
                    source = "greedy_eda"

                ep_ext += float(extrinsic)
                ep_int += float(interestingness)
                log_returned_sets(output_datasets, output_states, episode_set_ids, logged_state_ids,
                    state_writer, trace_writer, [episode, worker_id, step],
                    [family, parameter, input_set_id, operation_action, source],
                    float(extrinsic), float(interestingness))

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
                partial_writer.writerow(
                    [episode, ep_ext, ep_int, len(episode_set_ids), ep_ext]
                )
                partial_file.flush()
            trace_file.flush()
            state_file.flush()
            print(
                f"EP{episode} greedy_eda worker{worker_id} | "
                f"Ext_R: {ep_ext:.3f} | Int: {ep_int:.3f} | "
                f"sets_viewed: {len(episode_set_ids)}"
            )
    finally:
        flush_events()
        trace_file.close()
        state_file.close()
        if partial_file is not None:
            partial_file.close()

    return rows, target_items, trace_partial_path, state_partial_path


def _merge_results(args, results):
    helpers = _random_helpers()
    rows = []
    target_items = []
    trace_partial_paths = []
    state_partial_paths = []
    for worker_rows, worker_targets, trace_path, state_path in results:
        rows.extend(worker_rows)
        if worker_targets and not target_items:
            target_items = worker_targets
        trace_partial_paths.append(trace_path)
        state_partial_paths.append(state_path)
    rows.sort(key=lambda row: row["episode"])
    episodes = [int(row["episode"]) for row in rows]
    expected = list(range(1, int(args.episodes) + 1))
    if episodes != expected:
        raise ValueError("Greedy EDA worker merge produced missing or duplicate episodes")

    if target_items:
        target_path = f"{args.output_prefix}_target_items.json"
        helpers.ensure_parent_dir(target_path)
        with open(target_path, "w", encoding="utf-8") as output:
            json.dump(target_items, output, indent=1)

    reward_path = f"{args.output_prefix}_{args.baseline}_rewards.csv"
    helpers.ensure_parent_dir(reward_path)
    global_sets = set()
    cumulative_ext = 0.0
    with open(reward_path, "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(helpers.CSV_HEADER)
        for row in rows:
            global_sets.update(row["set_ids"])
            cumulative_ext += float(row["extrinsic_reward"])
            episode_sets = int(row["sets_viewed"])
            cumulative_sets = len(global_sets)
            writer.writerow(
                [
                    row["episode"],
                    row["extrinsic_reward"],
                    row["interestingness"],
                    episode_sets,
                    row["total_reward"],
                    cumulative_sets,
                    float(row["extrinsic_reward"]) / max(episode_sets, 1),
                    cumulative_ext,
                    cumulative_ext / max(cumulative_sets, 1),
                ]
            )

    trace_path = f"{args.output_prefix}_{args.baseline}_exploration_trace.csv"
    helpers.ensure_parent_dir(trace_path)
    with open(trace_path, "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(helpers.TRACE_HEADER)
        for partial_path in trace_partial_paths:
            with open(partial_path, newline="") as source:
                reader = csv.reader(source)
                next(reader, None)
                writer.writerows(reader)
            os.remove(partial_path)

    states_path = f"{args.output_prefix}_{args.baseline}_visited_set_states.csv"
    helpers.ensure_parent_dir(states_path)
    seen_ids = set()
    header_written = False
    with open(states_path, "w", newline="") as output:
        writer = csv.writer(output)
        for partial_path in state_partial_paths:
            with open(partial_path, newline="") as source:
                reader = csv.reader(source)
                header = next(reader, None)
                if header is not None and not header_written:
                    writer.writerow(header)
                    header_written = True
                for row in reader:
                    if not row:
                        continue
                    set_id = int(row[0])
                    if set_id not in seen_ids:
                        seen_ids.add(set_id)
                        writer.writerow(row)
            os.remove(partial_path)

    return {
        "rewards": reward_path,
        "trace": trace_path,
        "states": states_path,
        "config": _write_config(args),
    }


@shared_query_run
def run(args):
    set_query_report_path(Path(f"{args.output_prefix}_query_scheduler.json"))
    tasks = [
        (args, worker_id, start, end)
        for worker_id, start, end in episode_ranges(args.episodes, args.workers)
    ]
    results = run_worker_batch(run_worker, tasks, len(tasks))
    paths = _merge_results(args, results)
    for label, path in paths.items():
        print(f"Saved {label} to {path}")
    return paths


def build_parser():
    parser = argparse.ArgumentParser(
        description=''
    )
    parser.add_argument("--baseline", choices=["greedy_eda"], default="greedy_eda")
    parser.add_argument("--target_set", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--mode", type=str, default="scattered")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_seed", type=int, default=None)
    parser.add_argument("--target_samples_per_file", type=int, default=100)
    parser.add_argument("--output_prefix", type=str, default="greedy_eda")
    parser.add_argument("--write_partial", action="store_true")
    add_query_arguments(parser)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
