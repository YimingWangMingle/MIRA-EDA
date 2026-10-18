import argparse
import csv
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ATENA_BASIC = ROOT / "ATENA-A-EDA" / "atena-basic"
BENCHMARK = ROOT / "ATENA-A-EDA" / "benchmark"

for path in [str(ROOT.parent), str(ATENA_BASIC), str(BENCHMARK)]:
    if path not in sys.path:
        sys.path.insert(0, path)


from atena.evaluation.metrics import EvalInstance, get_dataframe_all_eval_metrics
from atena.simulation.actions import (
    AggregationFunction,
    BackAction,
    FilterAction,
    FilterOperator,
    GroupAction,
)
from atena.simulation.dataset import (
    CyberDatasetName,
    Dataset,
    DatasetMeta,
    FlightsDatasetName,
    SchemaName,
)


from revision_support.query_service import shared_query_run, set_query_report_path

@shared_query_run
def main():
    parser = argparse.ArgumentParser(
        description=''
    )
    parser.add_argument("--schema", choices=["flights", "cyber"], required=True)
    parser.add_argument("--dataset_number", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--load", required=True, help='')
    parser.add_argument("--episode_length", type=int, default=12)
    parser.add_argument("--output_dir", default=str(ROOT / "results" / "official_atena_eval"))
    parser.add_argument("--most_probable", action="store_true", default=True)
    args, passthrough = parser.parse_known_args()
    set_query_report_path(Path(args.output_dir) / ('query_scheduler_{}{}_seed{}.json'.format(
        args.schema, args.dataset_number, args.seed)))

    costs = {}
    official_steps, reward = generate_official_steps(args, passthrough, costs=costs)
    metric_started = time.perf_counter()
    dataset_meta = dataset_meta_from_args(args.schema, args.dataset_number)
    benchmark_actions = convert_official_steps(dataset_meta, official_steps)
    metrics = official_metrics(dataset_meta, benchmark_actions)
    costs["metric_seconds"] = time.perf_counter() - metric_started
    costs.update(session_count=1, simulator_calls=None,
                 counting_note="Native official engine is unchanged; simulator-call counts from custom methods are not applicable.")

    outdir = Path(args.output_dir) / args.schema / f"dataset{args.dataset_number}" / f"seed{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    row = {
        "method": "official_atena",
        "schema": args.schema,
        "dataset": args.dataset_number,
        "seed": args.seed,
        "load": str(args.load),
        "episode_reward": float(reward),
        "session_setup_seconds": costs["session_setup_seconds"],
        "session_generation_seconds": costs["session_generation_seconds"],
        "metric_seconds": costs["metric_seconds"],
        **metrics,
    }
    write_json(outdir / "raw_actions.json", to_jsonable(official_steps))
    write_json(outdir / "actions.json", [repr(action) for action in benchmark_actions])
    write_json(outdir / "final_metrics.json", row)
    write_csv(outdir / "eval_metrics.csv", [row])
    write_json(outdir / "costs.json", costs)
    print(json.dumps(row, indent=2))


def generate_official_steps(args, passthrough, costs=None):
    setup_started = time.perf_counter()
    official_dataset_index = official_dataset_number(args.dataset_number)
    sys.argv = [
        "evaluate_official_atena.py",
        "--load",
        str(args.load),
        "--env",
        "ATENAcont-v0",
        "--schema",
        "FLIGHTS" if args.schema == "flights" else "NETWORKING",
        "--dataset-number",
        str(int(official_dataset_index)),
        "--seed",
        str(int(args.seed)),
        "--algo",
        "chainerrl_ppo",
        "--arch",
        "FFParamSoftmax",
        "--episode-length",
        str(int(args.episode_length)),
        "--stack-obs-num",
        "3",
        *passthrough,
    ]

    from Utilities.Notebook.NotebookUtils import run_episode
    from Utilities.Utility_Functions import initialize_agent_and_env

    agent, env, official_args = initialize_agent_and_env(is_test=True)
    generation_started = time.perf_counter()
    if costs is not None:
        costs["session_setup_seconds"] = generation_started - setup_started
    info_hist, reward = run_episode(
        agent=agent,
        env=env.env,
        dataset_number=official_args.dataset_number,
        most_probable=True,
        verbose=False,
    )
    if costs is not None:
        costs["session_generation_seconds"] = time.perf_counter() - generation_started
    steps = []
    for info, step_reward in info_hist:
        raw_action = list(info["raw_action"])
        if len(raw_action) > 3:
            raw_action[3] -= 0.5
        steps.append({
            "raw_action": [float(value) for value in raw_action],
            "filter_term": info.get("filter_term"),
            "step_reward": float(step_reward),
            "action_text": str(info.get("action", "")),
        })
    return steps, reward


def official_dataset_number(dataset_number):
    dataset_number = int(dataset_number)
    if dataset_number < 1:
        raise ValueError(f"Benchmark dataset_number must be 1-4, got {dataset_number}")
    return dataset_number - 1


def dataset_meta_from_args(schema, dataset_number):
    if schema == "flights":
        return DatasetMeta(SchemaName.FLIGHTS, FlightsDatasetName(int(dataset_number)))
    if schema == "cyber":
        return DatasetMeta(SchemaName.CYBER, CyberDatasetName(int(dataset_number)))
    raise ValueError(f"Unsupported schema: {schema}")


def convert_official_steps(dataset_meta, official_steps):
    dataset = Dataset(dataset_meta)
    converted = []
    for step in official_steps:
        raw = step["raw_action"]
        action = [int(round(float(value))) for value in raw]
        action_type = action[0]
        if action_type == 0:
            converted.append(BackAction())
        elif action_type == 1:
            column = dataset.columns[clamp(action[1], 0, len(dataset.columns) - 1)]
            operator = convert_filter_operator(action[2])
            term = str(step.get("filter_term") if step.get("filter_term") is not None else raw[3])
            converted.append(FilterAction(column, operator, term))
        elif action_type == 2:
            group_column = dataset.columns[clamp(action[1], 0, len(dataset.columns) - 1)]
            agg_column = dataset.primary_key_columns[0]
            converted.append(GroupAction(group_column, agg_column, AggregationFunction.COUNT))
        else:
            raise ValueError(f"Unsupported official raw action type {action_type}: {raw}")
    return converted


def convert_filter_operator(operator_id):
    if operator_id in {0, 1, 2}:
        return FilterOperator.EQUAL
    if operator_id in {3, 4, 5}:
        return FilterOperator.NOTEQUAL
    if operator_id in {6, 7, 8}:
        return FilterOperator.CONTAINS
    return FilterOperator.EQUAL


def official_metrics(dataset_meta, actions):
    df = get_dataframe_all_eval_metrics([EvalInstance(dataset_meta, actions)])
    return {str(key): float(value) for key, value in df.iloc[0].to_dict().items()}


def clamp(value, low, high):
    return max(int(low), min(int(high), int(value)))


def to_jsonable(official_steps):
    return official_steps


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
