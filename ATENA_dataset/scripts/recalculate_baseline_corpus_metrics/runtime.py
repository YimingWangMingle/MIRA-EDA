import csv
import json
import os
from pathlib import Path

from . import load_result_config, rollout_argmax
from .compat import enable_legacy_pandas


LAST_LOADED_WEIGHT_PATH = ""


def _default_env_factory(schema, dataset, seed, args, reward_mode):
    from atena_baselines.env import make_env

    return make_env(schema, dataset, seed, args, reward_mode=reward_mode)


def _default_model_factory(state_dim, action_dim, hidden):
    from atena_baselines.models import PolicyValueNet

    return PolicyValueNet(state_dim, action_dim, hidden=hidden)


def _default_mira_env_factory(schema, dataset, seed, args):
    enable_legacy_pandas()
    import sys

    mira_root = Path(__file__).resolve().parents[2] / "MIRA"
    if str(mira_root) not in sys.path:
        sys.path.insert(0, str(mira_root))
    from mira.env import make_env

    return make_env(schema, dataset, seed, args)


def reconstruct_policy_session(result_dir, env_factory=None, model_factory=None):
    global LAST_LOADED_WEIGHT_PATH

    result_dir = Path(result_dir)
    args = load_result_config(result_dir / "config.json")
    env_factory = env_factory or _default_env_factory
    model_factory = model_factory or _default_model_factory
    env = env_factory(
        args.schema,
        args.dataset_number,
        args.seed + 777,
        args,
        _reward_mode(args.method),
    )
    model = model_factory(env.state_dim, env.action_dim, args.hidden)
    weights_path = result_dir / "policy.weights.h5"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    model.load_weights(str(weights_path))
    LAST_LOADED_WEIGHT_PATH = str(weights_path)
    return rollout_argmax(env, model)


def reconstruct_mira_session(result_dir, env_factory=None, model_factory=None):
    global LAST_LOADED_WEIGHT_PATH

    result_dir = Path(result_dir)
    args = load_result_config(result_dir / "config.json")
    env_factory = env_factory or _default_mira_env_factory
    if model_factory is None:
        from .policy import NumpyPolicyValueNet

        model_factory = NumpyPolicyValueNet
    env = env_factory(
        args.schema,
        args.dataset_number,
        args.seed + 777,
        args,
    )
    model = model_factory(env.state_dim, env.action_dim, args.hidden)
    weights_path = result_dir / "policy.weights.h5"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    model.load_weights(str(weights_path))
    LAST_LOADED_WEIGHT_PATH = str(weights_path)
    return rollout_argmax(env, model)


def reconstruct_greedy_session(result_dir, env_factory=None, action_selector=None):
    result_dir = Path(result_dir)
    args = load_result_config(result_dir / "config.json")
    env_factory = env_factory or _default_env_factory
    if action_selector is None:
        from atena_baselines.greedy import select_greedy_action

        action_selector = select_greedy_action
    env = env_factory(
        args.schema,
        args.dataset_number,
        args.seed,
        args,
        "official_compound",
    )
    env.reset()
    done = False
    while not done:
        action_index, preview_reward, _ = action_selector(env)
        _, committed_reward, done, _ = env.step(action_index)
        if abs(float(preview_reward) - float(committed_reward)) > 1e-9:
            raise ValueError(
                f"greedy preview reward {preview_reward} != committed reward {committed_reward}"
            )
    if len(env.actions) != 12:
        raise ValueError(f"expected 12 actions, got {len(env.actions)}")
    return list(env.actions)


def reconstruct_official_session(
    results_dir,
    schema,
    dataset,
    seed=0,
    meta_factory=None,
    converter=None,
):
    if meta_factory is None or converter is None:
        from evaluate_official_atena import (
            convert_official_steps,
            dataset_meta_from_args,
        )

        meta_factory = meta_factory or dataset_meta_from_args
        converter = converter or convert_official_steps
    source = (
        Path(results_dir)
        / "official_atena_eval"
        / schema
        / f"dataset{dataset}"
        / f"seed{seed}"
        / "raw_actions.json"
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    raw_actions = json.loads(source.read_text(encoding="utf-8"))
    actions = converter(meta_factory(schema, dataset), raw_actions)
    if len(actions) != 12:
        raise ValueError(f"{source} expected 12 actions, got {len(actions)}")
    return list(actions)


def write_outputs(output_dir, detail_rows, summary_rows, manifest):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "all_methods_official_corpus_detail.csv"
    summary_path = output_dir / "all_methods_official_corpus_summary.csv"
    manifest_path = output_dir / "all_methods_official_corpus_manifest.json"
    pending = []
    try:
        pending.append((_write_csv_temp(detail_path, detail_rows), detail_path))
        pending.append((_write_csv_temp(summary_path, summary_rows), summary_path))
        pending.append((_write_json_temp(manifest_path, manifest), manifest_path))
        for temporary, destination in pending:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in pending:
            if temporary.exists():
                temporary.unlink()
    return detail_path, summary_path, manifest_path


def _write_csv_temp(destination, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"cannot write empty CSV {destination}")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return temporary


def _write_json_temp(destination, payload):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return temporary


def _reward_mode(method):
    if method == "dora":
        return "dora"
    return "official_compound"
