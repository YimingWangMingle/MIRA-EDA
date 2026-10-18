import argparse

from atena_baselines.train import METHODS, run_experiment


def validate_args(args):
    if not 1 <= int(args.workers) <= 32:
        raise ValueError("workers must be between 1 and 32")
    if not 1 <= int(args.dataset_number) <= 4:
        raise ValueError("dataset_number must be between 1 and 4")
    return args


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=''
    )
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--schema", choices=["flights", "cyber"], required=True)
    parser.add_argument("--dataset_number", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--episode_length", type=int, default=12)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--output_dir", type=str, default="results")

    parser.add_argument("--max_terms_per_column", type=int, default=10)
    parser.add_argument("--reward_mode", type=str, default="official_compound")
    parser.add_argument("--w_interestingness", type=float, default=1.0)
    parser.add_argument("--w_diversity", type=float, default=2.0)
    parser.add_argument("--w_coherency", type=float, default=1.0)
    parser.add_argument("--w_kl", type=float, default=1.5)
    parser.add_argument("--w_compaction", type=float, default=2.0)
    parser.add_argument("--w_official_diversity", type=float, default=2.0)
    parser.add_argument("--dora_curiosity_ratio", type=float, default=0.25)
    parser.add_argument("--dora_target_size", type=int, default=12)
    parser.add_argument("--dora_target_seed", type=int, default=0)
    parser.add_argument("--empty_penalty", type=float, default=-1.0)
    parser.add_argument("--back_penalty", type=float, default=-0.2)
    parser.add_argument("--repeat_penalty", type=float, default=-1.0)

    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.97)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=40.0)
    parser.add_argument("--random_eval_episodes", type=int, default=16)

    args = parser.parse_args(argv)
    try:
        return validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    run_experiment(parse_args())
