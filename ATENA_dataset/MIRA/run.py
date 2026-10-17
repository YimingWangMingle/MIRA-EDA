import argparse

from mira import METHOD


def validate_args(args):
    if int(args.dataset_number) not in {1, 2, 3, 4}:
        raise ValueError("dataset_number must be between 1 and 4")
    if int(args.workers) < 1:
        raise ValueError("workers must be positive")
    if int(args.steps) < 1:
        raise ValueError("steps must be positive")
    if int(args.episode_length) != 12:
        raise ValueError("MIRA requires episode_length=12")
    return args


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run standalone MIRA on ATENA."
    )
    parser.add_argument("--schema", choices=["flights", "cyber"], required=True)
    parser.add_argument("--dataset_number", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--episode_length", type=int, default=12)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--max_terms_per_column", type=int, default=10)


    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.97)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=40.0)

    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mira_latent_dim", type=int, default=16)
    parser.add_argument("--mira_hidden", type=int, default=256)
    parser.add_argument("--mira_lr", type=float, default=3e-4)
    parser.add_argument("--mira_tau", type=float, default=0.01)
    parser.add_argument("--mira_lambda_ana", type=float, default=1.0)
    parser.add_argument("--mira_mc_samples", type=int, default=4)
    parser.add_argument("--mira_replay_buffer_size", type=int, default=50000)
    parser.add_argument("--mira_phi_batch_size", type=int, default=128)
    parser.add_argument("--mira_min_replay_size", type=int, default=256)
    parser.add_argument("--mira_bonus_clip", type=float, default=1.0)

    parser.add_argument("--w_kl", type=float, default=1.5)
    parser.add_argument("--w_compaction", type=float, default=2.0)
    parser.add_argument("--w_official_diversity", type=float, default=2.0)
    parser.add_argument("--empty_penalty", type=float, default=-1.0)
    parser.add_argument("--repeat_penalty", type=float, default=-1.0)


    args = parser.parse_args(argv)
    args.mira_core_version = "mira_grounded_latent_v2"
    args.method = METHOD
    args.policy_protocol = "final_online_policy"
    try:
        return validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))


def main():
    from mira.train import run_experiment

    run_experiment(parse_args())


if __name__ == "__main__":
    main()
