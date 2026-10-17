import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from covertype_rl import full_a3c


def _ensure_pure_a3c_registered():

    full_a3c.BASELINES = set(full_a3c.BASELINES)
    full_a3c.BASELINES.add("pure_a3c")

    original_training_reward = full_a3c._training_reward

    def training_reward(baseline, r_ext, r_int, counter_curiosity, r_coh, r_div, mira_bonus, args):
        if baseline == "pure_a3c":
            return float(args.w_ext) * float(r_ext)
        return original_training_reward(baseline, r_ext, r_int, counter_curiosity, r_coh, r_div, mira_bonus, args)

    full_a3c._training_reward = training_reward


def main():
    _ensure_pure_a3c_registered()
    parser = full_a3c.build_parser()
    parser.set_defaults(baseline="pure_a3c", w_ext=2.0)
    args = parser.parse_args()
    paths = full_a3c.run(args)
    print("Wrote:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
