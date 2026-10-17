import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from atena_baselines.env import AtenaEDAEnv as SharedAtenaEnv, AtenaActionSpace, dataset_enum


class AtenaEDAEnv(SharedAtenaEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.statistic_indices = list(range(len(self.dataset.columns) * 3 + 5))


def make_env(schema, dataset_number, seed, args):
    return AtenaEDAEnv(
        schema=schema,
        dataset_number=dataset_number,
        episode_length=args.episode_length,
        max_terms_per_column=args.max_terms_per_column,
        seed=seed,
        reward_mode='official_compound',
        w_kl=getattr(args, 'w_kl', 1.5),
        w_compaction=getattr(args, 'w_compaction', 2.0),
        w_official_diversity=getattr(args, 'w_official_diversity', 2.0),
        empty_penalty=args.empty_penalty,
        repeat_penalty=args.repeat_penalty,
    )
