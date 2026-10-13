"""One reward CSV schema for PCA collection and Galaxy learning episodes."""
import csv
from pathlib import Path

from .csv_output import locked_csv_append

FIELDS = ('episode', 'extrinsic_reward', 'interestingness', 'mira_bonus',
          'exploration_reward', 'total_reward', 'sets_viewed',
          'cumulative_unique_sets_viewed', 'target_efficiency',
          'cumulative_extrinsic_reward', 'cumulative_target_efficiency', 'z_source',
          'mira_phi_loss', 'mira_dyn_loss', 'mira_prediction_error',
          'mira_mean_d_phi', 'mira_mean_metric_target', 'mira_update_count',
          'mira_pair_source', 'mira_replay_size')


def write_reward(args, row):
    directory = Path(args.result_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with locked_csv_append(directory / (args.name + '_fusion_rewards.csv')) as (stream, exists):
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, 0.) for key in FIELDS})
