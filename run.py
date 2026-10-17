import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BASELINES = ('mira', 'mira_no_ext', 'pure_a3c', 'dora', 'atena', 'atena_no_ext', 'greedy_eda', 'random')
ABLATIONS = ('encoder_stats_pca', 'encoder_vae', 'encoder_frozen_random',
             'reward_rnd', 'reward_count', 'reward_knn', 'reward_random')
METHODS = BASELINES + ABLATIONS
DATASETS = ('galaxy', 'covertype')


def build_jobs(root=ROOT, datasets=DATASETS, methods=METHODS, seeds=(1, 2, 3), smoke=False, output=None):
    root = Path(root).resolve()
    output = Path(output).resolve() if output else root/('smoke_runs' if smoke else 'runs')
    episodes = 2 if smoke else 1000
    jobs = []
    for dataset in datasets:
        for method in methods:
            for seed in seeds:
                if dataset not in DATASETS or method not in METHODS or seed not in (1, 2, 3):
                    raise ValueError('Unknown dataset, method or fixed target seed')
                ablation = method in ABLATIONS
                baseline = 'mira' if ablation else method
                task = '{}{}_seed{}'.format('smoke_' if smoke else '', dataset, seed)
                name = '{}_{}'.format(task, method)
                base = task if ablation else name
                directory = output/name
                common = ['--episodes', str(episodes), '--steps', '250', '--workers', '14',
                          '--target_set', 'fixed_seed_{}'.format(seed)]
                if dataset == 'galaxy':
                    launchers = dict(mira='RL-launcher.py', mira_no_ext='RL-launcher.py',
                        pure_a3c='RL-launcher-pure-a3c.py', dora='RL-launcher-dora.py',
                        atena='RL-launcher-atena.py', atena_no_ext='RL-launcher-atena-no-ext.py',
                        greedy_eda='RL-launcher-greedy-eda.py', random='RL-launcher-random-baseline.py')
                    command = [sys.executable, str(root/launchers[baseline])] + common
                    command += ['--mode', 'scattered', '--target_seed', str(seed)]


                    if baseline not in ('dora', 'atena', 'atena_no_ext'):
                        command += ['--seed', str(seed)]
                    if baseline in ('random', 'greedy_eda'):
                        command += ['--baseline', baseline, '--output_prefix', str(directory/name)]
                        prefix = name + '_' + baseline
                        suffix = '_rewards.csv'
                    else:
                        command += ['--name', base, '--output_dir', str(output), '--output_profile', 'csv']
                        prefix = name + ('_pure_a3c' if baseline == 'pure_a3c' else '')
                        suffix = '_rewards.csv' if baseline == 'pure_a3c' else '_fusion_rewards.csv'
                    if baseline in ('mira', 'mira_no_ext'):
                        command += ['--w_ext', '0' if baseline == 'mira_no_ext' else '2',
                                    '--strict_episode_budget', '--exploration_wup', '0', '--spool_size', '0']
                    elif baseline in ('pure_a3c', 'atena'):
                        command += ['--w_ext', '2']
                else:
                    script = ('covertype-exploration/baselines/{}/run.py'.format(baseline)
                              if baseline in ('random', 'greedy_eda') else 'covertype-exploration/run_full_a3c.py')
                    command = [sys.executable, str(root/script)] + common
                    command += ['--seed', str(seed), '--preprocess_name', 'by_distribution_path100k_seed{}'.format(seed)]
                    if baseline in ('random', 'greedy_eda'):
                        command += ['--output_dir', str(directory), '--output_prefix', name]
                    else:
                        command += ['--baseline', baseline, '--output_dir', str(output), '--output_prefix', base,
                                    '--output_profile', 'csv', '--exploration_wup', '0', '--spool_size', '0',
                                    '--w_ext', '0' if baseline.endswith('no_ext') else '2']
                    if baseline != 'greedy_eda':
                        command += ['--target_seed', str(seed), '--n_sets', '100000',
                                    '--csv_path', str(root/'covertype-exploration/covertype.csv')]
                    prefix, suffix = name + '_' + baseline, '_rewards.csv'
                if baseline in ('atena', 'atena_no_ext'):
                    command += ['--w_int', '0.1', '--w_coh', '0.1', '--w_div', '0.1']
                if baseline == 'dora':
                    command += ['--counter_curiosity_ratio', '0.25']
                collection = 0
                if method.startswith('encoder_'):
                    command += ['--encoder_variant', method[len('encoder_'):]]
                    if method == 'encoder_stats_pca':
                        collection = 1 if smoke else 16
                        command += ['--pca_collection_episodes', str(collection)]
                elif method.startswith('reward_'):
                    command += ['--exploration_reward', method[len('reward_'):]]
                jobs.append(dict(dataset=dataset, method=method, seed=seed, name=name,
                    episodes=episodes, steps=250, workers=14, collection_episodes=collection,
                    learning_episodes=episodes-collection, command=command,
                    directory=str(directory), measurement_dir=str(output/'cost'/name),
                    rewards_path=str(directory/(prefix+suffix))))
    return jobs


def validate_rewards(job):
    with Path(job['rewards_path']).open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    indexed = {int(row['episode']): row for row in rows}
    if len(rows) != job['episodes'] or set(indexed) != set(range(1, job['episodes']+1)):
        raise ValueError('Missing or duplicate completed episodes: ' + job['name'])
    total = 0.
    for episode in range(1, job['episodes']+1):
        row = indexed[episode]
        total += float(row['extrinsic_reward'])
        if not math.isclose(float(row['cumulative_extrinsic_reward']), total, rel_tol=1e-8, abs_tol=1e-6):
            raise ValueError('Cumulative reward mismatch: ' + job['name'])
    final = indexed[job['episodes']]
    return dict(episodes=job['episodes'], cumulative_task_reward=total,
                cumulative_unique_sets_viewed=int(final['cumulative_unique_sets_viewed']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('all',)+DATASETS, default='all')
    parser.add_argument('--method', choices=('all', 'baselines', 'ablations')+METHODS, default='all')
    parser.add_argument('--seeds', nargs='+', type=int, choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument('--smoke', action='store_true', help='Two full-length episodes; PCA uses one collection + one learning')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true', help='Without this flag, print the plan only')
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error('Repeated seeds are not permitted')
    methods = {'all': METHODS, 'baselines': BASELINES, 'ablations': ABLATIONS}.get(args.method, (args.method,))
    jobs = build_jobs(ROOT, DATASETS if args.dataset == 'all' else (args.dataset,), methods,
                      args.seeds, args.smoke, args.output)
    if not args.execute:
        print(json.dumps(jobs, indent=2))
        return
    os.chdir(ROOT)
    from revision_support.training_runtime import single_thread_environment
    env = single_thread_environment()
    env.update(CUDA_VISIBLE_DEVICES='', RAY_ADDRESS='local', WANDB_MODE='disabled',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, str(ROOT/'verify.py'), '--runtime'], check=True, env=env)
    for job in jobs:
        directory, cost = Path(job['directory']), Path(job['measurement_dir'])
        if directory.exists() or cost.exists() or (ROOT/'saved_models'/job['name']).exists():
            raise FileExistsError('Choose a fresh output directory; existing run is preserved: ' + job['name'])

        directory.parent.mkdir(parents=True, exist_ok=True)
        print('START ' + job['name'], flush=True)
        subprocess.run([sys.executable, str(ROOT/'scripts/measure_experiment.py'), '--output', str(cost),
                        '--profile-mode', 'aggregate', '--'] + job['command'], check=True, env=env)
        metrics = validate_rewards(job)
        (directory/'run.json').write_text(json.dumps(dict(job=job, metrics=metrics), indent=2), encoding='utf-8')
        print('COMPLETE ' + job['name'], flush=True)


if __name__ == '__main__':
    main()
