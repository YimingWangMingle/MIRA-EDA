import argparse
import csv
import json
from pathlib import Path

METHODS = ('MIRA', 'official_atena', 'pure_a3c', 'dora', 'greedy', 'random')


def summarize(root, seed, resource_root=None):
    root = Path(root)
    rows = []
    for method in METHODS:
        for schema in ('cyber', 'flights'):
            for number in range(1, 5):
                directory = (root / 'official_atena_eval' / schema / f'dataset{number}' / f'seed{seed}'
                             if method == 'official_atena' else root / method / f'{schema}{number}' / f'seed{seed}')
                costs = json.loads((directory / 'costs.json').read_text(encoding='utf8'))
                random = method == 'random'
                final = costs.get('evaluations', [{}])[-1]
                count = costs['session_count'] if random else 1
                generation = (costs['generation_seconds_per_session'] if random else
                              final.get('generation_seconds', costs.get('session_generation_seconds')))
                queries = (costs.get('simulator_calls_per_session') if random else
                           sum(final['simulator_calls'].values()) if 'simulator_calls' in final else None)
                row = dict(method=method, schema=schema, dataset=number, training_seed=int(seed),
                           evaluation_draws=count, generation_seconds_per_session=generation,
                           simulator_calls_per_session=queries,
                           training_steps=costs.get('training_steps'),
                           training_wall_seconds=costs.get('training_wall_seconds'),
                           total_metric_seconds=costs['metric_seconds'],
                           full_run_wall_seconds=None, peak_process_tree_rss_bytes=None)
                if resource_root is not None:
                    path = Path(resource_root) / f'{method}_{schema}{number}_seed{seed}' / 'resources.json'
                    resources = json.loads(path.read_text(encoding='utf8'))
                    if resources['status'] != 'completed':
                        raise ValueError(f'Incomplete resource measurement: {path}')
                    row.update(full_run_wall_seconds=resources['full_run_wall_seconds'],
                               peak_process_tree_rss_bytes=resources['peak_process_tree_rss_bytes'])
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--root', type=Path, default=Path('results'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resource-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows = summarize(args.root, args.seed, args.resource_root)
    output = args.output or args.root / f'costs_seed{args.seed}.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == '__main__':
    main()
