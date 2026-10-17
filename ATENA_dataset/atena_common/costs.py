import json
import time
from pathlib import Path


COUNTER_KEYS = ('commit', 'preview', 'replay', 'preview_replay')


class QueryCosts:
    def __init__(self):
        self.counts = dict.fromkeys(COUNTER_KEYS, 0)
        self.seconds = 0.0

    def snapshot(self):
        return dict(self.counts, total=sum(self.counts.values()), query_seconds=self.seconds)

    def add(self, counts):
        for key in COUNTER_KEYS:
            self.counts[key] += int(counts.get(key, 0))
        self.seconds += float(counts.get('query_seconds', 0.0))


def action_records(actions):

    return [{key: (value.name if hasattr(value, 'name') else value)
             for key, value in vars(action).items()} for action in actions]


def write_session(path, actions):
    Path(path).write_text(json.dumps(action_records(actions), indent=2, default=str), encoding='utf8')


def session_cost_row(env, generation_started, metrics_function):
    generation_seconds = time.perf_counter() - generation_started
    metric_started = time.perf_counter()
    metrics = metrics_function()
    metric_seconds = time.perf_counter() - metric_started
    costs = getattr(env, 'query_costs', QueryCosts()).snapshot()
    return dict(metrics, session_generation_seconds=generation_seconds,
                metric_seconds=metric_seconds,
                **{'simulator_' + key: value for key, value in costs.items()})


def write_costs(result_dir, rows, training_started=None, training_steps=0, training_queries=None):

    queries = QueryCosts()
    for row in rows:
        queries.add({key: row.get('simulator_' + key, 0) for key in COUNTER_KEYS + ('query_seconds',)})
    setup = sum(float(row.get("session_setup_seconds", 0)) for row in rows)
    generation = sum(float(row.get('session_generation_seconds', 0)) for row in rows)
    metrics = sum(float(row.get('metric_seconds', 0)) for row in rows)
    elapsed = time.perf_counter() - training_started if training_started is not None else setup + generation + metrics
    report = dict(
        training_steps=int(training_steps),
        training_wall_seconds=max(0.0, elapsed - setup - generation - metrics) if training_started is not None else 0.0,
        training_simulator_calls=(training_queries or QueryCosts()).snapshot(),
        session_count=len(rows), session_setup_seconds=setup, session_generation_seconds=generation, metric_seconds=metrics,
        generation_seconds_per_session=generation / max(len(rows), 1),
        simulator_calls=queries.snapshot(),
        simulator_calls_per_session=queries.snapshot()['total'] / max(len(rows), 1),
        evaluations=[dict(steps=row.get('steps', 0), episode=row.get('episode', 0),
                          setup_seconds=row.get('session_setup_seconds', 0),
                          generation_seconds=row.get('session_generation_seconds', 0),
                          metric_seconds=row.get('metric_seconds', 0),
                          simulator_calls={key: row.get('simulator_' + key, 0) for key in COUNTER_KEYS}) for row in rows],
        counting_unit='Simulator action executions, including candidate previews and reward-history replays; not physical SQL queries.',
        training_wall_scope='Setup, rollout, optimizer updates and checkpoint serialization; excludes timed evaluation setup, session generation and metric evaluation.',
    )
    if rows and rows[0].get("method") != "random":
        report['final_session_generation_seconds'] = float(rows[-1].get('session_generation_seconds', 0))
    Path(result_dir, 'costs.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    return report


def average_random_rows(rows):
    if not rows:
        raise ValueError('Random evaluation requires at least one session')
    identity = {'method', 'schema', 'dataset', 'seed', 'steps'}
    final = {key: value for key, value in rows[0].items() if key in identity}
    for key in rows[0]:
        if key not in identity and key != 'episode':
            final[key] = sum(float(row[key]) for row in rows) / len(rows)
    final['evaluation_sessions'] = len(rows)
    return final
