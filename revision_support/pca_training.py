import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .pca import fit_projection, projection_weights

PROTOCOL = 'pca_in_budget_v1'


def collection_budget(args):
    count = int(getattr(args, 'pca_collection_episodes', 16))
    if not 1 <= count < int(args.episodes):
        raise ValueError('PCA collection must leave at least one learning episode')
    if int(args.steps) < 1:
        raise ValueError('PCA episode length must be positive')
    return count


def prepare_in_budget(env, args, backend, result_dir):

    if getattr(args, 'encoder_variant', 'learned') != 'stats_pca':
        return []
    count = collection_budget(args)
    artifact = Path(args.encoder_pca_path or Path(result_dir) / 'pca.npz').resolve()
    metadata = artifact.with_suffix('.json')
    if artifact.exists() or metadata.exists():
        raise FileExistsError('PCA requires fresh in-budget collection: ' + str(artifact))
    if getattr(env, 'observation_only', False):
        raise ValueError('In-budget PCA requires evaluation and exploration logs')
    seed = getattr(args, 'seed', None)
    if seed is None:
        seed = getattr(args, 'target_seed', None)
    rng = np.random.default_rng(seed)
    n_set = env.set_action_space.n if hasattr(env, 'set_action_space') else env.set_action_dim
    n_op = env.operation_action_space.n if hasattr(env, 'operation_action_space') else env.operation_action_dim

    def draw(probabilities):
        p = np.asarray(probabilities, dtype=np.float64)
        if not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
            raise ValueError('Invalid action mask during PCA collection')
        return int(rng.choice(len(p), p=p/p.sum()))

    started = time.perf_counter()
    states, records = [], []
    for episode in range(1, count + 1):
        state = env.reset()
        totals = dict.fromkeys(('extrinsic_reward', 'target_hits',
            'familiarity', 'counter_curiosity', 'coherency', 'diversity', 'total_reward',
            'valid_steps', 'mira_bonus', 'exploration_reward'), 0.)
        totals.update(mira_z_source='pca_collection', mira_pair_source='pca_collection')
        for step in range(1, int(args.steps) + 1):
            observation = np.asarray(state, dtype=np.float32).reshape(-1).copy()
            if observation.shape != (env.set_state_dim,) or not np.isfinite(observation).all():
                raise ValueError('Invalid PCA observation')
            states.append(observation)

            a = (0 if hasattr(env, 'datasets') and len(env.datasets) == 0 else
                 draw(env.fix_possible_set_action_probs(np.ones(n_set)/n_set)))
            b = draw(env.fix_possible_operation_action_probs(a, np.ones(n_op)/n_op))
            transition = env.step(a, b)
            state, external, _, done = transition[:4]
            if bool(done) != (step == int(args.steps)):
                raise ValueError('PCA environment episode length differs from declared budget')
            metrics = transition[5] if len(transition) > 5 else {}


            exploration = 0.0
            external_weight = 0. if getattr(args, 'baseline', '') == 'mira_no_ext' else float(args.w_ext)
            total = external_weight * float(external) + float(args.alpha) * exploration
            totals['extrinsic_reward'] += float(external)
            totals['target_hits'] += int(metrics.get('target_hits', 0))
            totals['familiarity'] += float(external)
            totals['valid_steps'] += int(metrics.get('valid', 0))
            totals['exploration_reward'] += exploration
            totals['total_reward'] += total
        trace, visited = env.consume_exploration_logs()
        for row in trace:
            row.update(mira_z_source='pca_collection', step_mira_bonus=0.,
                step_total_reward=external_weight * float(row.get('step_extrinsic_reward', 0.)))
        records.append(dict(episode=episode, totals=totals,
                            set_ids=list(env.sets_viewed), trace=trace, states=visited))
        print('PCA collection episode {}/{} (included in {} total episodes)'.format(
            episode, count, args.episodes), flush=True)
    collected = time.perf_counter()
    observations = np.asarray(states, dtype=np.float32)
    fitted = fit_projection(observations, int(args.mira_latent_dim))
    projection_weights(fitted, env.set_state_dim, int(args.mira_latent_dim))
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with artifact.open('xb') as stream:
        np.savez(stream, **fitted)
    manifest = dict(protocol=PROTOCOL, dataset=backend, seed=seed,
        total_episodes=int(args.episodes), collection_episodes=count,
        learning_episodes=int(args.episodes)-count, steps_per_episode=int(args.steps),
        interactions=count*int(args.steps), observations=len(observations),
        state_dim=env.set_state_dim, latent_dim=int(args.mira_latent_dim),
        collection_seconds=collected-started, fit_and_save_seconds=time.perf_counter()-collected,
        pca_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        observations_sha256=hashlib.sha256(observations.tobytes()).hexdigest(),
        note='Uniform masked actions; rewards logged for evaluation only. Collection and fit included in full-run costs.')
    with metadata.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
    args.encoder_pca_path = str(artifact)
    args.pca_protocol = PROTOCOL
    args.pca_collection_episodes = count
    return records


def record_collection(server, records, backend):

    if server.episodes_done or server.episodes_reserved:
        raise ValueError('PCA collection must be recorded before learning')
    for expected, record in enumerate(records, 1):
        if record['episode'] != expected:
            raise ValueError('PCA collection episodes must be contiguous')
        totals = record['totals']
        if backend == 'covertype':
            server.record_episode(expected, totals, record['set_ids'], record['trace'], record['states'])
        else:
            from .galaxy_rewards import write_reward
            metrics = server.record_episode_metrics(totals['extrinsic_reward'], record['set_ids'])
            server.record_exploration_logs(expected, record['trace'], record['states'],
                dict(z_source='pca_collection', mira_pair_source='pca_collection'))
            write_reward(server.args, dict(totals, **metrics, z_source='pca_collection'))
    server.episodes_reserved = server.episodes_done
    return server.episodes_done
