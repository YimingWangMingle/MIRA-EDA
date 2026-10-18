from pathlib import Path

ENCODERS = ('learned', 'stats_pca', 'vae', 'frozen_random')
REWARDS = ('directional', 'rnd', 'count', 'knn', 'random')


def add_ablation_arguments(parser, backend='galaxy'):
    from revision_support.query_config import add_query_arguments
    add_query_arguments(parser)
    parser.add_argument('--exploration_wup', type=int, choices=(0,), default=0,
                        help='')
    parser.add_argument('--spool_size', type=int, choices=(0,), default=0,
                        help='')
    parser.add_argument('--output_profile', choices=('full', 'csv'), default='csv',
                        help='')
    parser.add_argument('--encoder_variant', choices=ENCODERS, default='learned')
    parser.add_argument('--encoder_pca_path', default=None,
                        help='')
    parser.add_argument('--pca_collection_episodes', type=int, default=16,
                        help='')
    parser.add_argument('--encoder_vae_kl', type=float, default=0.001)
    parser.add_argument('--exploration_reward', choices=REWARDS, default='directional',
                        help='')
    parser.add_argument('--replacement_reward_scale', type=float, default=1.)
    parser.add_argument('--replacement_rnd_lr', type=float, default=3e-4)
    parser.add_argument('--replacement_knn_k', type=int, default=5)
    parser.add_argument('--replacement_knn_capacity', type=int, default=2048,
                        help='')


def ablation_label(args):
    query = getattr(args, 'query_ablation', 'none')
    if query != 'none':
        return 'query_' + query
    encoder = getattr(args, 'encoder_variant', 'learned')
    reward = getattr(args, 'exploration_reward', 'directional')
    if encoder != 'learned':
        return 'encoder_' + encoder
    if reward != 'directional':
        return 'reward_' + reward
    return ''


def validate_ablation_args(args):
    from revision_support.query_config import configure_query_processing
    configure_query_processing(args)
    if getattr(args, 'exploration_wup', 0) != 0 or getattr(args, 'spool_size', 0) != 0:
        raise ValueError('This distribution requires exploration_wup=0')
    encoder = getattr(args, 'encoder_variant', 'learned')
    reward = getattr(args, 'exploration_reward', 'directional')
    if encoder not in ENCODERS or reward not in REWARDS:
        raise ValueError('Unknown encoder/reward replacement')
    if encoder != 'learned' and reward != 'directional':
        raise ValueError('Replace one component at a time')
    if reward == 'knn':
        k = int(getattr(args, 'replacement_knn_k', 5))
        capacity = int(getattr(args, 'replacement_knn_capacity', 2048))
        if not 1 <= k <= capacity:
            raise ValueError('kNN requires 1 <= replacement_knn_k <= replacement_knn_capacity')
    if encoder == 'stats_pca':
        from .pca_training import collection_budget
        collection_budget(args)
        path = getattr(args, 'encoder_pca_path', None)
        if path and (Path(path).exists() or Path(path).with_suffix('.json').exists()):
            raise ValueError('stats_pca requires fresh in-budget collection; existing PCA cannot be reused')
    import math
    for name in ('encoder_vae_kl', 'replacement_reward_scale', 'replacement_rnd_lr'):
        value = float(getattr(args, name, 1.))
        if not math.isfinite(value) or value < 0 or (name == 'replacement_rnd_lr' and value == 0):
            raise ValueError(name + ' must be finite and nonnegative (RND learning rate positive)')
    if ablation_label(args):
        if (getattr(args, 'resume', False) or getattr(args, 'resume_step', None) is not None
                or getattr(args, 'resume_model_dir', None) or getattr(args, 'resume_start_episode', 0)):
            raise ValueError('Replacement experiments require a clean initialization')
        if getattr(args, 'baseline', 'mira') not in ('mira', 'mira_no_ext'):
            raise ValueError('Replacement experiments apply to MIRA, not independent baselines')
    return args


def representation_options(args):
    return dict(statistic_indices=getattr(args, 'mira_statistic_indices', None),
                task_reward_enabled=(getattr(args, 'baseline', 'mira') != 'mira_no_ext'
                                     and float(getattr(args, 'w_ext', 2.)) != 0.),
                lambda_ana=getattr(args, 'mira_lambda_ana', 1.), mc_samples=getattr(args, 'mira_mc_samples', 4),
                encoder_variant=getattr(args, 'encoder_variant', 'learned'),
                encoder_pca_path=getattr(args, 'encoder_pca_path', None),
                encoder_vae_kl=getattr(args, 'encoder_vae_kl', .001))
