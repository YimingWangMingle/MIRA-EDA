import math
import os

QUERY_ABLATIONS = ('no_result_reuse', 'no_heuristic_pruning', 'no_shared_execution')
QUERY_PROTOCOL_VERSION = 'common_query_processing_v1'


def add_query_arguments(parser):
    parser.add_argument('--query_ablation', choices=('none',) + QUERY_ABLATIONS, default='none',
                        help='')
    parser.add_argument('--query_cache_mb', type=float, default=128.)
    parser.add_argument('--query_batch_ms', type=float, default=2.)


def query_protocol(variant='none', cache_mb=128., batch_ms=2.):
    if variant not in ('none',) + QUERY_ABLATIONS:
        raise ValueError('Unknown query ablation: ' + str(variant))
    if not all(math.isfinite(float(x)) and float(x) >= 0 for x in (cache_mb, batch_ms)):
        raise ValueError('Query cache and batch limits must be finite and nonnegative')
    return dict(version=QUERY_PROTOCOL_VERSION, query_ablation=variant,
        result_reuse=variant != 'no_result_reuse',
        heuristic_pruning=variant != 'no_heuristic_pruning',
        shared_execution=variant != 'no_shared_execution',
        cache_mb=float(cache_mb), batch_ms=float(batch_ms))


def configure_query_processing(args, method=None):
    variant = getattr(args, 'query_ablation', 'none')
    method = str(method or getattr(args, 'baseline', getattr(args, 'method', 'mira'))).lower()
    if variant != 'none':
        if method != 'mira':
            raise ValueError('Query ablations apply only to full MIRA')
        if (getattr(args, 'encoder_variant', 'learned') != 'learned'
                or getattr(args, 'exploration_reward', 'directional') != 'directional'):
            raise ValueError('Replace one component at a time')
        if float(getattr(args, 'w_ext', 2.)) == 0.:
            raise ValueError('Query ablations use full MIRA, not MIRA without external reward')
    protocol = query_protocol(variant, getattr(args, 'query_cache_mb', 128.),
                              getattr(args, 'query_batch_ms', 2.))
    from .query_service import service_available, execution_options
    if service_available():
        current = execution_options()
        requested = (protocol['result_reuse'], protocol['shared_execution'],
                     int(protocol['cache_mb']*1024**2), protocol['batch_ms']/1000.)
        if tuple(current[k] for k in ('result_reuse', 'shared_execution', 'cache_bytes', 'batch_seconds')) != requested:
            raise ValueError('Query protocol must be configured before starting the service')
    for key in ('result_reuse', 'heuristic_pruning', 'shared_execution'):
        os.environ['MIRA_' + key.upper()] = '1' if protocol[key] else '0'
    os.environ['MIRA_QUERY_CACHE_MB'] = str(protocol['cache_mb'])
    os.environ['MIRA_QUERY_BATCH_MS'] = str(protocol['batch_ms'])
    args.query_processing = protocol
    return args


def heuristic_pruning_enabled():
    value = os.environ.get('MIRA_HEURISTIC_PRUNING', '1')
    if value not in ('0', '1'):
        raise ValueError('MIRA_HEURISTIC_PRUNING must be 0 or 1')
    return value == '1'


def query_worker_environment():
    return {k: v for k, v in os.environ.items() if k.startswith('MIRA_QUERY_')
            or k in ('MIRA_RESULT_REUSE', 'MIRA_HEURISTIC_PRUNING', 'MIRA_SHARED_EXECUTION',
                     'MIRA_COVERTYPE_QUERY_SNAPSHOT')}


def query_actor(actor):
    return actor.options(runtime_env={'env_vars': query_worker_environment()})
