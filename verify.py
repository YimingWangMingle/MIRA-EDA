import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_source(root=ROOT):
    root = Path(root)
    sources = [path for path in root.rglob('*.py')
               if not any(part.startswith('.venv') or part == '__pycache__'
                          for part in path.relative_to(root).parts[:-1])]
    local_roots = {'app', 'rl', 'baselines', 'revision_support', 'covertype_rl', 'atena_common', 'atena_baselines', 'mira'}
    banned = ('ExplorationWup', 'create_exploration_wup', 'bootstrap_active',
              'escape_active', 'candidate_prior', 'success_direction_reuse', 'legacy_artifacts')
    for path in sources:
        relative = path.relative_to(root)
        source = path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(relative), feature_version=(3, 8))
        official = relative.parts[:2] == ('ATENA_dataset', 'ATENA-A-EDA')
        if path.name != 'verify.py' and not official:
            if 'bile' in source.lower() or any(token in source for token in banned):
                raise ValueError('Excluded implementation/reference in ' + str(relative))
        if any(part in relative.parts for part in ('output', 'outputs', 'results', 'saved_models', 'local_mechanisms')):
            raise ValueError('Unexpected source in release: ' + str(relative))
        parts = list(relative.with_suffix('').parts)
        if parts[0] == 'covertype-exploration':
            parts = parts[1:]
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package = parts[:-1]
                    base = package[:len(package)-node.level+1]
                    modules = ['.'.join(base + (node.module.split('.') if node.module else []))]
                else:
                    modules = [node.module or '']
            for module in modules:
                if module.split('.')[0] not in local_roots:
                    continue
                base = root/'covertype-exploration' if module.startswith('covertype_rl') else root
                if module.split('.')[0] in ('atena_common', 'atena_baselines'):
                    base = root/'ATENA_dataset'
                elif module.split('.')[0] == 'mira':
                    base = root/'ATENA_dataset/MIRA'
                candidate = base.joinpath(*module.split('.'))
                if not candidate.is_dir() and not candidate.with_suffix('.py').is_file():
                    raise ValueError('Missing local import {} in {}'.format(module, relative))
    return dict(python_sources=len(sources), python38_syntax=True, local_imports=True)


def parser_from_source(root, dataset, method):

    import copy
    from run import ABLATIONS, BASELINES
    if method in ABLATIONS or method == 'mira_no_ext':
        method = 'mira'
    if dataset == 'covertype':
        path = root/('covertype-exploration/baselines/{}/run.py'.format(method)
                     if method in ('random', 'greedy_eda') else 'covertype-exploration/covertype_rl/full_a3c.py')
    else:
        path = root/('rl/A3C_2_actors/A3C.py' if method == 'mira' else
            'baselines/greedy_eda/galaxy.py' if method == 'greedy_eda' else
            'baselines/random/random_baseline.py' if method == 'random' else
            'baselines/{}/A3C_2_actors/A3C.py'.format(method))
    scope = dict(argparse=argparse, BASELINES=set(BASELINES)-{'random', 'greedy_eda'})
    for helper in ('revision_support/ablation_config.py', 'revision_support/covertype_defaults.py'):
        nodes = ast.parse((root/helper).read_text(encoding='utf-8')).body
        wanted = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                  and n.name in ('add_ablation_arguments', 'CovertypeArgumentParser')]
        scope.update(ENCODERS=('learned', 'stats_pca', 'vae', 'frozen_random'),
                     REWARDS=('directional', 'rnd', 'count', 'knn', 'random'))
        exec(compile(ast.Module(body=wanted, type_ignores=[]), helper, 'exec'), scope)
    tree = ast.parse(path.read_text(encoding='utf-8'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('build_parser', 'parse_args')]
    if functions:
        node = copy.deepcopy(next((n for n in functions if n.name == 'build_parser'), functions[0]))
        node.body = [n for n in node.body if not isinstance(n, ast.ImportFrom)]
        node.body[-1] = ast.Return(value=ast.Name(id='parser', ctx=ast.Load()))
        compiled = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(compiled, str(path), 'exec'), scope)
        return scope[node.name]()
    nodes = [n for n in tree.body if (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'parser' for t in n.targets)) or (
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and (
            isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == 'parser'
            or isinstance(n.value.func, ast.Name) and n.value.func.id == 'add_ablation_arguments'))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['parser']


def verify_commands(root=ROOT):
    from run import build_jobs
    total = 0
    for smoke in (False, True):
        for job in build_jobs(root, smoke=smoke):
            args = parser_from_source(root, job['dataset'], job['method']).parse_args(job['command'][2:])
            if (args.episodes, args.steps, args.workers) != (job['episodes'], 250, 14):
                raise ValueError('CLI budget mismatch')
            if args.target_set != 'fixed_seed_{}'.format(job['seed']):
                raise ValueError('CLI target mismatch')
            if getattr(args, 'exploration_wup', 0) or getattr(args, 'spool_size', 0):
                raise ValueError('Release controls must remain zero')
            if job['method'] in ('atena', 'atena_no_ext') and (args.w_int, args.w_coh, args.w_div) != (.1, .1, .1):
                raise ValueError('ATENA weights differ from the protocol')
            if job['method'] == 'dora' and args.counter_curiosity_ratio != .25:
                raise ValueError('DORA ratio differs from the protocol')
            total += 1
    return dict(valid_commands=total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', action='store_true', help='Also check pinned packages and actual model updates')
    parser.add_argument('--source-only', action='store_true', help='Check source, manifest and CLI; skip data and runtime')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    report = verify_source(ROOT)
    report.update(verify_commands(ROOT))
    manifest = json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        path = ROOT/entry['path']
        if not path.is_file() or path.stat().st_size != entry['size'] or sha256(path) != entry['sha256']:
            raise ValueError('Missing/modified release file: ' + entry['path'])
    report['manifest_files'] = len(manifest['files'])
    if not args.source_only:
        from revision_support.training_runtime import configure_training_threads
        configure_training_threads()
        from revision_support.release_checks import check_data, check_runtime, check_models
        check_data()
        report['data_validated'] = True
        if args.runtime:
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            report.update(check_runtime())
            check_models()
            report['runtime_validated'] = True
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
