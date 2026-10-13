"""Validate immutable Covertype cache inputs; certify legacy contents explicitly."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

ARRAYS = {'constraints': ('constraints.npy', 'int16'),
          'set_states': ('set_states.npy', 'float32'),
          'set_sizes': ('set_sizes.npy', 'int32'),
          'graph': ('set_graph.npy', 'int32'),
          'target_offsets': ('target_offsets.npy', 'int64'),
          'target_items': ('target_items.npy', 'int64')}
VERSION = 'covertype_cache_validation_v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def builder_fingerprint():
    """Hash construction source consistently on Python 3.8-3.11.

    AST dumps vary by Python version (notably subscript nodes). Use the AST
    only to locate source segments, and include data schema constants too.
    """
    root = Path(__file__).parent
    parts = []
    for name in ('data.py', 'actions.py', 'fixed_sets.py'):
        source = (root / name).read_text(encoding='utf-8')
        if name != 'fixed_sets.py':
            parts.append(source)
            continue
        tree = ast.parse(source)
        nodes = [node for node in tree.body
                 if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign))
                 or (isinstance(node, (ast.FunctionDef, ast.ClassDef))
                     and (node.name.startswith('_') or node.name in ('Primitive', 'build_fixed_universe')))]
        parts.extend(ast.get_source_segment(source, node) for node in nodes)
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()


def _inspect(directory, labels, expected):
    from .fixed_sets import load_fixed_universe
    directory = Path(directory)
    universe = load_fixed_universe(directory)
    metadata = universe.metadata
    for key, value in expected.items():
        if key != 'n_bins' and metadata.get(key) != value:
            raise ValueError('Cache parameter mismatch: {} (cached {!r}, requested {!r})'.format(key, metadata.get(key), value))
    if metadata.get('schema') != 'fixed_set_universe_v1':
        raise ValueError('Unknown cache schema')
    if metadata.get('action_labels') != list(labels):
        raise ValueError('Cache action_labels differ from the ordered runtime actions')
    n, d, a = (int(metadata[key]) for key in ('n_sets', 'state_dim', 'action_dim'))
    if not 0 < n <= int(metadata['requested_sets']) or a != len(labels):
        raise ValueError('Invalid cache dimensions')
    widths = universe.constraints.shape
    if len(widths) != 2 or widths[0] != n or widths[1] < 4:
        raise ValueError('Invalid constraints shape')
    shapes = {'set_states': (n, d), 'set_sizes': (n,), 'graph': (n, a),
              'target_offsets': (n + 1,), 'target_items': (universe.target_items.size,)}
    records = {}
    for attr, (filename, dtype) in ARRAYS.items():
        array = getattr(universe, attr)
        if array.dtype != np.dtype(dtype) or (attr in shapes and array.shape != shapes[attr]):
            raise ValueError('Invalid shape/dtype: ' + filename)
        records[filename] = dict(sha256=sha256(directory / filename), shape=list(array.shape), dtype=str(array.dtype))
    if not np.isfinite(universe.set_states).all() or (universe.set_sizes < 0).any():
        raise ValueError('Invalid state/size values')
    if (universe.graph < 0).any() or (universe.graph >= n).any():
        raise ValueError('Graph contains out-of-range set IDs')
    offsets = universe.target_offsets
    if offsets[0] != 0 or (np.diff(offsets) < 0).any() or offsets[-1] != universe.target_items.size:
        raise ValueError('Invalid target offsets')
    if int(metadata.get('root_set_id', -1)) != 0 or not (universe.constraints[0] == -1).all():
        raise ValueError('Invalid root set')
    return universe, records


def validate_cache(directory, csv_path, target_path, action_labels, expected):
    directory = Path(directory)
    path = directory / 'validation.json'
    if not path.is_file():
        raise ValueError('Cache needs explicit content certification: ' + str(directory))
    saved = json.loads(path.read_text(encoding='utf-8'))
    if saved.get('schema') != VERSION or saved.get('builder_sha256') != builder_fingerprint():
        raise ValueError('Cache build version changed; recertification is required')
    for key, value in expected.items():
        if saved.get('parameters', {}).get(key) != value:
            raise ValueError('Cache parameter mismatch: ' + key)
    for key, source in [('csv', csv_path), ('target', target_path), ('metadata', directory / 'metadata.json')]:
        if saved.get(key + '_sha256') != sha256(source):
            raise ValueError('Cache {} fingerprint mismatch: {}'.format(key, source))
    universe, records = _inspect(directory, action_labels, expected)
    for name, record in records.items():
        if saved.get('arrays', {}).get(name) != record:
            raise ValueError('Cache array fingerprint mismatch: ' + name)
    return universe


def _graph_column(constraints, action, n_bins):
    """Vectorized equivalent of _apply_action_to_constraint for certification."""
    result = np.array(constraints, copy=True)
    n_cont = result.shape[1] - 3
    op = action.op
    if op == 'by_facet_cont': result[:, action.feature] = action.value
    elif op == 'by_superset_cont': result[:, action.feature] = -1
    elif op == 'by_neighbors_cont':
        values = np.where(result[:, action.feature] < 0, 5, result[:, action.feature])
        result[:, action.feature] = np.clip(values + action.delta, 0, 9)
    elif op.startswith('by_facet_') or op.startswith('by_superset_'):
        index = n_cont + {'cover': 0, 'wilderness': 1, 'soil': 2}[op.rsplit('_', 1)[1]]
        result[:, index] = action.value if op.startswith('by_facet_') else -1
    elif op == 'by_distribution':
        active = result[:, :n_cont] >= 0
        values = result[:, :n_cont] + action.delta
        valid = (active.sum(axis=1) > 1) & ~((active & ((values < 0) | (values >= n_bins))).any(axis=1))
        result[:, :n_cont] = np.where(active & valid[:, None], values, result[:, :n_cont])
    else:
        raise ValueError('Unknown graph action: ' + op)
    return result


def certify_cache(directory, csv_path, target_path, actions, expected, progress=None):
    """Check every cached set/state/target/edge before writing only a sidecar.

    Sampling seed is the legacy metadata declaration. Certification establishes
    consistency of all stored values with current source data and graph rules;
    it does not regenerate the selected universe or change its set IDs.
    """
    from .data import load_covertype
    from .fixed_sets import _state_for_indices
    from .targets import load_target_items
    directory = Path(directory)
    manifest = directory / 'validation.json'
    if manifest.exists():
        return validate_cache(directory, csv_path, target_path, [a.label for a in actions], expected)
    before = {key: sha256(path) for key, path in [('csv', csv_path), ('target', target_path), ('metadata', directory / 'metadata.json')]}
    data = load_covertype(csv_path, n_bins=int(expected['n_bins']))
    targets = np.asarray(load_target_items(target_path), dtype=np.int64)
    if (targets < 0).any() or (targets >= data.n_rows).any():
        raise ValueError('Target item outside dataset')
    universe, records = _inspect(directory, [a.label for a in actions], expected)
    if universe.target_set_size != len(targets):
        raise ValueError('Target count mismatch')
    columns = [data.continuous_bins[:, i] for i in range(data.continuous_bins.shape[1])]
    columns += [data.cover_type, data.wilderness, data.soil]
    indices = {(i, int(value)): np.flatnonzero(column == value)
               for i, column in enumerate(columns) for value in np.unique(column)}
    all_rows = np.arange(data.n_rows)
    target_mask = np.zeros(data.n_rows, dtype=bool); target_mask[targets] = True
    for i, constraint in enumerate(universe.constraints):
        parts = [indices.get((j, int(value)), np.empty(0, dtype=np.int64))
                 for j, value in enumerate(constraint) if value >= 0]
        parts.sort(key=len)
        rows = parts[0] if parts else all_rows
        for part in parts[1:]: rows = np.intersect1d(rows, part, assume_unique=True)
        if len(rows) != universe.size_for_set(i):
            raise ValueError('Cache set size differs from source data at set ' + str(i))
        state = _state_for_indices(data, constraint, rows)
        # Float32 reductions differ slightly across NumPy/CPU implementations
        # (including near-zero standardized means on the 581,012-row root).
        if not np.allclose(state, universe.state_for_set(i), rtol=1e-4, atol=1e-5):
            raise ValueError('Cache state differs from source data at set ' + str(i))
        if not np.array_equal(rows[target_mask[rows]], universe.targets_for_set(i)):
            raise ValueError('Cache target membership differs at set ' + str(i))
        if progress and (i + 1) % 10000 == 0: progress('sets {}/{}'.format(i + 1, universe.n_sets))
    constraints = np.asarray(universe.constraints)
    lookup = {row.tobytes(): i for i, row in enumerate(constraints)}
    if len(lookup) != universe.n_sets:
        raise ValueError('Duplicate constraints in cache')
    for j, action in enumerate(actions):
        changed = _graph_column(constraints, action, data.n_bins)
        expected_column = np.fromiter((lookup.get(row.tobytes(), i) for i, row in enumerate(changed)), dtype=np.int32, count=universe.n_sets)
        if not np.array_equal(expected_column, universe.graph[:, j]):
            raise ValueError('Cached graph differs for action ' + action.label)
        if progress and (j + 1) % 50 == 0: progress('actions {}/{}'.format(j + 1, len(actions)))
    for key, source in [('csv', csv_path), ('target', target_path), ('metadata', directory / 'metadata.json')]:
        if sha256(source) != before[key]: raise ValueError('Source changed during certification: ' + key)
    for name, record in records.items():
        if sha256(directory / name) != record['sha256']: raise ValueError('Cache changed during certification: ' + name)
    payload = dict(schema=VERSION, builder_sha256=builder_fingerprint(), parameters=dict(expected),
                   arrays=records, verification='all_set_states_target_memberships_and_graph_edges',
                   sampling_seed_provenance='existing_metadata',
                   state_validation_tolerance=dict(rtol=1e-4, atol=1e-5),
                   **{key + '_sha256': value for key, value in before.items()})
    with manifest.open('x', encoding='utf-8') as stream: json.dump(payload, stream, indent=2)
    return universe
