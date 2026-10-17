import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_data():
    import numpy as np
    manifest = ROOT/'manifest.json'
    if manifest.exists():
        entries = json.loads(manifest.read_text())['files']
        for item in entries:
            path = ROOT/item['path']
            if not path.is_file() or path.stat().st_size != item['size'] or sha256(path) != item['sha256']:
                raise ValueError('Missing/modified bundle file: '+str(path))
        print('Verified {} bundled file checksums'.format(len(entries)), flush=True)
    else:
        print('Source checkout: no bundle manifest; validating data structure', flush=True)
    for name in ('galaxies.csv', 'bins.json', 'column_forced_types.csv',
                 'columns_to_discretize.csv', 'foreign_keys.csv',
                 'galaxies_index/groups.csv', 'galaxies_index/correspondences.csv'):
        path = ROOT/'app/data/sdss'/name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('Missing Galaxy data: '+str(path))
    if not (ROOT/'covertype-exploration/covertype.csv').is_file():
        raise ValueError('Missing Covertype CSV')
    sys.path.insert(0, str(ROOT/'covertype-exploration'))
    from covertype_rl.actions import build_action_space
    from covertype_rl.cache_contract import validate_cache
    labels = [a.label for a in build_action_space(n_continuous=10, n_bins=10)]
    for seed in (1, 2, 3):
        galaxy = json.loads((ROOT/'rl/targets'/('fixed_seed_{}.json'.format(seed))).read_text())
        if not galaxy:
            raise ValueError('Empty Galaxy target set')
        targets = json.loads((ROOT/'covertype-exploration/targets'/('fixed_seed_{}.json'.format(seed))).read_text())
        universe = validate_cache(ROOT/'covertype-exploration/preprocessed'/'by_distribution_path100k_seed{}'.format(seed),
            ROOT/'covertype-exploration/covertype.csv', ROOT/'covertype-exploration/targets'/'fixed_seed_{}.json'.format(seed),
            labels, dict(seed=seed, requested_sets=100000, min_set_size=10, max_set_size=200000, n_bins=10))
        if (universe.set_states.shape != (100000, 84) or universe.graph.shape != (100000, 202)
            or universe.metadata['seed'] != seed or universe.metadata['action_labels'] != labels
            or universe.target_offsets.shape != (100001,)
            or universe.target_set_size != len(targets['target_items'])
            or not np.isin(universe.target_items, targets['target_items']).all()
            or not np.isfinite(universe.set_states).all()):
            raise ValueError('Covertype seed {} cached universe mismatch'.format(seed))
        print('Validated Galaxy/Covertype seed {}'.format(seed), flush=True)


def check_runtime():
    if platform.system() != 'Linux' or not (3, 8) <= sys.version_info[:2] <= (3, 10):
        raise RuntimeError('Use Linux with Python 3.8-3.10; recommended Python 3.9 for this TF 2.9 stack')
    required = dict(tensorflow='2.9.0', ray='2.9.0', numpy='1.23.5', pandas='1.4.4',
                    scipy='1.9.3', psutil='5.9.5', gym='0.25.2', protobuf='3.19.6')
    versions = {name: metadata.version(name) for name in required}
    if versions != required:
        raise RuntimeError('Runtime differs from requirements.txt: '+str(versions))
    import psutil
    import tensorflow as tf
    import ray
    affinity = len(psutil.Process().cpu_affinity())
    if affinity < 14:
        raise RuntimeError('Only {} CPUs are available to this process; the default suite requires 14 workers'.format(affinity))
    return dict(platform=platform.platform(), python=sys.version, cpu=platform.processor(),
                logical_cpus=os.cpu_count(), affinity_cpus=affinity,
                physical_cpus=psutil.cpu_count(logical=False),
                total_memory_bytes=psutil.virtual_memory().total, packages=versions,
                visible_tensorflow_devices=[d.name for d in tf.config.list_physical_devices()],
                training_device='CPU; workers recorded in runtime.json; one numerical compute thread per worker',
                ray_version=ray.__version__)


def check_models():
    import numpy as np
    import tensorflow as tf
    sys.path.insert(0, str(ROOT/'covertype-exploration'))
    from rl.A3C_2_actors.mira import MIRAModule as GalaxyMIRA
    from covertype_rl.mira import MIRAModule as CovertypeMIRA
    from revision_support.pca import fit_projection
    states = np.random.default_rng(1).normal(size=(16, 3)).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix='mira_preflight_') as temporary:
        artifact = Path(temporary)/'pca.npz'
        np.savez(artifact, **fit_projection(states, 2))
        for cls in (GalaxyMIRA, CovertypeMIRA):
            for variant in ('learned', 'stats_pca', 'vae', 'frozen_random'):
                tf.random.set_seed(1)
                model = cls(3, 2, 2, latent_dim=2, dense_units=8,
                            statistic_indices=[0, 1], encoder_variant=variant, encoder_pca_path=str(artifact))
                before = model.get_weights()

                np.testing.assert_allclose(model.embed(states),
                    model.encoder.predict(states, verbose=0), rtol=1e-5, atol=1e-6)
                model.train_batch(states, model.action_features([0]*16, [1]*16),
                                  [1.]*16, states[::-1].copy())
                if not np.isfinite(model.embed(states)).all():
                    raise ValueError('Nonfinite encoder output: '+variant)
                if variant in ('stats_pca', 'frozen_random'):
                    for key in ('encoder', 'target_encoder'):
                        for old, new in zip(before[key], model.get_weights()[key]):
                            np.testing.assert_array_equal(old, new)
                print('Validated {} {}'.format(cls.__module__, variant), flush=True)
