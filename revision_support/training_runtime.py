import os


NUMERIC_THREADS = (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS',
)


def single_thread_environment(base=None):
    env = dict(os.environ if base is None else base)
    env.update({key: '1' for key in NUMERIC_THREADS})
    env['TF_NUM_INTEROP_THREADS'] = '1'
    return env


def configure_training_threads():
    os.environ.update(single_thread_environment())


def seed_training(seed, worker_id=None):
    if seed is None:
        return
    import random
    import numpy as np
    import tensorflow as tf
    value = int(seed) if worker_id is None else int(seed) * 1000 + int(worker_id)
    random.seed(value)
    np.random.seed(value)
    tf.random.set_seed(value)
