import argparse
import os

from .training_runtime import configure_training_threads


class CovertypeArgumentParser(argparse.ArgumentParser):
    def parse_known_args(self, args=None, namespace=None):
        args, unknown = super().parse_known_args(args, namespace)
        method = getattr(args, 'baseline', None)
        if hasattr(args, 'w_int'):
            if method in ('mira', 'mira_no_ext'):
                if args.w_int is not None:
                    self.error('--w_int is only available for ATENA baselines')
                del args.w_int
            elif args.w_int is None:
                args.w_int = .1 if method in ('atena', 'atena_no_ext') else 0.
        if hasattr(args, 'w_ext'):
            if method in ('mira_no_ext', 'atena_no_ext'):
                args.w_ext = 0.
            elif args.w_ext is None:
                args.w_ext = 2.
        seed = int(args.seed)
        if args.target_set is None:
            args.target_set = 'fixed_seed_{}'.format(seed)
        if hasattr(args, 'target_seed') and args.target_seed is None:
            args.target_seed = seed


        official = (seed in (1, 2, 3) and args.target_set == 'fixed_seed_{}'.format(seed)
                    and getattr(args, 'target_seed', seed) == seed
                    and getattr(args, 'csv_path', 'covertype.csv') == 'covertype.csv'
                    and all(getattr(args, key, value) == value for key, value in
                            [('n_sets', 100000), ('n_bins', 10), ('target_size', 1000),
                             ('min_set_size', 10), ('max_set_size', 200000)]))
        if args.preprocess_name is None and official:
            args.preprocess_name = 'by_distribution_path100k_seed{}'.format(seed)
        return args, unknown


def configure_covertype_runtime():

    configure_training_threads()
    for key, value in [('CUDA_VISIBLE_DEVICES', ''), ('RAY_ADDRESS', 'local'),
                       ('WANDB_MODE', 'disabled'), ('PYTHONUNBUFFERED', '1')]:
        os.environ.setdefault(key, value)
