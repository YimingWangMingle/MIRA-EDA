import numpy as np


def unassisted_direction(owner, backend):
    if backend == 'covertype' and not owner.is_mira:
        return dict(z=[], z_source='none')
    from .mira_core import sample_direction
    direction = sample_direction(owner.mira_rng, owner.mira_latent_dim if backend == 'galaxy' else owner.args.mira_latent_dim)
    owner.z_sample_count += 1
    return dict(z=direction.tolist() if backend == 'covertype' else direction, z_source='uniform_sphere')
