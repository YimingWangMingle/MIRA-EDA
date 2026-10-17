import numpy as np


def fit_projection(states, latent_dim):
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or not np.isfinite(states).all() or len(states) < 2:
        raise ValueError('PCA requires at least two finite training observations')
    latent_dim = int(latent_dim)
    if not 1 <= latent_dim <= min(len(states)-1, states.shape[1]):
        raise ValueError('PCA dimension exceeds available observations/features')
    states = np.clip(states, -5., 5.)
    mean, scale = states.mean(axis=0), states.std(axis=0)
    scale[scale < 1e-8] = 1.
    _, singular, vectors = np.linalg.svd((states-mean)/scale, full_matrices=False)
    if singular[latent_dim-1] <= 1e-10:
        raise ValueError('Training observations have insufficient rank for the requested PCA dimension')
    components = vectors[:latent_dim].copy()
    for row in components:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    return dict(mean=mean, scale=scale, components=components, n_samples=len(states))


def projection_weights(fitted, state_dim, latent_dim):
    mean, scale, components = (np.asarray(fitted[k]) for k in ('mean', 'scale', 'components'))
    if mean.shape != (state_dim,) or scale.shape != mean.shape or components.shape != (latent_dim, state_dim):
        raise ValueError('PCA artifact dimensions do not match the configured encoder')
    if not all(np.isfinite(x).all() for x in (mean, scale, components)) or np.any(scale <= 0):
        raise ValueError('Invalid PCA statistics')
    kernel = components.T / scale[:, None]
    return kernel.astype(np.float32), (-mean @ kernel).astype(np.float32)
