"""Representation replacements shared by the three MIRA implementations."""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from .pca import projection_weights


class RepresentationVariants:
    def configure_representation(self, variant='learned', pca_path=None, vae_kl=.001, hidden=256):
        if variant not in ('learned', 'frozen_random', 'stats_pca', 'vae'):
            raise ValueError('Unknown representation variant')
        self.encoder_variant = variant
        self.encoder_vae_kl = float(vae_kl)
        if variant == 'stats_pca':
            with np.load(pca_path, allow_pickle=False) as fitted:
                kernel, bias = projection_weights(fitted, self.state_dim, self.latent_dim)
            self.encoder = tf.keras.Sequential([Input((self.state_dim,), dtype=tf.float32),
                Dense(self.latent_dim, dtype=tf.float32, kernel_initializer='zeros', bias_initializer='zeros')])
            self.target_encoder = tf.keras.models.clone_model(self.encoder)
            self.encoder.set_weights([kernel, bias])
            self.target_encoder.set_weights([kernel, bias])
        if variant in ('frozen_random', 'stats_pca'):
            self.encoder.trainable = False
            self.target_encoder.trainable = False
        if variant == 'vae':
            # Keep the original encoder as the posterior mean. The variance
            # head and decoder are auxiliary training parameters, not policy inputs.
            self.vae_logvar = tf.keras.Sequential([Input((self.latent_dim,), dtype=tf.float32),
                Dense(self.latent_dim, dtype=tf.float32)])
            self.vae_decoder = tf.keras.Sequential([Input((self.latent_dim,), dtype=tf.float32),
                Dense(hidden, activation='relu', dtype=tf.float32),
                Dense(hidden, activation='relu', dtype=tf.float32), Dense(self.state_dim, dtype=tf.float32)])
            self.vae_rng = tf.random.Generator.from_seed(1729)

    @property
    def representation_frozen(self):
        return getattr(self, 'encoder_variant', 'learned') in ('frozen_random', 'stats_pca')

    @property
    def representation_variables(self):
        variables = list(self.encoder.trainable_variables)
        if self.encoder_variant == 'vae':
            variables += self.vae_logvar.trainable_variables + self.vae_decoder.trainable_variables
        return variables

    def representation_loss(self, states, mean, metric_loss):
        if self.encoder_variant == 'vae':
            logvar = tf.clip_by_value(self.vae_logvar(mean), -10., 10.)
            latent = mean + tf.exp(.5*logvar) * self.vae_rng.normal(tf.shape(mean))
            reconstruction = self.vae_decoder(latent, training=True)
            error = tf.reduce_mean(tf.square(reconstruction-states))
            kl = -.5 * tf.reduce_mean(1. + logvar - tf.square(mean) - tf.exp(logvar))
            return error + self.encoder_vae_kl * kl
        if self.representation_frozen:
            return tf.constant(0., dtype=tf.float32)
        return metric_loss

    def representation_metadata(self):
        # Preserve the original learned-mode synchronization payload.
        result = {} if self.encoder_variant == 'learned' else dict(encoder_variant=self.encoder_variant)
        if self.encoder_variant == 'vae':
            result.update(vae_logvar=self.vae_logvar.get_weights(), vae_decoder=self.vae_decoder.get_weights())
        return result

    def restore_representation_metadata(self, weights):
        if weights.get('encoder_variant', 'learned') != self.encoder_variant:
            raise ValueError('Cannot mix encoder variants when synchronizing weights')
        if self.encoder_variant == 'vae':
            self.vae_logvar.set_weights(weights['vae_logvar'])
            self.vae_decoder.set_weights(weights['vae_decoder'])

    def save_representation_metadata(self, directory):
        directory = Path(directory)
        (directory / 'representation.json').write_text(json.dumps(dict(
            encoder_variant=self.encoder_variant, encoder_vae_kl=self.encoder_vae_kl,
            state_dim=self.state_dim, latent_dim=self.latent_dim)), encoding='utf-8')
        if self.encoder_variant == 'vae':
            self.vae_logvar.save(directory / 'vae_logvar')
            self.vae_decoder.save(directory / 'vae_decoder')
