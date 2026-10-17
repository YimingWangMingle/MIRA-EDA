import numpy as np
from .mira_core import MIRACore


class DualActionMIRA(MIRACore):
    def __init__(self, state_dim, set_action_dim, operation_action_dim, latent_dim=16,
                 lr=3e-4, dense_units=256, target_tau=.01, encoder_variant='learned',
                 encoder_pca_path=None, encoder_vae_kl=.001, statistic_indices=None,
                 lambda_ana=1., mc_samples=4):
        self.set_action_dim, self.operation_action_dim = int(set_action_dim), int(operation_action_dim)
        super().__init__(state_dim,self.set_action_dim+self.operation_action_dim,
            latent_dim,dense_units,lr,target_tau,statistic_indices,lambda_ana,mc_samples,
            encoder_variant,encoder_pca_path,encoder_vae_kl)

    def action_features(self, set_actions, operation_actions):
        choices = np.column_stack([np.asarray(set_actions).reshape(-1),np.asarray(operation_actions).reshape(-1)])
        return self.encode_actions(choices)

    def encode_actions(self, actions):
        actions = np.asarray(actions,dtype=np.int64).reshape(-1,2)
        result = np.zeros((len(actions),self.action_dim),np.float32)
        for col,dim,offset in [(0,self.set_action_dim,0),(1,self.operation_action_dim,self.set_action_dim)]:
            if np.any(actions[:,col]<0) or np.any(actions[:,col]>=dim):
                raise ValueError('MIRA replay action outside declared schema')
            result[np.arange(len(actions)),offset+actions[:,col]] = 1.
        return result
