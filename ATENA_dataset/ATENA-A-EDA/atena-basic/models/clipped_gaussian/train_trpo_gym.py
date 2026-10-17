from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import *

from future import standard_library

from models.clipped_gaussian.clipped_model import ClippedModel

standard_library.install_aliases()

import os

import chainer
from chainer import functions as F
import gym
import gym.wrappers


import numpy as np

import chainerrl

from models.clipped_gaussian.clipped_gaussian import ClippedGaussianPolicy


class TRPOModel(ClippedModel):
    def __init__(
            self,
            env,
            gpu,
            n_hidden_channels,
            trpo_update_interval,
            outdir,
            load,
            use_clipped_gaussian=True
    ):
        super().__init__()
        obs_space = env.observation_space
        action_space = env.action_space
        print('Observation space:', obs_space)
        print('Action space:', action_space, action_space.low, action_space.high)

        if not isinstance(obs_space, gym.spaces.Box):
            print("""\
    This example only supports gym.spaces.Box observation spaces. To apply it to
    other observation spaces, use a custom phi function that convert an observation
    to numpy.ndarray of numpy.float32.""")
            return


        def var_func(x): return F.exp(x) ** 2


        obs_normalizer = chainerrl.links.EmpiricalNormalization(
            obs_space.low.size)

        assert isinstance(action_space, gym.spaces.Box)

        if use_clipped_gaussian:
            policy = \
                ClippedGaussianPolicy(
                    obs_space.low.size,
                    action_space.low.size,
                    n_hidden_channels=n_hidden_channels,
                    n_hidden_layers=2,
                    mean_wscale=0.01,
                    nonlinearity=F.tanh,
                    var_type='diagonal',
                    var_func=var_func,
                    var_param_init=0,
                    min_action=action_space.low.astype(np.float32),
                    max_action=action_space.high.astype(np.float32),
                )
        else:
            policy = \
                chainerrl.policies.FCGaussianPolicyWithStateIndependentCovariance(
                    obs_space.low.size,
                    action_space.low.size,
                    n_hidden_channels=n_hidden_channels,
                    n_hidden_layers=2,
                    mean_wscale=0.01,
                    nonlinearity=F.tanh,
                    var_type='diagonal',
                    var_func=var_func,
                    var_param_init=0,
                )


        vf = chainerrl.v_functions.FCVFunction(
            obs_space.low.size,
            n_hidden_channels=n_hidden_channels,
            n_hidden_layers=2,
            last_wscale=0.01,
            nonlinearity=F.tanh,
        )

        if gpu >= 0:
            chainer.cuda.get_device_from_id(gpu).use()
            policy.to_gpu(gpu)
            vf.to_gpu(gpu)
            obs_normalizer.to_gpu(gpu)


        vf_opt = chainer.optimizers.Adam()
        vf_opt.setup(vf)


        fake_obs = chainer.Variable(
            policy.xp.zeros_like(obs_space.low, dtype=np.float32)[None],
            name='observation')
        chainerrl.misc.draw_computational_graph(
            [policy(fake_obs)], os.path.join(outdir, 'policy'))
        chainerrl.misc.draw_computational_graph(
            [vf(fake_obs)], os.path.join(outdir, 'vf'))


        agent = chainerrl.agents.TRPO(
            policy=policy,
            vf=vf,
            vf_optimizer=vf_opt,
            obs_normalizer=obs_normalizer,
            phi=lambda x: x.astype(np.float32, copy=False),
            update_interval=trpo_update_interval,
            conjugate_gradient_max_iter=20,
            conjugate_gradient_damping=1e-1,
            gamma=0.995,
            lambd=0.97,
            vf_epochs=5,
            entropy_coef=0,
        )

        if load:
            agent.load(load)

        self._agent = agent
