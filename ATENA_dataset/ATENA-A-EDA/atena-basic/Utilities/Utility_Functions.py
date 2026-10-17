import logging
import os
import pickle
from copy import deepcopy

import chainerrl

from arguments import get_args, AlgoName, FilterTermsBinsSizes
from envs import make_env
from models.chinerrl_models.chainerrl_ppo import PPOchianerrl
from models.clipped_gaussian.train_ppo_gym import PPOModel
from models.clipped_gaussian.train_trpo_gym import TRPOModel

import Utilities.Configuration.config as cfg
from gym_atena.envs.atena_env_cont import ATENAEnvCont


def initialize_agent_and_env(is_test=False):

    args = get_args(is_test)


    cfg.schema = args.schema


    cfg.dataset_number = args.dataset_number


    cfg.MAX_NUM_OF_STEPS = args.episode_length

    atena_init_default_args = list(ATENAEnvCont.__init__.__defaults__)
    atena_init_default_args[0] = cfg.MAX_NUM_OF_STEPS
    ATENAEnvCont.__init__.__defaults__ = tuple(atena_init_default_args)


    cfg.stack_obs_num = args.stack_obs_num
    cfg.obs_with_step_num = args.obs_with_step_num
    cfg.no_back = args.no_back
    cfg.bins_sizes = args.bins_sizes


    cfg.no_diversity = args.no_diversity
    cfg.no_interestingness = args.no_inter
    cfg.use_humans_reward = args.use_humans_reward
    cfg.humans_reward_interval = args.humans_reward_interval
    cfg.count_data_driven = args.count_data_driven


    cfg.n_hidden_channels = args.n_hidden_channels


    cfg.arch = args.arch
    cfg.beta = args.beta


    cfg.max_nn_tokens = args.max_nn_tokens
    cfg.cache_dfs_size = args.cache_dfs_size
    cfg.cache_tokenization_size = args.cache_tokenization_size
    cfg.cache_distances_size = args.cache_distances_size


    cfg.humanity_coeff = args.humanity_coeff
    cfg.diversity_coeff = args.diversity_coeff
    cfg.kl_coeff = args.kl_coeff
    cfg.compaction_coeff = args.compaction_coeff

    args.outdir = chainerrl.experiments.prepare_output_dir(args, args.outdir)
    cfg.outdir = args.outdir


    log_path = os.path.join(args.outdir, 'training_results.log')

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        filename=log_path,
        datefmt='%H:%M:%S'
    )


    console = logging.StreamHandler()
    console.setLevel(args.logger_level)

    logging.getLogger('').addHandler(console)


    cfg.log_interval = args.log_interval
    cfg.num_envs = args.num_envs
    ATENAEnvCont.LOG_INTERVAL = int(args.log_interval / args.num_envs)


    chainerrl.misc.set_random_seed(args.seed, gpus=(args.gpu,))


    env = make_env(args, args.env, args.seed, args.render, args.outdir)


    args.algo = AlgoName(args.algo)
    if args.algo is AlgoName.CAPG_PPO:
        model = PPOModel(env,
                         args.gpu,
                         args.n_hidden_channels,
                         args.adam_lr,
                         args.ppo_update_interval,
                         args.outdir,
                         args.load,
                         args.use_clipped_gaussian)
    elif args.algo is AlgoName.CAPG_TRPO:
        model = TRPOModel(env,
                          args.gpu,
                          args.n_hidden_channels,
                          args.trpo_update_interval,
                          args.outdir,
                          args.load,
                          args.use_clipped_gaussian)
    elif args.algo == AlgoName.CHAINERRL_PPO:
        model = PPOchianerrl(args, env)
    else:
        raise NotImplementedError

    agent = model.agent

    return agent, env, args


def setup_logger(name, log_file, level=logging.INFO):


    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
