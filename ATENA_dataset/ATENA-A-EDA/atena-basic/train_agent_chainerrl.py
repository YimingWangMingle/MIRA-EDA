from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import pickle
from copy import deepcopy

from future import standard_library

import Utilities.Configuration.config as cfg
from arguments import ArchName, SchemaName
from gym_atena.global_env_prop import update_global_env_prop_from_cfg

standard_library.install_aliases()

import logging
import os
from collections import deque, defaultdict

import numpy as np

import chainer

from chainerrl.experiments.evaluator import Evaluator
from chainerrl.experiments.evaluator import save_agent
from chainerrl.misc.ask_yes_no import ask_yes_no
from os import makedirs

from gym_atena.envs.atena_env_cont import ATENAEnvCont
import gym_atena.lib.helpers as ATENAUtils

from tensorboard import summary
from tensorboardX import SummaryWriter


SUMMARY_EPISODE_SLOT = 25

summary_writer = None


def act_most_probable(agent, obs):

    xp = agent.xp
    b_state = agent.batch_states([obs], xp, agent.phi)

    if agent.obs_normalizer:
        b_state = agent.obs_normalizer(b_state, update=False)

    with chainer.using_config('train', False), chainer.no_backprop_mode():
        arch = ArchName(cfg.arch)
        if arch is ArchName.FF_GAUSSIAN:
            action = agent.model.pi.hidden_layers(b_state).data[0]
        elif arch is ArchName.FF_PARAM_SOFTMAX:
            action = agent.model(b_state)[0].most_probable.data[0]
        elif arch is ArchName.FF_SOFTMAX:
            action = agent.model(b_state)[0].most_probable.data[0]
        else:
            raise TypeError("This architecture is not supported: {}".format(cfg.arch))

    return action


def batch_act_with_mean(agent, batch_obs):

    xp = agent.xp
    b_state = agent.batch_states(batch_obs, xp, agent.phi)

    if agent.obs_normalizer:
        b_state = agent.obs_normalizer(b_state, update=False)

    with chainer.using_config('train', False), chainer.no_backprop_mode():
        arch = ArchName(cfg.arch)
        if arch is ArchName.FF_GAUSSIAN:
            actions = agent.model.pi.hidden_layers(b_state).data
        elif arch is ArchName.FF_PARAM_SOFTMAX:
            actions = agent.model(b_state)[0].most_probable.data
        else:
            raise TypeError("This architecture is not supported: {}".format(cfg.arch))

    return actions


def batch_act_and_train(self, batch_obs):
    xp = self.xp
    b_state = self.batch_states(batch_obs, xp, self.phi)

    if self.obs_normalizer:
        b_state = self.obs_normalizer(b_state, update=False)

    num_envs = len(batch_obs)
    if self.batch_last_episode is None:
        self._initialize_batch_variables(num_envs)
    assert len(self.batch_last_episode) == num_envs
    assert len(self.batch_last_state) == num_envs
    assert len(self.batch_last_action) == num_envs


    with chainer.using_config('train', False), chainer.no_backprop_mode():
        action_distrib, batch_value = self.model(b_state)
        batch_action = chainer.cuda.to_cpu(action_distrib.sample().data)
        self.entropy_record.extend(
            chainer.cuda.to_cpu(action_distrib.entropy.data))
        self.value_record.extend(chainer.cuda.to_cpu((batch_value.data)))

    self.batch_last_state = list(batch_obs)
    self.batch_last_action = list(batch_action)

    return batch_action, action_distrib


def log_results(logger, outdir, t, episode_idx, episode_r, statistics, episode_action_type_hist):

    original_logging_level = logging.INFO
    stream_handler_found = False
    for handler in logger.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            original_logging_level = handler.level
            stream_handler_found = True

    if episode_idx % SUMMARY_EPISODE_SLOT == 0:
        handler.setLevel(logging.INFO)

    logger.info('outdir:%s step:%s episode:%s R:%s',
                outdir, t, episode_idx, episode_r)
    logger.info('statistics:%s', statistics)
    if episode_idx % SUMMARY_EPISODE_SLOT == 0:
        logger.info('action types in episode: %s', episode_action_type_hist)


    if stream_handler_found:
        handler.setLevel(original_logging_level)


def save_agent_replay_buffer(agent, t, outdir, suffix='', logger=None):
    logger = logger or logging.getLogger(__name__)
    filename = os.path.join(outdir, '{}{}.replay.pkl'.format(t, suffix))
    agent.replay_buffer.save(filename)
    logger.info('Saved the current replay buffer to %s', filename)


def ask_and_save_agent_replay_buffer(agent, t, outdir, suffix=''):
    if hasattr(agent, 'replay_buffer') and \
            ask_yes_no('Replay buffer has {} transitions. Do you save them to a file?'.format(
                len(agent.replay_buffer))):
        save_agent_replay_buffer(agent, t, outdir, suffix=suffix)


def train_agent(agent, env, steps, outdir, max_episode_len=None,
                step_offset=0, evaluator=None, successful_score=None,
                step_hooks=[], logger=None):
    logger = logger or logging.getLogger(__name__)

    episode_r = 0


    episode_action_type_hist = []
    episode_idx = 0


    obs = env.reset()
    r = 0

    t = step_offset
    if hasattr(agent, 't'):
        agent.t = step_offset

    episode_len = 0
    try:
        while t < steps:


            action = agent.act_and_train(obs, r)
            if episode_idx % SUMMARY_EPISODE_SLOT == 0:

                action = env.env_prop.compressed2full_range(action)
                action_disc = env.cont2dis(action)
                episode_action_type_hist.append(action_disc[0])

            obs, r, done, info = env.step(action)
            t += 1
            episode_r += r
            episode_len += 1

            for hook in step_hooks:
                hook(env, agent, t)

            if done or episode_len == max_episode_len or t == steps:
                agent.stop_episode_and_train(obs, r, done=done)


                log_results(logger,
                            outdir,
                            t,
                            episode_idx,
                            episode_r,
                            agent.get_statistics(),
                            episode_action_type_hist
                            )

                if episode_idx % SUMMARY_EPISODE_SLOT == 0:
                    summary_writer.add_scalar('episode_reward', episode_r, episode_idx)
                    summary_writer.add_histogram('operators_hist', np.array(episode_action_type_hist), episode_idx)
                    episode_action_type_hist = []

                if evaluator is not None:
                    evaluator.evaluate_if_necessary(
                        t=t, episodes=episode_idx + 1)
                    if (successful_score is not None and
                            evaluator.max_score >= successful_score):
                        break
                if t == steps:
                    break

                episode_r = 0
                episode_idx += 1
                episode_len = 0
                obs = env.reset()
                r = 0

    except (Exception, KeyboardInterrupt):

        save_agent(agent, t, outdir, logger, suffix='_except')
        raise


    save_agent(agent, t, outdir, logger, suffix='_finish')


def train_agent_with_evaluation(agent,
                                env,
                                steps,
                                eval_n_runs,
                                eval_interval,
                                outdir,
                                max_episode_len=None,
                                step_offset=0,
                                eval_explorer=None,
                                eval_max_episode_len=None,
                                eval_env=None,
                                successful_score=None,
                                step_hooks=[],
                                save_best_so_far_agent=True,
                                logger=None,
                                ):

    global summary_writer

    logger = logger or logging.getLogger(__name__)

    makedirs(outdir, exist_ok=True)

    summary_writer = SummaryWriter(outdir)

    if eval_env is None:
        eval_env = env

    if eval_max_episode_len is None:
        eval_max_episode_len = max_episode_len

    evaluator = Evaluator(agent=agent,
                          n_steps=max_episode_len,
                          n_episodes=eval_n_runs,
                          eval_interval=eval_interval, outdir=outdir,
                          max_episode_len=eval_max_episode_len,

                          env=eval_env,
                          step_offset=step_offset,
                          save_best_so_far_agent=save_best_so_far_agent,
                          logger=logger,
                          )

    train_agent(
        agent, env, steps, outdir,
        max_episode_len=max_episode_len,
        step_offset=step_offset,
        evaluator=evaluator,
        successful_score=successful_score,
        step_hooks=step_hooks,
        logger=logger)

    summary_writer.close()


def train_agent_batch(agent, env, steps, outdir, log_interval=None,
                      max_episode_len=None, eval_interval=None,
                      step_offset=0, evaluator=None, successful_score=None,
                      step_hooks=[], return_window_size=100, logger=None,
                      use_humans_reward=False,
                      humans_reward_interval=2048):

    env_prop = update_global_env_prop_from_cfg()
    schema_name = SchemaName(cfg.schema)

    agent_humanity_rate = -1
    log_idx = 0
    logger = logger or logging.getLogger(__name__)

    arch = ArchName(cfg.arch)


    recent_returns = deque(maxlen=return_window_size)
    recent_returns_without_humanity = deque(maxlen=return_window_size)
    recent_rewards = defaultdict(lambda: deque(maxlen=return_window_size))


    entropy_values = deque(maxlen=100)


    action_types_cntr = {"back": 0, "filter": 0, "group": 0}


    actions_cntr = defaultdict(int)

    agent_humanity_info = {}

    num_envs = env.num_envs
    episode_r = np.zeros(num_envs, dtype=np.float64)
    episode_r_without_humanity = np.zeros(num_envs, dtype=np.float64)
    episode_idx = np.zeros(num_envs, dtype='i')
    episode_len = np.zeros(num_envs, dtype='i')


    obss = env.reset()
    rs = np.zeros(num_envs, dtype='f')

    t = step_offset
    if hasattr(agent, 't'):
        agent.t = step_offset

    try:
        while t < steps:


            actions, actions_distrib = batch_act_and_train(agent, obss)


            entropy_values.append(actions_distrib.entropy.data[0])


            obss, rs, dones, infos = env.step(actions)


            action_types = []
            for action in actions:
                if arch is ArchName.FF_GAUSSIAN:
                    action = env_prop.compressed2full_range(action)
                elif arch is ArchName.FF_PARAM_SOFTMAX or arch is ArchName.FF_SOFTMAX:
                    actions_cntr[action] += 1
                    action = env_prop.static_param_softmax_idx_to_action_type(action)
                action_disc = ATENAEnvCont.cont2dis(action)
                action_type = ATENAUtils.OPERATOR_TYPE_LOOKUP.get(action_disc[0])
                action_types_cntr[action_type] += 1
                action_types.append(action_type)

            episode_r_without_humanity += rs


            for i, info in enumerate(infos):
                action_type = action_types[i]
                step_reward_info = info["reward_info"]
                for reward_type, value in step_reward_info.items():
                    if (value != 0
                            or reward_type in {"back", "same_display_seen_already", "empty_display", "empty_groupings",
                                               "humanity"}
                            or (action_type == "group" and reward_type in {"compaction_gain", "diversity"})
                            or (action_type == "filter" and reward_type in {"kl_distance", "diversity"})):
                        recent_rewards[reward_type].append(value)


            episode_r += rs

            episode_len += 1


            if max_episode_len is None:
                resets = np.zeros(num_envs, dtype=bool)
            else:
                resets = (episode_len == max_episode_len)

            agent.batch_observe_and_train(obss, rs, dones, resets)


            end = np.logical_or(resets, dones)
            not_end = np.logical_not(end)


            episode_idx += end
            recent_returns.extend(episode_r[end])
            recent_returns_without_humanity.extend(episode_r_without_humanity[end])
            episode_r[end] = 0
            episode_r_without_humanity[end] = 0
            episode_len[end] = 0

            obss = env.reset(not_end)

            for _ in range(num_envs):
                t += 1
                for hook in step_hooks:
                    hook(env, agent, t)


            if (log_interval is not None
                    and t >= log_interval
                    and t % log_interval < num_envs):
                logger.info(
                    'outdir:{} step:{} episode:{} last_R: {} average_R:{}'.format(
                        outdir,
                        t,
                        np.sum(episode_idx),
                        recent_returns[-1] if recent_returns else np.nan,
                        np.mean(recent_returns) if recent_returns else np.nan,
                    ))
                summary_writer.add_scalar('episode_reward', np.mean(recent_returns), t)
                summary_writer.add_scalar('episode_r_without_humanity', np.mean(recent_returns_without_humanity), t)
                summary_writer.add_scalar('agent_humanity_rate', agent_humanity_rate, t)
                for reward_type, reward_vals in recent_rewards.items():
                    summary_writer.add_scalar(reward_type, np.mean(reward_vals), t)

                if not cfg.obs_with_step_num and cfg.stack_obs_num == 1:
                    for elem in ['success_count_per_action_type', 'failure_count_per_action_type']:
                        summary_writer.add_scalars(elem, agent_humanity_info[elem], t)

                summary_writer.add_scalars('action_types_count', action_types_cntr, t)

                log_idx += 1
                logger.info('statistics: {}'.format(agent.get_statistics()))


                summary_writer.add_scalar('avg_entropy', np.mean(entropy_values), t)

            if evaluator:
                if evaluator.evaluate_if_necessary(
                        t=t, episodes=np.sum(episode_idx)):
                    if (successful_score is not None and
                            evaluator.max_score >= successful_score):
                        break

    except (Exception, KeyboardInterrupt):


        if arch is ArchName.FF_PARAM_SOFTMAX or arch is ArchName.FF_SOFTMAX:
            actions_cntr_path = os.path.join(outdir, 'actions_cntr.pickle')
            with open(actions_cntr_path, 'wb') as handle:
                pickle.dump(actions_cntr, handle, protocol=pickle.HIGHEST_PROTOCOL)


        save_agent(agent, t, outdir, logger, suffix='_except')
        env.close()
        if evaluator:
            evaluator.env.close()


        if not cfg.obs_with_step_num and cfg.stack_obs_num == 1:
            failure_obs_path = os.path.join(outdir, 'hard_obs.pickle')
            with open(failure_obs_path, 'wb') as handle:
                pickle.dump(agent_humanity_info['failure_obs'], handle, protocol=pickle.HIGHEST_PROTOCOL)

        raise
    else:


        if arch is ArchName.FF_PARAM_SOFTMAX or arch is ArchName.FF_SOFTMAX:
            actions_cntr_path = os.path.join(outdir, 'actions_cntr.pickle')
            with open(actions_cntr_path, 'wb') as handle:
                pickle.dump(actions_cntr, handle, protocol=pickle.HIGHEST_PROTOCOL)


        save_agent(agent, t, outdir, logger, suffix='_finish')


        if not cfg.obs_with_step_num and cfg.stack_obs_num == 1:
            failure_obs_path = os.path.join(outdir, 'hard_obs.pickle')
            with open(failure_obs_path, 'wb') as handle:
                pickle.dump(agent_humanity_info['failure_obs'], handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_agent_batch_with_evaluation(agent,
                                      env,
                                      steps,
                                      eval_n_runs,
                                      eval_interval,
                                      outdir,
                                      max_episode_len=None,
                                      step_offset=0,
                                      eval_max_episode_len=None,
                                      return_window_size=100,
                                      eval_env=None,
                                      log_interval=None,
                                      successful_score=None,
                                      step_hooks=[],
                                      save_best_so_far_agent=True,
                                      logger=None,
                                      use_humans_reward=False,
                                      humans_reward_interval=2048
                                      ):

    global summary_writer

    logger = logger or logging.getLogger(__name__)

    makedirs(outdir, exist_ok=True)

    summary_writer = SummaryWriter(outdir)

    if eval_env is None:
        eval_env = env

    if eval_max_episode_len is None:
        eval_max_episode_len = max_episode_len

    evaluator = Evaluator(agent=agent,
                          n_steps=None,
                          n_episodes=eval_n_runs,
                          eval_interval=eval_interval, outdir=outdir,
                          max_episode_len=eval_max_episode_len,
                          env=eval_env,
                          step_offset=step_offset,
                          save_best_so_far_agent=save_best_so_far_agent,
                          logger=logger,
                          )

    train_agent_batch(
        agent, env, steps, outdir,
        max_episode_len=max_episode_len,
        step_offset=step_offset,
        eval_interval=eval_interval,
        evaluator=evaluator,
        successful_score=successful_score,
        return_window_size=return_window_size,
        log_interval=log_interval,
        step_hooks=step_hooks,
        logger=logger,
        use_humans_reward=use_humans_reward,
        humans_reward_interval=humans_reward_interval)

    summary_writer.close()
