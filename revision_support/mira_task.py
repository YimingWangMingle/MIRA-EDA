import numpy as np


def task_reward(external, args):
    external = np.asarray(external, dtype=np.float32)
    if getattr(args, 'baseline', None) == 'mira_no_ext' or float(args.w_ext) == 0.:
        return np.zeros_like(external)
    return float(args.w_ext) * external


def replay_task_reward(external, args):
    reward = task_reward(external, args)
    if not np.isfinite(reward).all():
        raise ValueError('Nonfinite MIRA task reward')
    return reward.reshape(-1, 1)
