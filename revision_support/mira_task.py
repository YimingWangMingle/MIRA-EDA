"""Explicit task-reward boundary, shared by MIRA policy and replay adapters."""
import numpy as np


def task_reward(external, interestingness, args):
    intrinsic = float(args.alpha)*float(args.w_int)*np.asarray(interestingness,dtype=np.float32)
    if getattr(args,'baseline',None)=='mira_no_ext' or float(args.w_ext)==0.:
        return intrinsic
    return float(args.w_ext)*np.asarray(external,dtype=np.float32)+intrinsic


def replay_task_reward(components, args):
    components=np.asarray(components,dtype=np.float32).reshape(-1,2)
    reward=task_reward(components[:,0],components[:,1],args)
    if not np.isfinite(reward).all():
        raise ValueError('Nonfinite MIRA task reward')
    return reward.reshape(-1,1)
