import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from revision_support.query_service import query_service_scope, set_query_report_path
from Utilities.Utility_Functions import initialize_agent_and_env
from arguments import AlgoName

from train_agent_chainerrl import train_agent_with_evaluation, train_agent_batch_with_evaluation
import chainerrl
from chainerrl import experiments

from envs import make_env, make_batch_env


def train_or_evaluate(
        args,
        agent,
        env_id,
        env,
        seed,
        render,
        eval_n_runs,
        steps,
        eval_interval,
        outdir,
):
    timestep_limit = env.spec.max_episode_steps

    if args.algo in [AlgoName.CAPG_PPO, AlgoName.CAPG_TRPO]:
        train_agent_with_evaluation(
            agent=agent,
            env=env,
            eval_env=make_env(args, env_id, seed, render, outdir, is_test=True),
            outdir=outdir,
            steps=steps,
            eval_n_runs=eval_n_runs,
            eval_interval=eval_interval,
            max_episode_len=timestep_limit,
            save_best_so_far_agent=True,
        )
    elif args.algo is AlgoName.CHAINERRL_PPO:

        def lr_setter(env, agent, value):
            agent.optimizer.alpha = value

        lr_decay_hook = experiments.LinearInterpolationHook(
            args.steps, args.adam_lr, 0, lr_setter)


        def clip_eps_setter(env, agent, value):
            agent.clip_eps = value

        clip_eps_decay_hook = experiments.LinearInterpolationHook(
            args.steps, 0.2, 0, clip_eps_setter)

        train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(args, False),
            eval_env=make_batch_env(args, True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_runs=args.eval_n_runs,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            return_window_size=args.window_size,
            max_episode_len=timestep_limit,
            save_best_so_far_agent=True,
            step_hooks=[
                lr_decay_hook,
                clip_eps_decay_hook,
            ],
            use_humans_reward=args.use_humans_reward,
            humans_reward_interval=args.humans_reward_interval,
            step_offset=args.offset_steps,
        )


if __name__ == '__main__':
    from types import SimpleNamespace
    from revision_support.query_config import configure_query_processing
    configure_query_processing(SimpleNamespace(), method='atena')
    with query_service_scope():
        agent, env, args = initialize_agent_and_env()
        set_query_report_path(Path(args.outdir) / 'query_scheduler.json')

        train_or_evaluate(args, agent,
                          args.env,
                          env,
                          args.seed,
                          args.render,
                          args.eval_n_runs,
                          args.steps,
                          args.eval_interval,
                          args.outdir)
