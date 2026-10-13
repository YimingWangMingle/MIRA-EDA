import argparse
import csv
import json
import os
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import ray
import tensorflow as tf

from rl.A3C_2_actors.critic import Critic
from rl.A3C_2_actors.operation_actor import OperationActor
from rl.A3C_2_actors.pipeline_environment import PipelineEnvironment
from rl.A3C_2_actors.set_actor import SetActor
from revision_support.inference import predict as fast_predict
from revision_support.csv_output import locked_csv_append
from revision_support.galaxy_execution import complete_training


tf.keras.backend.set_floatx("float64")


def build_parser():
    parser = argparse.ArgumentParser(description="Run Galaxy pure dual-actor A3C baseline.")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--update_interval", type=int, default=20)
    parser.add_argument("--actor_lr", type=float, default=0.00003)
    parser.add_argument("--critic_lr", type=float, default=0.00003)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lstm_steps", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--output_profile", choices=('csv', 'full'), default='csv')
    parser.add_argument("--target_set", type=str, default=None)
    parser.add_argument("--target_seed", type=int, default=None)
    parser.add_argument("--target_samples_per_file", type=int, default=100)
    parser.add_argument("--w_ext", type=float, default=2.0)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--mode", type=str, default="scattered")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_step", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument(
        "--operators",
        nargs="+",
        type=str,
        default=["by_facet", "by_superset", "by_neighbors", "by_distribution"],
    )
    return parser


def prepare_args(args):
    args.method = "pure_a3c"
    if args.resume_step is not None:
        args.resume = True
    if not args.name:
        stamp = datetime.now().strftime("%m%d%Y_%H%M%S")
        args.name = f"{args.mode}-pure-a3c-lstm-{args.lstm_steps}-{stamp}"
    args.result_dir = str(Path(args.output_dir) / args.name)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    Path("saved_models", args.name).mkdir(parents=True, exist_ok=True)
    info_path = Path("saved_models") / args.name / "info.json"
    if args.resume and info_path.exists():
        with open(info_path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get('method', 'pure_a3c') != 'pure_a3c':
            raise ValueError('Checkpoint method does not match pure A3C')
        for key, value in saved.items():
            if key not in {"resume", "resume_step"}:
                setattr(args, key, value)
    else:
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=1)
        with open(Path(args.result_dir) / "info.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=1)
    return args


@ray.remote
class ParameterServer:
    def __init__(self, set_state_dim, operation_state_dim, set_action_dim, operation_action_dim, args):
        physical_devices = tf.config.list_physical_devices("GPU")
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)

        self.args = args
        resume_episode = args.resume_step if args.resume_step is not None else 0
        self.episodes_reserved = int(resume_episode)
        self.episodes_done = int(resume_episode)
        model_path = None
        if args.resume:
            model_path = f"./saved_models/{args.name}/{args.resume_step}/" if args.resume_step else f"./saved_models/{args.name}/current/"

        self.global_set_actor = SetActor(
            set_state_dim,
            set_action_dim,
            args.lstm_steps,
            args.actor_lr,
            args.name,
            model_path=model_path + "set_actor" if model_path else None,
        )
        self.global_operation_actor = OperationActor(
            operation_state_dim,
            operation_action_dim,
            args.lstm_steps,
            args.actor_lr,
            args.name,
            model_path=model_path + "operation_actor" if model_path else None,
        )
        self.global_critic = Critic(
            set_state_dim,
            args.lstm_steps,
            args.critic_lr,
            args.name,
            model_path=model_path + "critic" if model_path else None,
        )
        self.global_sets_viewed = set()
        self.cumulative_extrinsic_reward = 0.0
        self.logged_exploration_state_ids = set()
        self.max_episodes = int(getattr(args, "episodes", 1000))

    def next_episode(self):
        if self.episodes_reserved >= self.max_episodes:
            return None
        self.episodes_reserved += 1
        return int(self.episodes_reserved)

    def record_episode_metrics(self, episode, ep_ext_score, episode_set_ids):
        self.episodes_done += 1
        completed_episode = int(self.episodes_done)
        episode_sets = set()
        for set_id in episode_set_ids:
            try:
                parsed_set_id = int(set_id)
            except (TypeError, ValueError):
                continue
            if parsed_set_id >= 0:
                episode_sets.add(parsed_set_id)

        self.global_sets_viewed.update(episode_sets)
        self.cumulative_extrinsic_reward += float(ep_ext_score)
        episode_sets_viewed = len(episode_sets)
        cumulative_unique_sets_viewed = len(self.global_sets_viewed)
        stats = {
            "sets_viewed": episode_sets_viewed,
            "cumulative_unique_sets_viewed": cumulative_unique_sets_viewed,
            "target_efficiency": float(ep_ext_score) / max(episode_sets_viewed, 1),
            "cumulative_extrinsic_reward": self.cumulative_extrinsic_reward,
            "cumulative_target_efficiency": self.cumulative_extrinsic_reward / max(cumulative_unique_sets_viewed, 1),
        }
        save_interval = int(getattr(self.args, "save_interval", 250))
        if self.args.output_profile == 'full' and save_interval > 0 and completed_episode % save_interval == 0:
            self.global_operation_actor.save_model(step=completed_episode)
            self.global_set_actor.save_model(step=completed_episode)
            self.global_critic.save_model(step=completed_episode)
        stats["episode"] = completed_episode
        stats["reserved_episode"] = int(episode)
        return stats

    def record_exploration_logs(self, episode, trace_rows, state_rows):
        result_dir = Path(getattr(self.args, "result_dir", "."))
        result_dir.mkdir(parents=True, exist_ok=True)

        if trace_rows:
            trace_file = result_dir / f"{self.args.name}_{self.args.method}_exploration_trace.csv"
            file_exists = os.path.isfile(trace_file)
            with open(trace_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(
                        [
                            "episode",
                            "agent_id",
                            "step",
                            "set_id",
                            "step_extrinsic_reward",
                            "step_interestingness",
                            "operator",
                            "parameter",
                            "input_set_id",
                            "operation_action",
                            "z_source",
                            "step_total_reward",
                        ]
                    )
                for row in trace_rows:
                    writer.writerow(
                        [
                            episode,
                            row.get("agent_id", -1),
                            row.get("step", -1),
                            row.get("set_id", -1),
                            row.get("step_extrinsic_reward", 0.0),
                            row.get("step_interestingness", 0.0),
                            row.get("operator", ""),
                            row.get("parameter", ""),
                            row.get("input_set_id", -1),
                            row.get("operation_action", -1),
                            self.args.method,
                            row.get("step_total_reward", 0.0),
                        ]
                    )

        new_state_rows = []
        for row in state_rows:
            try:
                set_id = int(row.get("set_id", -1))
            except (TypeError, ValueError):
                continue
            if set_id < 0 or set_id in self.logged_exploration_state_ids:
                continue
            state = row.get("state", [])
            if not state:
                continue
            self.logged_exploration_state_ids.add(set_id)
            new_state_rows.append((set_id, state))

        if new_state_rows:
            state_file = result_dir / f"{self.args.name}_{self.args.method}_visited_set_states.csv"
            file_exists = os.path.isfile(state_file)
            with open(state_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    state_dim = len(new_state_rows[0][1])
                    writer.writerow(["set_id"] + [f"state_{i}" for i in range(state_dim)])
                for set_id, state in new_state_rows:
                    writer.writerow([set_id] + state)

    @staticmethod
    def _cast_grads_to_model_dtype(grads, variables):
        casted_grads = []
        for grad, variable in zip(grads, variables):
            if grad is None:
                casted_grads.append(None)
            else:
                casted_grads.append(tf.convert_to_tensor(grad, dtype=variable.dtype))
        return casted_grads

    def apply_gradients_and_get_weights(self, set_grads, op_grads, critic_grads):
        if set_grads:
            self.global_set_actor.apply_gradients(
                self._cast_grads_to_model_dtype(set_grads, self.global_set_actor.model.trainable_variables)
            )
        if op_grads:
            self.global_operation_actor.apply_gradients(
                self._cast_grads_to_model_dtype(op_grads, self.global_operation_actor.model.trainable_variables)
            )
        if critic_grads:
            self.global_critic.apply_gradients(
                self._cast_grads_to_model_dtype(critic_grads, self.global_critic.model.trainable_variables)
            )
        return self.get_weights()

    def get_weights(self):
        return {
            "set": self.global_set_actor.model.get_weights(),
            "op": self.global_operation_actor.model.get_weights(),
            "critic": self.global_critic.model.get_weights(),
        }

    def save_final_models(self):
        if self.args.output_profile == 'csv':
            return
        self.global_operation_actor.save_model()
        self.global_set_actor.save_model()
        self.global_critic.save_model()


@ray.remote
class WorkerAgent:
    def __init__(self, pipeline, ps_handle, agent_id, args, episode_steps):
        physical_devices = tf.config.list_physical_devices("GPU")
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)

        self.agent_id = int(agent_id)
        self.ps = ps_handle
        self.args = args
        self.episode_steps = int(episode_steps)
        self.env = PipelineEnvironment(
            pipeline,
            mode=args.mode,
            agentId=self.agent_id,
            episode_steps=self.episode_steps,
            operators=args.operators,
            target_set_name=args.target_set,
            target_seed=args.target_seed,
            target_samples_per_file=args.target_samples_per_file,
        )
        self.set_state_dim = self.env.set_state_dim
        self.operation_state_dim = self.env.operation_state_dim
        self.set_action_dim = self.env.set_action_space.n
        self.operation_action_dim = self.env.operation_action_space.n
        self.steps = int(args.lstm_steps)
        self.set_actor = SetActor(self.set_state_dim, self.set_action_dim, self.steps, args.actor_lr, args.name)
        self.operation_actor = OperationActor(self.operation_state_dim, self.operation_action_dim, self.steps, args.actor_lr, args.name)
        self.critic = Critic(self.set_state_dim, self.steps, args.critic_lr, args.name)
        self.sync_with_ps()

    def sync_with_ps(self):
        weights = ray.get(self.ps.get_weights.remote())
        self.set_actor.model.set_weights(weights["set"])
        self.operation_actor.model.set_weights(weights["op"])
        self.critic.model.set_weights(weights["critic"])

    def n_step_td_target(self, rewards, next_v_value, done):
        rewards = np.asarray(rewards, dtype=np.float64)
        td_targets = np.zeros_like(rewards, dtype=np.float64)
        cumulative = 0.0
        if not done:
            cumulative = float(np.squeeze(next_v_value))
        for index in reversed(range(0, len(rewards))):
            cumulative = float(self.args.gamma) * cumulative + float(np.squeeze(rewards[index]))
            td_targets[index, 0] = cumulative
        return td_targets.astype(np.float64)

    @staticmethod
    def list_to_batch(list_data):
        batch = list_data[0]
        for item in list_data[1:]:
            batch = np.append(batch, item, axis=0)
        return batch

    @staticmethod
    def sample_action(probs, action_dim, fallback_action=None):
        probs = np.asarray(probs, dtype=np.float64).flatten()
        if all(np.isnan(value) for value in probs):
            return int(fallback_action if fallback_action is not None else np.random.choice(action_dim))
        return int(np.random.choice(action_dim, p=probs))

    def train_loop(self):
        while True:
            episode = ray.get(self.ps.next_episode.remote())
            if episode is None:
                break
            set_state_batch = []
            operation_state_batch = []
            set_action_batch = []
            operation_action_batch = []
            reward_batch = []
            ep_ext_score = 0.0
            ep_int_score = 0.0
            ep_total_reward = 0.0
            done = False
            failed = False
            set_action_steps = [[0.0] * self.set_state_dim] * self.steps
            operation_action_steps = [[0.0] * self.operation_state_dim] * self.steps
            set_state = self.env.reset()
            set_action_steps.pop(0)
            set_action_steps.append(set_state)

            try:
                while not done:
                    set_probs = fast_predict(self.set_actor.model,
                        np.asarray(set_action_steps, dtype=np.float64).reshape((1, self.steps, self.set_state_dim)),
                    )
                    set_probs = self.env.fix_possible_set_action_probs(set_probs[0])
                    set_action = self.sample_action(set_probs, self.set_action_dim, fallback_action=0)

                    operation_state = self.env.get_operation_state(set_action)
                    operation_action_steps.pop(0)
                    operation_action_steps.append(operation_state)
                    operation_probs = fast_predict(self.operation_actor.model,
                        np.asarray(operation_action_steps, dtype=np.float64).reshape((1, self.steps, self.operation_state_dim)),
                    )
                    operation_probs = self.env.fix_possible_operation_action_probs(set_action, operation_probs[0])
                    operation_action = self.sample_action(operation_probs, self.operation_action_dim)

                    next_set_state, env_r_ext, env_r_int_js, done, _ = self.env.step(set_action, operation_action)
                    r_ext = float(np.squeeze(env_r_ext))
                    r_int = float(np.squeeze(env_r_int_js))
                    total_step_reward = float(self.args.w_ext) * r_ext

                    ep_ext_score += r_ext
                    ep_int_score += r_int
                    ep_total_reward += total_step_reward

                    next_set_action_steps = list(set_action_steps)
                    next_set_action_steps.pop(0)
                    next_set_action_steps.append(next_set_state)
                    reward_batch.append(np.asarray([[total_step_reward]], dtype=np.float64))
                    set_state_batch.append(np.asarray(set_action_steps, dtype=np.float64).reshape((1, self.steps, self.set_state_dim)))
                    set_action_batch.append(np.asarray([[set_action]], dtype=np.int32))
                    operation_state_batch.append(
                        np.asarray(operation_action_steps, dtype=np.float64).reshape((1, self.steps, self.operation_state_dim))
                    )
                    operation_action_batch.append(np.asarray([[operation_action]], dtype=np.int32))

                    if self.env.exploration_trace_rows:
                        self.env.exploration_trace_rows[-1]["step_total_reward"] = float(total_step_reward)

                    if len(set_state_batch) >= int(self.args.update_interval) or done:
                        set_states = self.list_to_batch(set_state_batch)
                        set_actions = self.list_to_batch(set_action_batch)
                        operation_states = self.list_to_batch(operation_state_batch)
                        operation_actions = self.list_to_batch(operation_action_batch)
                        rewards = self.list_to_batch(reward_batch)
                        next_v_value = fast_predict(self.critic.model,
                            np.asarray(next_set_action_steps, dtype=np.float64).reshape((1, self.steps, self.set_state_dim)),
                        )
                        td_targets = self.n_step_td_target(rewards, next_v_value, done)
                        advantages = (
                            td_targets - np.asarray(fast_predict(self.critic.model, set_states), dtype=np.float64)
                        ).astype(np.float64)
                        try:
                            set_grads, _ = self.set_actor.get_gradients(set_states, set_actions, advantages)
                            op_grads, _ = self.operation_actor.get_gradients(operation_states, operation_actions, advantages)
                            critic_grads, _ = self.critic.get_gradients(set_states, td_targets)
                            new_weights = ray.get(
                                self.ps.apply_gradients_and_get_weights.remote(
                                    [None if grad is None else grad.numpy().astype(np.float64) for grad in set_grads],
                                    [None if grad is None else grad.numpy().astype(np.float64) for grad in op_grads],
                                    [None if grad is None else grad.numpy().astype(np.float64) for grad in critic_grads],
                                )
                            )
                            self.set_actor.model.set_weights(new_weights["set"])
                            self.operation_actor.model.set_weights(new_weights["op"])
                            self.critic.model.set_weights(new_weights["critic"])
                        except Exception as error:
                            print(error)
                            traceback.print_tb(error.__traceback__)
                            raise RuntimeError("Episode gradient push failed") from error

                        set_state_batch = []
                        operation_state_batch = []
                        set_action_batch = []
                        operation_action_batch = []
                        reward_batch = []

                    set_action_steps = next_set_action_steps
                    set_state = next_set_state

                if not failed:
                    episode_set_ids = list(getattr(self.env, "sets_viewed", []))
                    metrics = ray.get(self.ps.record_episode_metrics.remote(episode, ep_ext_score, episode_set_ids))
                    curr_episode = int(metrics["episode"])
                    trace_rows, state_rows = self.env.consume_exploration_logs()
                    ray.get(self.ps.record_exploration_logs.remote(curr_episode, trace_rows, state_rows))
                    self._write_reward_row(curr_episode, ep_ext_score, ep_int_score, ep_total_reward, metrics)
                    print(
                        f"EP{curr_episode} {self.args.method} Agent{self.agent_id} | "
                        f"Ext_R: {ep_ext_score:.1f} | Total: {ep_total_reward:.1f}",
                        flush=True,
                    )
            except Exception as error:
                print(error)
                traceback.print_tb(error.__traceback__)
                raise RuntimeError("Episode failed") from error

    def _write_reward_row(self, episode, ep_ext_score, ep_int_score, ep_total_reward, metrics):
        result_dir = Path(getattr(self.args, "result_dir", "."))
        result_dir.mkdir(parents=True, exist_ok=True)
        csv_file = result_dir / f"{self.args.name}_{self.args.method}_rewards.csv"
        file_exists = os.path.isfile(csv_file)
        with locked_csv_append(csv_file) as (f, file_exists):
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "episode",
                        "extrinsic_reward",
                        "interestingness",
                        "exploration_reward",
                        "total_reward",
                        "sets_viewed",
                        "cumulative_unique_sets_viewed",
                        "target_efficiency",
                        "cumulative_extrinsic_reward",
                        "cumulative_target_efficiency",
                    ]
                )
            writer.writerow(
                [
                    episode,
                    ep_ext_score,
                    ep_int_score,
                    ep_total_reward - float(self.args.w_ext) * ep_ext_score,
                    ep_total_reward,
                    metrics["sets_viewed"],
                    metrics["cumulative_unique_sets_viewed"],
                    metrics["target_efficiency"],
                    metrics["cumulative_extrinsic_reward"],
                    metrics["cumulative_target_efficiency"],
                ]
            )


class Agent:
    def __init__(self, env_name, pipeline=None, args=None):
        self.args = prepare_args(args or build_parser().parse_args())
        self.pipeline = pipeline
        self.env_name = env_name
        self.episode_steps = self.args.steps
        self.num_workers = int(self.args.workers)
        dummy_env = PipelineEnvironment(
            self.pipeline,
            mode=self.args.mode,
            episode_steps=self.episode_steps,
            operators=self.args.operators,
            target_set_name=self.args.target_set,
            target_seed=self.args.target_seed,
            target_samples_per_file=self.args.target_samples_per_file,
        )
        self.set_state_dim = dummy_env.set_state_dim
        self.operation_state_dim = dummy_env.operation_state_dim
        self.set_action_dim = dummy_env.set_action_space.n
        self.operation_action_dim = dummy_env.operation_action_space.n
        self.target_items = sorted(map(int, dummy_env.target_items)) if hasattr(dummy_env, "target_items") else []
        if not self.args.resume and self.target_items:
            for path in (
                Path(self.args.result_dir) / "target_items.json",
                Path("saved_models") / self.args.name / "target_items.json",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.target_items, f, indent=1)

    def train(self, max_episodes=None):
        max_episodes = int(self.args.episodes if max_episodes is None else max_episodes)
        self.args.episodes = max_episodes
        ps = ParameterServer.remote(
            self.set_state_dim,
            self.operation_state_dim,
            self.set_action_dim,
            self.operation_action_dim,
            self.args,
        )
        workers = [
            WorkerAgent.remote(self.pipeline, ps, worker_id, self.args, self.episode_steps)
            for worker_id in range(self.num_workers)
        ]
        complete_training(ray, ps, workers, [worker.train_loop.remote() for worker in workers])
