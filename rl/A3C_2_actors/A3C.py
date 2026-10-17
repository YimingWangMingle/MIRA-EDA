import argparse
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from revision_support.inference import predict as fast_predict
from revision_support.csv_output import locked_csv_append
from revision_support.galaxy_execution import complete_training
from revision_support.ablation_config import add_ablation_arguments, validate_ablation_args, representation_options, ablation_label
from revision_support.exploration_rewards import make_reward_replacement, select_bonus
from revision_support.release_config import training_config
import ray
import tensorflow as tf
os.environ.setdefault("WANDB_MODE", "offline")
import wandb

from .mira import MIRAModule
from .critic import Critic
from .operation_actor import OperationActor
from .pipeline_environment import PipelineEnvironment
from .set_actor import SetActor

tf.keras.backend.set_floatx('float64')

now = datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=20)
parser.add_argument('--actor_lr', type=float, default=0.00003)
parser.add_argument('--critic_lr', type=float, default=0.00003)
parser.add_argument('--workers', type=int, default=14)
parser.add_argument('--episodes', type=int, default=1000)
parser.add_argument('--steps', type=int, default=250)
parser.add_argument('--strict_episode_budget', action='store_true', default=True,
                    help='Reserve episodes centrally and fail on worker errors; prevent async overshoot')
parser.add_argument('--lstm_steps', type=int, default=5)
parser.add_argument('--target_set', type=str, default=None)
parser.add_argument('--target_seed', type=int, default=None)
parser.add_argument('--seed', type=int, default=None,
                    help="Training RNG seed; separate from target_seed. Async scheduling is not deterministic.")
add_ablation_arguments(parser)
parser.add_argument('--target_samples_per_file', type=int, default=100)

parser.add_argument('--alpha', type=float, default=0.5, help="Scaling factor for Exploration Reward")
parser.add_argument('--w_ext', type=float, default=2.0, help="Weight for Extrinsic Reward")
parser.add_argument('--w_mira', type=float, default=1.0, help="Weight for MIRA directional reward")
parser.add_argument('--mira_bonus_clip', type=float, default=1.0, help="Clip absolute MIRA directional reward")
parser.add_argument('--mira_latent_dim', type=int, default=16)
parser.add_argument('--mira_lr', type=float, default=0.0003)
parser.add_argument('--mira_dense_units', type=int, default=256)
parser.add_argument('--mira_lambda_ana', type=float, default=1.0)
parser.add_argument('--mira_mc_samples', type=int, default=4)
parser.add_argument('--mira_target_tau', type=float, default=0.01)
parser.add_argument('--mira_phi_weight', type=float, default=1.0)
parser.add_argument('--mira_dyn_weight', type=float, default=1.0)
parser.add_argument('--mira_pair_warmup_episodes', type=int, default=0, choices=(0,))
parser.add_argument('--mira_replay_buffer_size', type=int, default=50000)
parser.add_argument('--mira_phi_batch_size', type=int, default=128)
parser.add_argument('--mira_min_replay_size', type=int, default=256)
parser.add_argument('--mira_orthogonal_prob', type=float, default=0.0, choices=(0.0,), help="Compatibility option locked to zero; directions are uniform on the sphere")

parser.add_argument('--notes', type=str, default="")
parser.add_argument('--mode', type=str, default="scattered")
parser.add_argument('--name', type=str, default="")
parser.add_argument('--output_dir', type=str, default="outputs")
parser.add_argument('--resume', action='store_true')
parser.add_argument('--resume_step', type=int, default=None)
parser.add_argument(
    '--operators',
    nargs='+',
    type=str,
    default=["by_facet", "by_superset", "by_neighbors", "by_distribution"],
)
args = parser.parse_args()
args.mira_core_version = "mira_grounded_latent_v2"
if args.resume or args.resume_step is not None:
    raise ValueError("MIRA v2 requires a fresh run; checkpoint continuation is not supported")
validate_ablation_args(args)
args.no_ext_protocol = "shared_task_reward_without_external_feedback" if args.w_ext == 0 else None

if args.resume_step is not None:
    args.resume = True

if args.name == "":
    args.name = (
        f"{args.mode}-mira-exploration-lstm-{args.lstm_steps}-"
        f"alr-{args.actor_lr}-clr-{args.critic_lr}-{now.strftime('%m%d%Y_%H%M%S')}"
    )

if ablation_label(args):
    args.name += '_' + ablation_label(args)
    if (Path(args.output_dir) / args.name).exists() or (Path('saved_models') / args.name).exists():
        raise ValueError('Ablation output already exists; choose a fresh --name')

args.result_dir = str(Path(args.output_dir) / args.name)
Path(args.result_dir).mkdir(parents=True, exist_ok=True)

if not args.resume:
    args.id = wandb.util.generate_id()
    if args.output_profile == 'full':
        if not os.path.exists("saved_models/" + args.name):
            os.makedirs("saved_models/" + args.name)
        with open("saved_models/" + args.name + "/info.json", 'w') as f:
            json.dump(vars(args), f, indent=1)
    with open(Path(args.result_dir) / "info.json", 'w') as f:
        json.dump(vars(args), f, indent=1)
else:
    with open("./saved_models/" + args.name + "/info.json") as f:
        items = training_config(json.load(f))
        for key in items.keys():
            if key != "resume" and key != "resume_step":
                setattr(args, key, items[key])
    args.result_dir = str(Path(args.output_dir) / args.name)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

validate_ablation_args(args)
if args.resume and args.w_ext == 0:
    raise ValueError("Revised no-ext must start clean; legacy checkpoints contain target-conditioned training")
wandb.init(name=args.name, project="deep-rl-tf2", id=args.id, resume=args.resume, config=vars(args))


@ray.remote
class ParameterServer:
    def __init__(self, set_state_dim, operation_state_dim, set_action_dim, operation_action_dim, args):
        import tensorflow as tf

        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)

        self.args = args
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            tf.random.set_seed(args.seed)
        if args.w_ext == 0 and args.mode == "by_example":
            raise ValueError("Strict no-ext does not support target-derived example states")
        self.episodes_done = args.resume_step if args.resume_step is not None else 0
        self.episodes_reserved = self.episodes_done

        model_path = None
        if args.resume:
            if args.resume_step:
                model_path = f"./saved_models/{args.name}/{args.resume_step}/"
            else:
                model_path = f"./saved_models/{args.name}/current/"

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
        self.base_set_state_dim = int(getattr(args, "base_set_state_dim", max(1, set_state_dim - int(args.mira_latent_dim))))
        self.mira_latent_dim = int(args.mira_latent_dim)
        self.z_sample_count = 0
        mira_seed = args.seed if args.seed is not None else (args.target_seed or 0)
        self.mira_rng = np.random.default_rng(int(mira_seed) + 20260621)
        self.global_sets_viewed = set()
        self.cumulative_extrinsic_reward = 0.0
        self.logged_exploration_state_ids = set()
        self.reward_replacement = make_reward_replacement(args, self.base_set_state_dim)
        self.global_mira = MIRAModule(
            state_dim=self.base_set_state_dim,
            set_action_dim=set_action_dim,
            operation_action_dim=operation_action_dim,
            latent_dim=self.mira_latent_dim,
            lr=args.mira_lr,
            dense_units=args.mira_dense_units,
            target_tau=args.mira_target_tau,
            **representation_options(args),
        )
        from revision_support.mira_learner import RepresentationLearner
        self.mira_learner = RepresentationLearner(self.global_mira, capacity=args.mira_replay_buffer_size,
            batch_size=args.mira_phi_batch_size, min_size=args.mira_min_replay_size,
            seed=int(mira_seed), gamma=args.gamma, phi_weight=args.mira_phi_weight, dyn_weight=args.mira_dyn_weight)


    def record_pca_collection(self, records):
        from revision_support.pca_training import record_collection
        return record_collection(self, records, 'galaxy')

    def sample_mira_direction(self):
        from revision_support.direction_sampling import unassisted_direction
        return unassisted_direction(self, 'galaxy')


    def update_and_sample_mira_batch(self, states, set_actions, operation_actions, task_rewards, next_states, episode):

        actions = np.column_stack([np.asarray(set_actions).reshape(-1), np.asarray(operation_actions).reshape(-1)])
        from revision_support.mira_task import replay_task_reward
        return self.mira_learner.submit(states, actions, replay_task_reward(task_rewards, self.args), next_states)

    def save_mira_snapshot(self, directory):
        self.mira_learner.check()
        with self.mira_learner.lock:
            self.global_mira.save_models(directory)


    def record_episode_metrics(self, ep_ext_score, episode_set_ids):

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
        stats["episode"] = self.increment_and_check_save()
        return stats

    def record_exploration_logs(self, episode, trace_rows, state_rows, context=None):
        context = context or {}
        result_dir = Path(getattr(self.args, "result_dir", "."))
        result_dir.mkdir(parents=True, exist_ok=True)

        if trace_rows:
            trace_file = result_dir / f"{self.args.name}_exploration_trace.csv"
            file_exists = os.path.isfile(trace_file)
            with open(trace_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        'episode',
                        'agent_id',
                        'step',
                        'set_id',
                        'step_extrinsic_reward',
                        'operator',
                        'parameter',
                        'input_set_id',
                        'operation_action',
                        'z_source',
                    ])
                for row in trace_rows:
                    writer.writerow([
                        episode,
                        row.get("agent_id", -1),
                        row.get("step", -1),
                        row.get("set_id", -1),
                        row.get("step_extrinsic_reward", 0.0),
                        row.get("operator", ""),
                        row.get("parameter", ""),
                        row.get("input_set_id", -1),
                        row.get("operation_action", -1),
                        context.get("z_source", ""),
                    ])

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
            state_file = result_dir / f"{self.args.name}_visited_set_states.csv"
            file_exists = os.path.isfile(state_file)
            with open(state_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    state_dim = len(new_state_rows[0][1])
                    writer.writerow(['set_id'] + [f'state_{i}' for i in range(state_dim)])
                for set_id, state in new_state_rows:
                    writer.writerow([set_id] + state)

        return {
            "trace_rows": len(trace_rows),
            "new_state_rows": len(new_state_rows),
        }

    def apply_gradients_and_get_weights(
        self,
        set_grads,
        op_grads,
        critic_grads,
        mira_encoder_grads=None,
        mira_dynamics_grads=None,
    ):
        if set_grads:
            self.global_set_actor.apply_gradients(set_grads)
        if op_grads:
            self.global_operation_actor.apply_gradients(op_grads)
        if critic_grads:
            self.global_critic.apply_gradients(critic_grads)
        self.mira_learner.check()
        return self.get_weights()

    def get_weights(self):
        return {
            'set': self.global_set_actor.model.get_weights(),
            'op': self.global_operation_actor.model.get_weights(),
            'critic': self.global_critic.model.get_weights(),
            'mira': self.mira_learner.weights(),
        }

    def reserve_episode(self, max_episodes):
        if self.episodes_reserved >= int(max_episodes):
            return None
        self.episodes_reserved += 1
        return self.episodes_reserved

    def replacement_reward(self, next_state):
        embed = self.mira_learner.embed if self.reward_replacement.kind == 'knn' else None
        return float(self.args.replacement_reward_scale) * float(
            self.reward_replacement.observe([next_state], embed=embed)[0])

    def record_mira_reward_state(self, worker_id, state):
        self.mira_learner.record_reward_state(worker_id, state)

    def increment_and_check_save(self):
        self.episodes_done += 1
        ep = self.episodes_done
        if ep != 0 and ep % 250 == 0 and getattr(self.args, 'output_profile', 'full') == 'full':
            self.global_operation_actor.save_model(step=ep)
            self.global_set_actor.save_model(step=ep)
            self.global_critic.save_model(step=ep)
            self.save_mira_snapshot(Path("saved_models") / self.args.name / str(ep) / "mira")
            if self.reward_replacement is not None:
                self.reward_replacement.save(Path("saved_models") / self.args.name / str(ep) / "mira")
        return ep

    def save_final_models(self):
        self.mira_learner.close()
        self.mira_learner.save_report(self.args.result_dir)
        if getattr(self.args, 'output_profile', 'full') == 'csv':
            return
        self.global_operation_actor.save_model()
        self.global_set_actor.save_model()
        self.global_critic.save_model()
        self.global_mira.save_models(Path("saved_models") / self.args.name / "final" / "mira")
        if self.reward_replacement is not None:
            self.reward_replacement.save(Path("saved_models") / self.args.name / "final" / "mira")


@ray.remote
class WorkerAgent:
    def __init__(self, pipeline, ps_handle, agentId, args, episode_steps):
        import tensorflow as tf

        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)

        self.agentId = agentId
        if args.seed is not None:
            worker_seed = int(args.seed) * 1000 + int(agentId)
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            tf.random.set_seed(worker_seed)
        self.ps = ps_handle
        self.args = args
        self.episode_steps = episode_steps

        self.env = PipelineEnvironment(
            pipeline,
            mode=args.mode,
            agentId=agentId,
            episode_steps=episode_steps,
            operators=args.operators,
            target_set_name=args.target_set,
            target_seed=args.target_seed,
            target_samples_per_file=args.target_samples_per_file,
        )
        self.base_set_state_dim = self.env.set_state_dim
        self.base_operation_state_dim = self.env.operation_state_dim
        self.mira_latent_dim = int(args.mira_latent_dim)
        self.set_state_dim = self.base_set_state_dim + self.mira_latent_dim
        self.operation_state_dim = self.base_operation_state_dim + self.mira_latent_dim
        self.set_action_dim = self.env.set_action_space.n
        self.operation_action_dim = self.env.operation_action_space.n
        self.steps = args.lstm_steps

        self.set_actor = SetActor(self.set_state_dim, self.set_action_dim, self.steps, args.actor_lr, args.name)
        self.operation_actor = OperationActor(
            self.operation_state_dim,
            self.operation_action_dim,
            self.steps,
            args.actor_lr,
            args.name,
        )
        self.critic = Critic(self.set_state_dim, self.steps, args.critic_lr, args.name)
        self.mira = MIRAModule(
            state_dim=self.base_set_state_dim,
            set_action_dim=self.set_action_dim,
            operation_action_dim=self.operation_action_dim,
            latent_dim=self.mira_latent_dim,
            lr=args.mira_lr,
            dense_units=args.mira_dense_units,
            target_tau=args.mira_target_tau,
            **representation_options(args),
        )

        self.sync_with_ps()

    def sync_with_ps(self):
        weights = ray.get(self.ps.get_weights.remote())
        self.set_actor.model.set_weights(weights['set'])
        self.operation_actor.model.set_weights(weights['op'])
        self.critic.model.set_weights(weights['critic'])
        self.mira.set_weights(weights['mira'])

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value
        for k in reversed(range(0, len(rewards))):
            cumulative = self.args.gamma * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets

    def list_to_batch(self, list_data):
        batch = list_data[0]
        for elem in list_data[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def sample_mira_direction(self):
        sample = ray.get(self.ps.sample_mira_direction.remote())
        return (
            np.array(sample["z"], dtype=np.float32),
            str(sample["z_source"]),
        )

    def sample_action(self, probs, action_dim, fallback_action=None):
        probs = np.array(probs, dtype=np.float64).flatten()
        valid_actions = np.where(np.isfinite(probs) & (probs > 0.0))[0]

        if valid_actions.size == 0:
            if fallback_action is not None:
                return int(fallback_action)
            return int(np.random.choice(action_dim))

        safe_probs = np.zeros(action_dim, dtype=np.float64)
        copy_len = min(action_dim, probs.size)
        safe_probs[:copy_len] = np.nan_to_num(probs[:copy_len], nan=0.0, posinf=0.0, neginf=0.0)
        safe_probs[safe_probs < 0.0] = 0.0
        total = float(safe_probs.sum())
        if total <= 0.0:
            if fallback_action is not None:
                return int(fallback_action)
            return int(np.random.choice(valid_actions))

        safe_probs /= total
        return int(np.random.choice(action_dim, p=safe_probs))

    def augment_set_state(self, set_state, z):
        return np.concatenate([
            np.array(set_state, dtype=np.float32).flatten(),
            np.array(z, dtype=np.float32).flatten(),
        ])

    def augment_operation_state(self, operation_state, z):
        return np.concatenate([
            np.array(operation_state, dtype=np.float32).flatten(),
            np.array(z, dtype=np.float32).flatten(),
        ])

    def embed_mira_state(self, state):
        phi = self.mira.embed(np.asarray(state, dtype=np.float32).reshape((1, self.base_set_state_dim)))
        return np.asarray(phi[0], dtype=np.float32).reshape(-1)


    def train_loop(self, max_episodes):
        curr_episode = 0

        while max_episodes >= curr_episode:
            if getattr(self.args, 'strict_episode_budget', False):
                reservation = ray.get(self.ps.reserve_episode.remote(max_episodes))
                if reservation is None:
                    break
            set_state_batch = []
            operation_state_batch = []
            set_action_batch = []
            operation_action_batch = []
            reward_batch = []
            mira_state_batch = []
            mira_next_state_batch = []
            mira_set_action_batch = []
            mira_operation_action_batch = []
            mira_task_reward_batch = []

            ep_ext_score = 0
            ep_mira_score = 0
            ep_exploration_score = 0
            ep_total_reward = 0
            episode_history_phi = []
            mira_phi_loss = 0.0
            mira_dyn_loss = 0.0
            mira_prediction_error = 0.0
            mira_mean_d_phi = 0.0
            mira_mean_metric_target = 0.0
            mira_update_count = 0
            mira_pair_source = ""
            mira_replay_size = 0
            done = False

            set_action_steps = [[0.0] * self.set_state_dim] * self.steps
            operation_action_steps = [[0.0] * self.operation_state_dim] * self.steps

            (
                mira_z,
                z_source,
            ) = self.sample_mira_direction()
            self.sync_with_ps()
            self.mira.begin_episode()
            set_state = self.env.reset()
            set_action_steps.pop(0)
            set_action_steps.append(self.augment_set_state(set_state, mira_z))
            failed = False

            try:
                while not done:
                    probs = fast_predict(self.set_actor.model,
                        np.array(set_action_steps).reshape((1, self.steps, self.set_state_dim)),
                    )
                    probs = self.env.fix_possible_set_action_probs(probs[0])
                    set_action = self.sample_action(
                        probs,
                        self.set_action_dim,
                        fallback_action=0,
                    )

                    operation_state = self.augment_operation_state(
                        self.env.get_operation_state(set_action),
                        mira_z,
                    )
                    operation_action_steps.pop(0)
                    operation_action_steps.append(operation_state)

                    probs = fast_predict(self.operation_actor.model,
                        np.array(operation_action_steps).reshape((1, self.steps, self.operation_state_dim)),
                    )
                    probs = self.env.fix_possible_operation_action_probs(set_action, probs[0])
                    operation_action = self.sample_action(
                        probs,
                        self.operation_action_dim,
                    )

                    current_set_state = np.asarray(set_state, dtype=np.float32).reshape(-1)
                    next_set_state, env_r_ext, _, done, set_op_pair = self.env.step(
                        set_action,
                        operation_action,
                    )

                    r_ext = float(np.squeeze(env_r_ext))
                    r_mira = select_bonus(
                        getattr(self.args, 'exploration_reward', 'directional'),
                        lambda: self.mira.compute_bonus(current_set_state, next_set_state, mira_z,
                                                       clip_value=self.args.mira_bonus_clip),
                        lambda: float(ray.get(self.ps.replacement_reward.remote(next_set_state))))

                    exploration_reward = self.args.w_mira * r_mira
                    effective_alpha = float(self.args.alpha)
                    total_step_reward = (self.args.w_ext * r_ext) + (effective_alpha * exploration_reward)

                    ep_ext_score += r_ext
                    ep_mira_score += r_mira
                    ep_exploration_score += exploration_reward
                    ep_total_reward += total_step_reward

                    next_set_action_steps = set_action_steps.copy()
                    next_set_action_steps.pop(0)
                    next_set_action_steps.append(self.augment_set_state(next_set_state, mira_z))

                    reward_batch.append(np.reshape(total_step_reward, [1, 1]))
                    set_state_batch.append(np.array(set_action_steps).reshape((1, self.steps, self.set_state_dim)))
                    set_action_batch.append(np.reshape(set_action, [1, 1]))
                    operation_state_batch.append(
                        np.array(operation_action_steps).reshape((1, self.steps, self.operation_state_dim))
                    )
                    operation_action_batch.append(np.reshape(operation_action, [1, 1]))
                    mira_state_batch.append(current_set_state.reshape((1, self.base_set_state_dim)))
                    mira_next_state_batch.append(np.asarray(next_set_state, dtype=np.float32).reshape((1, self.base_set_state_dim)))
                    mira_set_action_batch.append(np.reshape(set_action, [1, 1]))
                    mira_operation_action_batch.append(np.reshape(operation_action, [1, 1]))
                    metric_reward = r_ext if self.args.w_ext != 0 else 0.0
                    mira_task_reward_batch.append(np.reshape(metric_reward, [1, 1]))

                    if len(set_state_batch) >= self.args.update_interval or done:
                        set_states = self.list_to_batch(set_state_batch)
                        set_actions = self.list_to_batch(set_action_batch)
                        operation_states = self.list_to_batch(operation_state_batch)
                        operation_actions = self.list_to_batch(operation_action_batch)
                        rewards = self.list_to_batch(reward_batch)

                        next_v_value = fast_predict(self.critic.model,
                            np.array(next_set_action_steps).reshape((1, self.steps, self.set_state_dim)),
                        )
                        td_targets = self.n_step_td_target(rewards, next_v_value, done)
                        advantages = td_targets - fast_predict(self.critic.model, set_states)

                        try:
                            set_grads, _ = self.set_actor.get_gradients(set_states, set_actions, advantages)
                            op_grads, _ = self.operation_actor.get_gradients(
                                operation_states,
                                operation_actions,
                                advantages,
                            )
                            critic_grads, _ = self.critic.get_gradients(set_states, td_targets)

                            set_grads_np = [g.numpy() for g in set_grads]
                            op_grads_np = [g.numpy() for g in op_grads]
                            critic_grads_np = [g.numpy() for g in critic_grads]

                            mira_encoder_grads_np = None
                            mira_dynamics_grads_np = None
                            if mira_state_batch:
                                mira_stats = ray.get(self.ps.update_and_sample_mira_batch.remote(
                                    self.list_to_batch(mira_state_batch), self.list_to_batch(mira_set_action_batch),
                                    self.list_to_batch(mira_operation_action_batch), self.list_to_batch(mira_task_reward_batch),
                                    self.list_to_batch(mira_next_state_batch), curr_episode+1))
                                mira_pair_source = 'async_replay'
                                mira_replay_size = mira_stats['replay_size']
                                mira_phi_loss += mira_stats['phi_loss']
                                mira_dyn_loss += mira_stats['dyn_loss']
                                mira_prediction_error += mira_stats['prediction_error']
                                mira_mean_d_phi += mira_stats['mean_d_phi']
                                mira_mean_metric_target += mira_stats['mean_metric_target']
                                mira_update_count += 1

                            new_weights = ray.get(
                                self.ps.apply_gradients_and_get_weights.remote(
                                    set_grads_np,
                                    op_grads_np,
                                    critic_grads_np,
                                    mira_encoder_grads_np,
                                    mira_dynamics_grads_np,
                                )
                            )

                            self.set_actor.model.set_weights(new_weights['set'])
                            self.operation_actor.model.set_weights(new_weights['op'])
                            self.critic.model.set_weights(new_weights['critic'])
                            self.mira.set_weights(new_weights['mira'])

                        except Exception as error:
                            raise RuntimeError(
                                f'Galaxy training update failed: worker={self.agentId}, '
                                f'step={self.env.step_count}'
                            ) from error

                        set_state_batch = []
                        operation_state_batch = []
                        set_action_batch = []
                        operation_action_batch = []
                        reward_batch = []
                        mira_state_batch = []
                        mira_next_state_batch = []
                        mira_set_action_batch = []
                        mira_operation_action_batch = []
                        mira_task_reward_batch = []

                    set_state = next_set_state
                    set_action_steps = next_set_action_steps

                if failed and getattr(self.args, 'strict_episode_budget', False):
                    raise RuntimeError('Strict-budget episode gradient push failed')
                if not failed:
                    ray.get(self.ps.record_mira_reward_state.remote(
                        self.agentId, self.mira.reward_moments.state()))
                    if mira_update_count > 0:
                        update_count = float(mira_update_count)
                        mira_phi_loss /= update_count
                        mira_dyn_loss /= update_count
                        mira_prediction_error /= update_count
                        mira_mean_d_phi /= update_count
                        mira_mean_metric_target /= update_count
                    episode_set_ids = list(getattr(self.env, "sets_viewed", []))
                    episode_metrics = ray.get(self.ps.record_episode_metrics.remote(ep_ext_score, episode_set_ids))
                    curr_episode = episode_metrics["episode"]
                    trace_rows, state_rows = self.env.consume_exploration_logs()
                    ray.get(self.ps.record_exploration_logs.remote(
                        curr_episode,
                        trace_rows,
                        state_rows,
                        {
                            "z_source": z_source,
                            "mira_pair_source": mira_pair_source,
                        },
                    ))

                    print(
                        f'EP{curr_episode} Agent{self.agentId} | '
                        f'Ext_R: {ep_ext_score:.1f} | '
                        f'MIRA: {ep_mira_score:.1f} | '
                        f'Explore: {ep_exploration_score:.1f} | '
                        f'Z: {z_source}'
                    )

                    from revision_support.galaxy_rewards import write_reward
                    write_reward(self.args, dict(episode_metrics,
                        extrinsic_reward=ep_ext_score,
                        mira_bonus=ep_mira_score, exploration_reward=ep_exploration_score,
                        total_reward=ep_total_reward, z_source=z_source,
                        mira_phi_loss=mira_phi_loss, mira_dyn_loss=mira_dyn_loss,
                        mira_prediction_error=mira_prediction_error, mira_mean_d_phi=mira_mean_d_phi,
                        mira_mean_metric_target=mira_mean_metric_target,
                        mira_update_count=mira_update_count, mira_pair_source=mira_pair_source,
                        mira_replay_size=mira_replay_size))

            except Exception as error:
                raise RuntimeError(
                    f'Galaxy episode failed: worker={self.agentId}'
                ) from error


class Agent:
    def __init__(self, env_name, pipeline=None):
        self.pipeline = pipeline
        self.env_name = env_name
        self.episode_steps = args.steps
        self.num_workers = args.workers

        dummy_env = PipelineEnvironment(
            self.pipeline,
            mode=args.mode,
            episode_steps=self.episode_steps,
            operators=args.operators,
            target_set_name=args.target_set,
            target_seed=args.target_seed,
            target_samples_per_file=args.target_samples_per_file,
        )
        self.base_set_state_dim = dummy_env.set_state_dim
        self.base_operation_state_dim = dummy_env.operation_state_dim
        args.mira_statistic_indices = dummy_env.statistic_indices
        args.base_set_state_dim = self.base_set_state_dim
        args.base_operation_state_dim = self.base_operation_state_dim
        self.mira_latent_dim = int(args.mira_latent_dim)
        self.set_state_dim = self.base_set_state_dim + self.mira_latent_dim
        self.operation_state_dim = self.base_operation_state_dim + self.mira_latent_dim
        self.set_action_dim = dummy_env.set_action_space.n
        self.operation_action_dim = dummy_env.operation_action_space.n
        self.target_items = sorted(map(int, dummy_env.target_items)) if hasattr(dummy_env, "target_items") else []

        if not args.resume and len(self.target_items) > 0:
            target_snapshot_path = Path(args.result_dir) / "target_items.json"
            target_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_snapshot_path, "w") as f:
                json.dump(self.target_items, f, indent=1)
            if args.output_profile == 'full':
                saved_model_target_snapshot_path = Path("saved_models") / args.name / "target_items.json"
                saved_model_target_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                with open(saved_model_target_snapshot_path, "w") as f:
                    json.dump(self.target_items, f, indent=1)

        from revision_support.pca_training import prepare_in_budget
        self.pca_records = prepare_in_budget(dummy_env, args, 'galaxy', Path(args.result_dir))
        if self.pca_records:
            config_paths = [Path(args.result_dir) / 'info.json']
            if args.output_profile == 'full':
                config_paths.append(Path('saved_models') / args.name / 'info.json')
            for path in config_paths:
                path.write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
            wandb.config.update(dict(encoder_pca_path=args.encoder_pca_path,
                pca_protocol=args.pca_protocol), allow_val_change=True)

    def train(self, max_episodes=None):
        max_episodes = int(args.episodes if max_episodes is None else max_episodes)
        ps = ParameterServer.remote(
            self.set_state_dim,
            self.operation_state_dim,
            self.set_action_dim,
            self.operation_action_dim,
            args,
        )
        if self.pca_records:
            if max_episodes != args.episodes:
                raise ValueError('PCA total episode budget cannot change after collection')
            ray.get(ps.record_pca_collection.remote(self.pca_records))
            self.pca_records = []
        workers = [WorkerAgent.remote(self.pipeline, ps, i, args, self.episode_steps) for i in range(self.num_workers)]
        complete_training(ray, ps, workers, [worker.train_loop.remote(max_episodes) for worker in workers])
