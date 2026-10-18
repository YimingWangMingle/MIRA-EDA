import argparse
import csv
import json
import os
import sys
import random
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from revision_support.query_service import shared_query_run, set_query_report_path
from revision_support.query_config import query_actor
from revision_support.covertype_defaults import configure_covertype_runtime

configure_covertype_runtime()

from revision_support.resources import flush_events, query_step

import numpy as np
from revision_support.inference import predict as fast_predict
from revision_support.ablation_config import add_ablation_arguments, validate_ablation_args, representation_options, ablation_label
from revision_support.exploration_rewards import make_reward_replacement, select_bonus
from revision_support.galaxy_execution import complete_training
import ray
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM

from .actions import build_action_space
from .mira import MIRAModule
from .data import load_covertype
from .environment import FixedSetEnvironment
from .fixed_sets import ensure_fixed_universe, load_fixed_universe
from .targets import resolve_target_set


tf.keras.backend.set_floatx("float32")


MIRA_BASELINES = {"mira", "mira_no_ext"}
BASELINES = {"dora", "atena", "atena_no_ext", "pure_a3c", *MIRA_BASELINES}


class SetActor:
    def __init__(self, state_dim, action_dim, steps, lr, name, model_path=None, dense_units=512, lstm_units=512):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.steps = int(steps)
        self.name = name
        self.dense_units = int(dense_units)
        self.lstm_units = int(lstm_units)
        self.opt = tf.keras.optimizers.Adam(learning_rate=float(lr))
        self.entropy_beta = 0.05
        self.model = tf.keras.models.load_model(model_path) if model_path else self.create_model()

    def create_model(self):
        return tf.keras.Sequential(
            [
                Input((self.steps, self.state_dim)),
                Dense(self.dense_units, activation="relu"),
                Dense(self.dense_units, activation="relu"),
                LSTM(self.lstm_units, return_sequences=False),
                Dense(self.action_dim, activation="softmax"),
            ]
        )

    def compute_loss(self, actions, logits, advantages):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        policy_loss = ce_loss(tf.cast(actions, tf.int32), logits, sample_weight=tf.stop_gradient(advantages))
        entropy = entropy_loss(logits, logits)
        return policy_loss - self.entropy_beta * entropy

    def get_gradients(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            loss = self.compute_loss(actions, logits, advantages)
            grads = tape.gradient(loss, self.model.trainable_variables)
        return grads, loss

    def apply_gradients(self, grads):
        clean = []
        for grad, variable in zip(grads, self.model.trainable_variables):
            if grad is not None:
                clean.append((grad, variable))
        if clean:
            self.opt.apply_gradients(clean)

    def save_model(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.model.save(directory)


class OperationActor:
    def __init__(self, state_dim, action_dim, steps, lr, name, model_path=None, dense_units=512, lstm_units=256):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.steps = int(steps)
        self.name = name
        self.dense_units = int(dense_units)
        self.lstm_units = int(lstm_units)
        self.opt = tf.keras.optimizers.Adam(learning_rate=float(lr))
        self.entropy_beta = 0.05
        self.model = tf.keras.models.load_model(model_path) if model_path else self.create_model()

    def create_model(self):
        return tf.keras.Sequential(
            [
                Input((self.steps, self.state_dim)),
                Dense(self.dense_units, activation="relu"),
                Dense(self.dense_units, activation="relu"),
                LSTM(self.lstm_units, return_sequences=False),
                Dense(self.action_dim, activation="softmax"),
            ]
        )

    def compute_loss(self, actions, logits, advantages):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        policy_loss = ce_loss(tf.cast(actions, tf.int32), logits, sample_weight=tf.stop_gradient(advantages))
        entropy = entropy_loss(logits, logits)
        return policy_loss - self.entropy_beta * entropy

    def get_gradients(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            loss = self.compute_loss(actions, logits, advantages)
            grads = tape.gradient(loss, self.model.trainable_variables)
        return grads, loss

    def apply_gradients(self, grads):
        clean = []
        for grad, variable in zip(grads, self.model.trainable_variables):
            if grad is not None:
                clean.append((grad, variable))
        if clean:
            self.opt.apply_gradients(clean)

    def save_model(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.model.save(directory)


class Critic:
    def __init__(self, state_dim, steps, lr, name, model_path=None, dense_units=(512, 256, 128), lstm_units=128):
        self.state_dim = int(state_dim)
        self.steps = int(steps)
        self.name = name
        self.dense_units = tuple(int(unit) for unit in dense_units)
        self.lstm_units = int(lstm_units)
        self.opt = tf.keras.optimizers.Adam(learning_rate=float(lr))
        self.model = tf.keras.models.load_model(model_path) if model_path else self.create_model()

    def create_model(self):
        return tf.keras.Sequential(
            [
                Input((self.steps, self.state_dim)),
                Dense(self.dense_units[0], activation="relu"),
                Dense(self.dense_units[1], activation="relu"),
                Dense(self.dense_units[2], activation="relu"),
                LSTM(self.lstm_units, return_sequences=False),
                Dense(1, activation="linear"),
            ]
        )

    def get_gradients(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            loss = tf.keras.losses.MeanSquaredError()(tf.stop_gradient(td_targets), v_pred)
            grads = tape.gradient(loss, self.model.trainable_variables)
        return grads, loss

    def apply_gradients(self, grads):
        clean = []
        for grad, variable in zip(grads, self.model.trainable_variables):
            if grad is not None:
                clean.append((grad, variable))
        if clean:
            self.opt.apply_gradients(clean)

    def save_model(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.model.save(directory)


class CovertypeDualActorEnvironment:


    def __init__(self, universe, actions, episode_steps=250, seed=0, candidate_slots=10):
        self.universe = universe
        self.actions = actions
        self.episode_steps = int(episode_steps)
        self.heuristic_candidate_limit = int(candidate_slots)
        self.candidate_slots = max(int(candidate_slots), len(actions))
        self.base_env = FixedSetEnvironment(universe, actions, episode_steps=episode_steps, seed=seed)
        self.set_state_dim = self.candidate_slots * self.universe.state_dim
        self.operation_state_dim = self.set_state_dim + self.universe.state_dim
        self.region_state_dim = self.universe.state_dim
        n_constraints = int(self.universe.constraints.shape[1])
        self.statistic_indices = [slot * self.region_state_dim + index
            for slot in range(self.candidate_slots)
            for index in range(self.region_state_dim - n_constraints)]
        self.set_action_dim = self.candidate_slots
        self.operation_action_dim = self.universe.action_dim
        self.step_count = 0
        self.set_state = None
        self.candidate_set_ids = []
        self.candidate_action_ids = []
        self.sets_viewed = set()
        self.exploration_trace_rows = []
        self.visited_set_state_rows = {}

    def reset(self):
        self.step_count = 0
        current = self.base_env.reset()
        self.sets_viewed = set()
        self.exploration_trace_rows = []
        self.visited_set_state_rows = {}
        self.visited_set_state_rows[int(self.base_env.current_set_id)] = current.astype(float).tolist()
        self._refresh_candidates()
        return self.set_state

    def _candidate_groups_for_set(self, set_id):
        from covertype_rl.candidates import _candidate_groups
        return _candidate_groups(self.universe, self.actions, set_id)

    def _region_state(self, set_id):
        return self.universe.state_for_set(set_id)

    def _build_set_state(self, candidate_set_ids):
        pieces = []
        for set_id in candidate_set_ids:
            if int(set_id) >= 0:
                pieces.append(self._region_state(int(set_id)))
            else:
                pieces.append(np.zeros(self.region_state_dim, dtype=np.float32))
        return np.concatenate(pieces).astype(np.float32)

    def _apply_candidates(self, candidate_set_ids, grouped):
        candidate_set_ids = list(candidate_set_ids)
        candidate_action_ids = [list(grouped.get(int(set_id), [])) if int(set_id) >= 0 else [] for set_id in candidate_set_ids]
        while len(candidate_set_ids) < self.candidate_slots:
            candidate_set_ids.append(-1)
            candidate_action_ids.append([])
        if len(candidate_set_ids) > self.candidate_slots:
            raise ValueError('Candidate storage would truncate the exact action space')

        self.candidate_set_ids = candidate_set_ids
        self.candidate_action_ids = candidate_action_ids
        self.set_state = self._build_set_state(self.candidate_set_ids)

    def _refresh_candidates(self):
        current_set_id = int(self.base_env.current_set_id)
        all_candidate_set_ids, grouped, family_by_set = self._candidate_groups_for_set(current_set_id)
        candidate_set_ids = self._select_candidate_set_ids(
            all_candidate_set_ids,
            family_by_set,
            reference_set_size=self.universe.size_for_set(current_set_id),
        )
        self._apply_candidates(candidate_set_ids, grouped)

    def _select_candidate_set_ids(self, all_candidate_set_ids, family_by_set, reference_set_size=None):
        from covertype_rl.candidates import _select_candidate_set_ids
        size = self.base_env.previous_set_size if reference_set_size is None else reference_set_size
        return _select_candidate_set_ids(self.universe, all_candidate_set_ids, family_by_set,
                                         size, self.heuristic_candidate_limit)


    def fix_possible_set_action_probs(self, probs):
        probs = np.asarray(probs, dtype=np.float64)
        mask = np.asarray([1.0 if int(set_id) >= 0 else 0.0 for set_id in self.candidate_set_ids], dtype=np.float64)
        return _masked_probs(probs, mask)

    def get_operation_state(self, set_action):
        set_action = int(set_action)
        if 0 <= set_action < len(self.candidate_set_ids) and int(self.candidate_set_ids[set_action]) >= 0:
            candidate_state = self._region_state(int(self.candidate_set_ids[set_action]))
        else:
            candidate_state = np.zeros(self.region_state_dim, dtype=np.float32)
        return np.concatenate([self.set_state, candidate_state]).astype(np.float32)

    def fix_possible_operation_action_probs(self, set_action, probs):
        probs = np.asarray(probs, dtype=np.float64)
        mask = np.zeros(self.operation_action_dim, dtype=np.float64)
        if 0 <= int(set_action) < len(self.candidate_action_ids):
            actions = self.candidate_action_ids[int(set_action)]
            if actions:
                mask[np.asarray(actions, dtype=np.int64)] = 1.0
        if mask.sum() <= 0.0:
            valid = self.base_env.valid_actions()
            mask[np.asarray(valid, dtype=np.int64)] = 1.0
        return _masked_probs(probs, mask)

    @query_step
    def step(self, set_action, operation_action):
        self.step_count += 1
        set_action = int(set_action)
        operation_action = int(operation_action)
        allowed = self.candidate_action_ids[set_action] if 0 <= set_action < len(self.candidate_action_ids) else []
        if operation_action not in allowed:
            operation_action = int(allowed[0]) if allowed else int(self.base_env.valid_actions()[0])

        previous_set_id = int(self.base_env.current_set_id)
        next_state, metrics, next_set_id, valid = self.base_env.step(operation_action)
        self.sets_viewed.add(int(next_set_id))
        self.visited_set_state_rows[int(next_set_id)] = np.asarray(next_state, dtype=np.float32).astype(float).tolist()

        action = self.actions[operation_action]
        self.exploration_trace_rows.append(
            {
                "step": int(self.step_count),
                "set_id": int(next_set_id),
                "step_extrinsic_reward": float(metrics["extrinsic_reward"]),
                "step_interestingness": float(metrics["interestingness"]),
                "operator": str(action.family),
                "parameter": str(action.label),
                "input_set_id": int(previous_set_id),
                "operation_action": int(operation_action),
                "set_action": int(set_action),
                "set_size": int(metrics["set_size"]),
                "valid": int(valid),
            }
        )

        done = self.step_count >= self.episode_steps
        self._refresh_candidates()
        return self.set_state, float(metrics["extrinsic_reward"]), float(metrics["interestingness"]), done, f"{previous_set_id}-{operation_action}", metrics

    def consume_exploration_logs(self):
        flush_events()
        trace_rows = list(self.exploration_trace_rows)
        state_rows = [
            {"set_id": int(set_id), "state": state}
            for set_id, state in self.visited_set_state_rows.items()
        ]
        return trace_rows, state_rows


@ray.remote
class ParameterServer:
    def __init__(self, set_state_dim, operation_state_dim, set_action_dim, operation_action_dim, args_dict):
        _configure_tf()
        self.args = SimpleNamespace(**args_dict)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        tf.random.set_seed(self.args.seed)
        self.episodes_reserved = int(getattr(self.args, "resume_start_episode", 0) or 0)
        self.episodes_done = int(getattr(self.args, "resume_start_episode", 0) or 0)
        self.global_sets_viewed = set()
        self.cumulative_extrinsic_reward = 0.0
        self.logged_state_ids = set()
        self.is_mira = self.args.baseline in MIRA_BASELINES
        self.mira_rng = np.random.default_rng(int(self.args.seed) + 20260619)
        self.z_sample_count = 0

        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = self.args.output_prefix or f"{self.args.baseline}_seed{self.args.seed}_full_a3c"
        self.run_dir = self.output_dir / self.prefix
        prepared = getattr(self.args, '_prepared_run_dir', None) == str(self.run_dir.resolve())
        self.run_dir.mkdir(parents=True, exist_ok=prepared or not bool(ablation_label(self.args)))
        self.rewards_path = self.run_dir / f"{self.prefix}_{self.args.baseline}_rewards.csv"
        self.trace_path = self.run_dir / f"{self.prefix}_{self.args.baseline}_exploration_trace.csv"
        self.states_path = self.run_dir / f"{self.prefix}_{self.args.baseline}_visited_set_states.csv"
        self.config_path = self.run_dir / f"{self.prefix}_{self.args.baseline}_config.json"
        self.model_dir = self.run_dir / f"{self.prefix}_{self.args.baseline}_models"

        critic_units = _parse_units(self.args.critic_dense_units)
        resume_model_dir = Path(self.args.resume_model_dir) if getattr(self.args, "resume_model_dir", None) else None
        set_model_path = str(resume_model_dir / "set_actor") if resume_model_dir else None
        op_model_path = str(resume_model_dir / "operation_actor") if resume_model_dir else None
        critic_model_path = str(resume_model_dir / "critic") if resume_model_dir else None

        self.global_set_actor = SetActor(
            set_state_dim,
            set_action_dim,
            self.args.lstm_steps,
            self.args.actor_lr,
            self.args.name,
            model_path=set_model_path,
            dense_units=self.args.set_dense_units,
            lstm_units=self.args.set_lstm_units,
        )
        self.global_operation_actor = OperationActor(
            operation_state_dim,
            operation_action_dim,
            self.args.lstm_steps,
            self.args.actor_lr,
            self.args.name,
            model_path=op_model_path,
            dense_units=self.args.op_dense_units,
            lstm_units=self.args.op_lstm_units,
        )
        self.global_critic = Critic(
            set_state_dim,
            self.args.lstm_steps,
            self.args.critic_lr,
            self.args.name,
            model_path=critic_model_path,
            dense_units=critic_units,
            lstm_units=self.args.critic_lstm_units,
        )
        self.reward_replacement = make_reward_replacement(self.args, self.args.base_set_state_dim)
        self.global_mira = None
        if self.is_mira:
            self.global_mira = MIRAModule(
                state_dim=self.args.base_set_state_dim,
                set_action_dim=set_action_dim,
                operation_action_dim=operation_action_dim,
                latent_dim=self.args.mira_latent_dim,
                lr=self.args.mira_lr,
                dense_units=self.args.mira_dense_units,
                target_tau=self.args.mira_target_tau,
                **representation_options(self.args),
            )
        self.mira_learner = None
        if self.global_mira is not None:
            from revision_support.mira_learner import RepresentationLearner
            self.mira_learner = RepresentationLearner(self.global_mira, capacity=self.args.mira_replay_buffer_size,
                batch_size=self.args.mira_phi_batch_size, min_size=self.args.mira_min_replay_size,
                seed=self.args.seed, gamma=self.args.gamma, phi_weight=self.args.mira_phi_weight, dyn_weight=self.args.mira_dyn_weight)

        self._write_config(set_state_dim, operation_state_dim, set_action_dim, operation_action_dim)
        self._init_csv_files()

    def _write_config(self, set_state_dim, operation_state_dim, set_action_dim, operation_action_dim):
        config = {
            **vars(self.args),
            "runner": "full_a3c_ray_dual_actor",
            "set_state_dim": int(set_state_dim),
            "operation_state_dim": int(operation_state_dim),
            "set_action_dim": int(set_action_dim),
            "operation_action_dim": int(operation_action_dim),
            "base_set_state_dim": int(getattr(self.args, "base_set_state_dim", set_state_dim)),
            "base_operation_state_dim": int(getattr(self.args, "base_operation_state_dim", operation_state_dim)),
            "note": "Covertype fixed-set universe with Galaxy-style set actor, operation actor, critic, optional MIRA encoder/dynamics, and Ray parameter server.",
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def _init_csv_files(self):
        with open(self.rewards_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "extrinsic_reward",
                    "mean_step_extrinsic",
                    "target_hits",
                    *([] if self.is_mira else ["interestingness"]),
                    "familiarity",
                    "counter_curiosity",
                    "coherency",
                    "diversity",
                    "total_reward",
                    "sets_viewed",
                    "cumulative_unique_sets_viewed",
                    "target_efficiency",
                    "cumulative_extrinsic_reward",
                    "cumulative_target_efficiency",
                    "valid_steps",
                    "mira_bonus",
                    "exploration_reward",
                    "mira_phi_loss",
                    "mira_dyn_loss",
                    "mira_prediction_error",
                    "mira_mean_d_phi",
                    "mira_mean_metric_target",
                    "mira_z_source",
                    "mira_pair_source",
                    "mira_replay_size",
                ]
            )
        with open(self.trace_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "agent_id",
                    "step",
                    "set_id",
                    "step_extrinsic_reward",
                    *([] if self.is_mira else ["step_interestingness"]),
                    "operator",
                    "parameter",
                    "input_set_id",
                    "set_action",
                    "operation_action",
                    "step_coherency",
                    "step_diversity",
                    "step_total_reward",
                    "set_size",
                    "valid",
                    "step_mira_bonus",
                    "mira_z_source",
                ]
            )

    def next_episode(self):
        if self.episodes_reserved >= int(self.args.episodes):
            return None
        self.episodes_reserved += 1
        return self.episodes_reserved

    def get_weights(self):
        return {
            "set": self.global_set_actor.model.get_weights(),
            "op": self.global_operation_actor.model.get_weights(),
            "critic": self.global_critic.model.get_weights(),
            "mira": self.mira_learner.weights() if self.mira_learner is not None else None,
        }

    def apply_gradients_and_get_weights(self, set_grads, op_grads, critic_grads, mira_encoder_grads=None, mira_dynamics_grads=None):
        if set_grads:
            self.global_set_actor.apply_gradients(set_grads)
        if op_grads:
            self.global_operation_actor.apply_gradients(op_grads)
        if critic_grads:
            self.global_critic.apply_gradients(critic_grads)
        if self.global_mira is not None:
            self.mira_learner.check()
        return self.get_weights()


    def sample_mira_direction(self):
        from revision_support.direction_sampling import unassisted_direction
        return unassisted_direction(self, 'covertype')


    def update_and_sample_mira_batch(self, states, set_actions, operation_actions, task_rewards, next_states, episode, metadata):

        actions = np.column_stack([np.asarray(set_actions).reshape(-1), np.asarray(operation_actions).reshape(-1)])
        from revision_support.mira_task import replay_task_reward
        return self.mira_learner.submit(states, actions, replay_task_reward(task_rewards, self.args), next_states,
            valid_action_masks=np.asarray([m[0] for m in metadata], dtype=bool),
            state_keys=[m[1] for m in metadata], next_terminals=[m[2] for m in metadata],
            next_state_keys=[m[3] for m in metadata])

    def save_mira_snapshot(self, directory):
        self.mira_learner.check()
        with self.mira_learner.lock:
            self.global_mira.save_models(directory)


    def record_pca_collection(self, records):
        from revision_support.pca_training import record_collection
        return record_collection(self, records, 'covertype')

    def record_episode(self, episode, totals, episode_set_ids, trace_rows, state_rows):
        self.episodes_done += 1
        completed_episode = int(self.episodes_done)
        episode_sets = {int(set_id) for set_id in episode_set_ids if int(set_id) >= 0}
        self.global_sets_viewed.update(episode_sets)
        self.cumulative_extrinsic_reward += float(totals["extrinsic_reward"])
        cumulative_unique = len(self.global_sets_viewed)
        sets_viewed = len(episode_sets)
        cumulative_efficiency = self.cumulative_extrinsic_reward / max(cumulative_unique, 1)

        with open(self.rewards_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    completed_episode,
                    float(totals["extrinsic_reward"]),
                    float(totals["extrinsic_reward"]) / max(int(self.args.steps), 1),
                    int(totals["target_hits"]),
                    *([] if self.is_mira else [float(totals["interestingness"])]),
                    float(totals["familiarity"]),
                    float(totals["counter_curiosity"]),
                    float(totals["coherency"]),
                    float(totals["diversity"]),
                    float(totals["total_reward"]),
                    int(sets_viewed),
                    int(cumulative_unique),
                    float(totals["extrinsic_reward"]) / max(sets_viewed, 1),
                    float(self.cumulative_extrinsic_reward),
                    float(cumulative_efficiency),
                    int(totals["valid_steps"]),
                    float(totals.get("mira_bonus", 0.0)),
                    float(totals.get("exploration_reward", 0.0)),
                    float(totals.get("mira_phi_loss", 0.0)),
                    float(totals.get("mira_dyn_loss", 0.0)),
                    float(totals.get("mira_prediction_error", 0.0)),
                    float(totals.get("mira_mean_d_phi", 0.0)),
                    float(totals.get("mira_mean_metric_target", 0.0)),
                    totals.get("mira_z_source", ""),
                    totals.get("mira_pair_source", ""),
                    int(totals.get("mira_replay_size", 0)),
                ]
            )

        with open(self.trace_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in trace_rows:
                writer.writerow(
                    [
                        completed_episode,
                        int(row.get("agent_id", -1)),
                        int(row.get("step", -1)),
                        int(row.get("set_id", -1)),
                        float(row.get("step_extrinsic_reward", 0.0)),
                        *([] if self.is_mira else [float(row.get("step_interestingness", 0.0))]),
                        row.get("operator", ""),
                        row.get("parameter", ""),
                        int(row.get("input_set_id", -1)),
                        int(row.get("set_action", -1)),
                        int(row.get("operation_action", -1)),
                        float(row.get("step_coherency", 0.0)),
                        float(row.get("step_diversity", 0.0)),
                        float(row.get("step_total_reward", 0.0)),
                        int(row.get("set_size", 0)),
                        int(row.get("valid", 0)),
                        float(row.get("step_mira_bonus", 0.0)),
                        row.get("mira_z_source", ""),
                    ]
                )

        new_state_rows = []
        for row in state_rows:
            set_id = int(row.get("set_id", -1))
            if set_id < 0 or set_id in self.logged_state_ids:
                continue
            state = row.get("state", [])
            if not state:
                continue
            self.logged_state_ids.add(set_id)
            new_state_rows.append((set_id, state))
        if new_state_rows:
            file_exists = self.states_path.exists()
            with open(self.states_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    state_dim = len(new_state_rows[0][1])
                    writer.writerow(["set_id"] + [f"state_{idx}" for idx in range(state_dim)])
                for set_id, state in new_state_rows:
                    writer.writerow([set_id] + list(state))

        if completed_episode % int(self.args.save_interval) == 0:
            self.save_models(f"episode_{completed_episode}")
        return {
            "episode": completed_episode,
            "reserved_episode": int(episode),
            "episodes_done": completed_episode,
            "sets_viewed": int(sets_viewed),
            "cumulative_unique_sets_viewed": int(cumulative_unique),
            "cumulative_target_efficiency": float(cumulative_efficiency),
        }

    def replacement_reward(self, next_state):
        embed = self.mira_learner.embed if self.reward_replacement.kind == 'knn' else None
        return float(self.args.replacement_reward_scale) * float(
            self.reward_replacement.observe([next_state], embed=embed)[0])

    def record_mira_reward_state(self, worker_id, state):
        self.mira_learner.record_reward_state(worker_id, state)

    def save_models(self, label="final"):
        if getattr(self.args, 'output_profile', 'full') == 'csv':
            return None
        target = self.model_dir / str(label)
        self.global_set_actor.save_model(target / "set_actor")
        self.global_operation_actor.save_model(target / "operation_actor")
        self.global_critic.save_model(target / "critic")
        if self.global_mira is not None:
            self.save_mira_snapshot(target / "mira")
        if self.reward_replacement is not None:
            self.reward_replacement.save(target / "mira")
        return str(target)

    def save_final_models(self):
        if self.mira_learner is not None:
            self.mira_learner.close()
            self.mira_learner.save_report(self.run_dir)
        return self.save_models("final")

    def output_paths(self):
        return {
            "rewards": str(self.rewards_path),
            "trace": str(self.trace_path),
            "states": str(self.states_path),
            "config": str(self.config_path),
            "models": str(self.model_dir),
        }


@ray.remote
class WorkerAgent:
    def __init__(self, universe_dir, action_labels, ps_handle, agent_id, args_dict):
        _configure_tf()
        self.agent_id = int(agent_id)
        self.ps = ps_handle
        self.args = SimpleNamespace(**args_dict)
        worker_seed = int(self.args.seed) * 1000 + self.agent_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        tf.random.set_seed(worker_seed)
        self.universe = load_fixed_universe(universe_dir)
        self.actions = _actions_from_labels(action_labels)
        self.env = CovertypeDualActorEnvironment(
            self.universe,
            self.actions,
            episode_steps=self.args.steps,
            seed=int(self.args.seed) * 1000 + self.agent_id,
            candidate_slots=self.args.candidate_slots,
        )
        self.is_mira = self.args.baseline in MIRA_BASELINES
        self.base_set_state_dim = self.env.set_state_dim
        self.base_operation_state_dim = self.env.operation_state_dim
        self.set_state_dim = self.base_set_state_dim + (int(self.args.mira_latent_dim) if self.is_mira else 0)
        self.operation_state_dim = self.base_operation_state_dim + (int(self.args.mira_latent_dim) if self.is_mira else 0)
        self.set_action_dim = self.env.set_action_dim
        self.operation_action_dim = self.env.operation_action_dim
        self.steps = int(self.args.lstm_steps)
        critic_units = _parse_units(self.args.critic_dense_units)
        self.mira = None
        if self.is_mira:
            self.mira = MIRAModule(
                state_dim=self.base_set_state_dim,
                set_action_dim=self.set_action_dim,
                operation_action_dim=self.operation_action_dim,
                latent_dim=self.args.mira_latent_dim,
                lr=self.args.mira_lr,
                dense_units=self.args.mira_dense_units,
                target_tau=self.args.mira_target_tau,
                **representation_options(self.args),
            )
        self.set_actor = SetActor(
            self.set_state_dim,
            self.set_action_dim,
            self.steps,
            self.args.actor_lr,
            self.args.name,
            dense_units=self.args.set_dense_units,
            lstm_units=self.args.set_lstm_units,
        )
        self.operation_actor = OperationActor(
            self.operation_state_dim,
            self.operation_action_dim,
            self.steps,
            self.args.actor_lr,
            self.args.name,
            dense_units=self.args.op_dense_units,
            lstm_units=self.args.op_lstm_units,
        )
        self.critic = Critic(
            self.set_state_dim,
            self.steps,
            self.args.critic_lr,
            self.args.name,
            dense_units=critic_units,
            lstm_units=self.args.critic_lstm_units,
        )
        self.set_op_counters = {}
        self.sync_with_ps()

    def sync_with_ps(self):
        weights = ray.get(self.ps.get_weights.remote())
        self.set_actor.model.set_weights(weights["set"])
        self.operation_actor.model.set_weights(weights["op"])
        self.critic.model.set_weights(weights["critic"])
        if self.mira is not None and weights.get("mira") is not None:
            self.mira.set_weights(weights["mira"])

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards, dtype=np.float64)
        cumulative = 0.0 if done else float(np.squeeze(next_v_value))
        for idx in reversed(range(len(rewards))):
            cumulative = float(self.args.gamma) * cumulative + np.asarray(rewards[idx]).item()
            td_targets[idx] = cumulative
        return td_targets

    def list_to_batch(self, items):
        return np.concatenate(items, axis=0)

    def augment_set_state(self, set_state, z):
        if not self.is_mira:
            return np.asarray(set_state, dtype=np.float32)
        return np.concatenate(
            [
                np.asarray(set_state, dtype=np.float32).reshape(-1),
                np.asarray(z, dtype=np.float32).reshape(-1),
            ]
        ).astype(np.float32)

    def augment_operation_state(self, operation_state, z):
        if not self.is_mira:
            return np.asarray(operation_state, dtype=np.float32)
        return np.concatenate(
            [
                np.asarray(operation_state, dtype=np.float32).reshape(-1),
                np.asarray(z, dtype=np.float32).reshape(-1),
            ]
        ).astype(np.float32)

    def sample_mira_direction(self):
        sample = ray.get(self.ps.sample_mira_direction.remote())
        return (
            np.asarray(sample["z"], dtype=np.float32),
            str(sample.get("z_source", "random")),
        )


    def sample_action(self, probs, action_dim, fallback_action=0):
        probs = np.asarray(probs, dtype=np.float64)
        valid_actions = np.flatnonzero(np.isfinite(probs) & (probs > 0.0))
        if valid_actions.size == 0:
            return int(fallback_action)
        safe_probs = np.zeros(int(action_dim), dtype=np.float64)
        copy_len = min(int(action_dim), probs.size)
        safe_probs[:copy_len] = np.nan_to_num(probs[:copy_len], nan=0.0, posinf=0.0, neginf=0.0)
        safe_probs[safe_probs < 0.0] = 0.0
        total = float(safe_probs.sum())
        if total <= 0.0:
            return int(fallback_action)
        safe_probs /= total
        return int(np.random.choice(int(action_dim), p=safe_probs))

    def embed_mira_state(self, state):
        if self.mira is None:
            return np.asarray(state, dtype=np.float32).reshape(-1)
        phi = self.mira.embed(np.asarray(state, dtype=np.float32).reshape((1, self.base_set_state_dim)))
        return np.asarray(phi[0], dtype=np.float32).reshape(-1)


    def train_loop(self):
        while True:
            episode = ray.get(self.ps.next_episode.remote())
            if episode is None:
                break
            try:
                self._run_episode(int(episode))
            except Exception as error:
                raise RuntimeError(f"Worker {self.agent_id} episode {episode} failed") from error
        return True

    def _run_episode(self, episode):
        self.sync_with_ps()
        if self.mira is not None:
            self.mira.begin_episode()

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
        mira_metadata_batch = []
        episode_set_op_counters = {}
        episode_history_phi = []
        if self.is_mira:
            (
                mira_z,
                mira_z_source,
            ) = self.sample_mira_direction()
        else:
            mira_z = None
            mira_z_source = "none"

        totals = _zero_totals()
        totals["mira_z_source"] = mira_z_source
        set_action_steps = [np.zeros(self.set_state_dim, dtype=np.float32) for _ in range(self.steps)]
        operation_action_steps = [np.zeros(self.operation_state_dim, dtype=np.float32) for _ in range(self.steps)]

        set_state = self.env.reset()
        set_state = self.env.set_state
        set_action_steps.pop(0)
        set_action_steps.append(self.augment_set_state(set_state, mira_z))
        if self.is_mira:
            episode_history_phi.append(self.embed_mira_state(set_state))
        done = False

        while not done:
            set_probs = fast_predict(self.set_actor.model,
                np.asarray(set_action_steps, dtype=np.float32).reshape((1, self.steps, self.set_state_dim)),
            )[0]
            set_probs = self.env.fix_possible_set_action_probs(set_probs)
            set_action = self.sample_action(
                set_probs,
                self.set_action_dim,
                fallback_action=0,
            )

            operation_state = self.augment_operation_state(self.env.get_operation_state(set_action), mira_z)
            operation_action_steps.pop(0)
            operation_action_steps.append(operation_state)
            op_probs = fast_predict(self.operation_actor.model,
                np.asarray(operation_action_steps, dtype=np.float32).reshape((1, self.steps, self.operation_state_dim)),
            )[0]
            op_probs = self.env.fix_possible_operation_action_probs(set_action, op_probs)
            operation_action = self.sample_action(
                op_probs,
                self.operation_action_dim,
                fallback_action=0,
            )

            if self.mira is not None:
                from revision_support.transition_metadata import dual_reference_metadata
                reference_mask, state_key = dual_reference_metadata(self.env)
            next_set_state, r_ext, r_int, done, set_op_pair, metrics = self.env.step(set_action, operation_action)
            if self.mira is not None:
                _, next_state_key = dual_reference_metadata(self.env)
                mira_metadata_batch.append((reference_mask, state_key, done, next_state_key))
            next_set_state = self.env.set_state
            r_ext = float(r_ext)
            r_int = float(r_int)
            phi_s_t = self.embed_mira_state(set_state) if self.is_mira else np.asarray(set_state, dtype=np.float32).flatten()
            phi_s_next = self.embed_mira_state(next_set_state) if self.is_mira else np.asarray(next_set_state, dtype=np.float32).flatten()
            r_coh = _coherency(phi_s_t, phi_s_next)
            r_div = _episode_diversity(phi_s_next, episode_history_phi, eta=self.args.diversity_eta)
            mira_bonus = 0.0
            if self.mira is not None:
                mira_bonus = select_bonus(
                    getattr(self.args, 'exploration_reward', 'directional'),
                    lambda: self.mira.compute_bonus(set_state, next_set_state, mira_z,
                                                   clip_value=self.args.mira_bonus_clip),
                    lambda: float(ray.get(self.ps.replacement_reward.remote(next_set_state))))
                r_div = mira_bonus
            episode_history_phi.append(phi_s_next)

            counter_curiosity = _counter_curiosity(
                set_op_pair,
                episode_set_op_counters,
                self.set_op_counters,
                episode_steps=self.args.steps,
            )
            exploration_reward = 0.0
            if self.is_mira:
                exploration_reward = float(self.args.w_mira) * float(mira_bonus)
            step_reward = _training_reward(self.args.baseline, r_ext, r_int,
                                            counter_curiosity, r_coh, r_div, mira_bonus, self.args)
            next_set_action_steps = list(set_action_steps)
            next_set_action_steps.pop(0)
            next_set_action_steps.append(self.augment_set_state(next_set_state, mira_z))

            reward_batch.append(np.reshape(step_reward, (1, 1)))
            set_state_batch.append(np.asarray(set_action_steps, dtype=np.float32).reshape((1, self.steps, self.set_state_dim)))
            set_action_batch.append(np.reshape(set_action, (1, 1)))
            operation_state_batch.append(np.asarray(operation_action_steps, dtype=np.float32).reshape((1, self.steps, self.operation_state_dim)))
            operation_action_batch.append(np.reshape(operation_action, (1, 1)))
            if self.is_mira:
                mira_state_batch.append(np.asarray(set_state, dtype=np.float32).reshape((1, self.base_set_state_dim)))
                mira_next_state_batch.append(np.asarray(next_set_state, dtype=np.float32).reshape((1, self.base_set_state_dim)))
                mira_set_action_batch.append(np.reshape(set_action, (1, 1)))
                mira_operation_action_batch.append(np.reshape(operation_action, (1, 1)))
                metric_reward = 0.0 if self.args.baseline == "mira_no_ext" else r_ext
                mira_task_reward_batch.append(np.reshape(metric_reward, (1, 1)))

            _add_totals(
                totals,
                metrics,
                r_ext,
                r_int,
                counter_curiosity,
                r_coh,
                r_div,
                step_reward,
                mira_bonus,
                exploration_reward,
            )
            if self.env.exploration_trace_rows:
                self.env.exploration_trace_rows[-1]["agent_id"] = int(self.agent_id)
                self.env.exploration_trace_rows[-1]["step_coherency"] = float(r_coh)
                self.env.exploration_trace_rows[-1]["step_diversity"] = float(r_div)
                self.env.exploration_trace_rows[-1]["step_total_reward"] = float(step_reward)
                self.env.exploration_trace_rows[-1]["step_mira_bonus"] = float(mira_bonus)
                self.env.exploration_trace_rows[-1]["mira_z_source"] = mira_z_source

            if len(set_state_batch) >= int(self.args.update_interval) or done:
                set_states = self.list_to_batch(set_state_batch)
                set_actions = self.list_to_batch(set_action_batch)
                operation_states = self.list_to_batch(operation_state_batch)
                operation_actions = self.list_to_batch(operation_action_batch)
                rewards = self.list_to_batch(reward_batch)
                next_v_value = fast_predict(self.critic.model,
                    np.asarray(next_set_action_steps, dtype=np.float32).reshape((1, self.steps, self.set_state_dim)),
                )
                td_targets = self.n_step_td_target(rewards, next_v_value, done)
                advantages = td_targets - fast_predict(self.critic.model, set_states)

                set_grads, _ = self.set_actor.get_gradients(set_states, set_actions, advantages)
                op_grads, _ = self.operation_actor.get_gradients(operation_states, operation_actions, advantages)
                critic_grads, _ = self.critic.get_gradients(set_states, td_targets)
                mira_encoder_grads = None
                mira_dynamics_grads = None
                if self.mira is not None and mira_state_batch:
                    mira_stats = ray.get(self.ps.update_and_sample_mira_batch.remote(
                        self.list_to_batch(mira_state_batch), self.list_to_batch(mira_set_action_batch),
                        self.list_to_batch(mira_operation_action_batch), self.list_to_batch(mira_task_reward_batch),
                        self.list_to_batch(mira_next_state_batch), episode, mira_metadata_batch))
                    totals['mira_pair_source'] = 'async_replay'
                    totals['mira_replay_size'] = mira_stats['replay_size']
                    for key in ('phi_loss','dyn_loss','prediction_error','mean_d_phi','mean_metric_target'):
                        totals['mira_'+key] += mira_stats[key]
                    totals['mira_update_count'] += 1
                new_weights = ray.get(
                    self.ps.apply_gradients_and_get_weights.remote(
                        [grad.numpy() if grad is not None else None for grad in set_grads],
                        [grad.numpy() if grad is not None else None for grad in op_grads],
                        [grad.numpy() if grad is not None else None for grad in critic_grads],
                        mira_encoder_grads,
                        mira_dynamics_grads,
                    )
                )
                self.set_actor.model.set_weights(new_weights["set"])
                self.operation_actor.model.set_weights(new_weights["op"])
                self.critic.model.set_weights(new_weights["critic"])
                if self.mira is not None and new_weights.get("mira") is not None:
                    self.mira.set_weights(new_weights["mira"])
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
                mira_metadata_batch = []

            set_state = next_set_state
            set_action_steps = next_set_action_steps

        if self.mira is not None:
            ray.get(self.ps.record_mira_reward_state.remote(
                self.agent_id, self.mira.reward_moments.state()))
        for pair, count in episode_set_op_counters.items():
            self.set_op_counters[pair] = self.set_op_counters.get(pair, 0) + count

        if totals.get("mira_update_count", 0) > 0:
            update_count = float(totals["mira_update_count"])
            for key in (
                "mira_phi_loss",
                "mira_dyn_loss",
                "mira_prediction_error",
                "mira_mean_d_phi",
                "mira_mean_metric_target",
            ):
                totals[key] = float(totals[key]) / update_count


        trace_rows, state_rows = self.env.consume_exploration_logs()
        stats = ray.get(
            self.ps.record_episode.remote(
                episode,
                totals,
                list(getattr(self.env, "sets_viewed", [])),
                trace_rows,
                state_rows,
            )
        )
        print(
            f"EP{stats['episode']} {self.args.baseline} Agent{self.agent_id} | "
            f"Ext_R: {totals['extrinsic_reward']:.2f} | Total: {totals['total_reward']:.2f} | "
            f"Sets: {stats['sets_viewed']} | Done: {stats['episodes_done']}",
            flush=True,
        )


def build_parser():
    from revision_support.covertype_defaults import CovertypeArgumentParser
    parser = CovertypeArgumentParser(description='')
    parser.add_argument("--baseline", choices=sorted(BASELINES), default="mira")
    parser.add_argument("--csv_path", default="covertype.csv")
    parser.add_argument("--target_set", default=None)
    parser.add_argument("--target_seed", type=int, default=None)
    parser.add_argument("--target_size", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--seed", type=int, default=1)
    add_ablation_arguments(parser, backend='covertype')
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--n_sets", type=int, default=100000)
    parser.add_argument("--min_set_size", type=int, default=10)
    parser.add_argument("--max_set_size", type=int, default=200000)
    parser.add_argument("--preprocess_name", default=None)
    parser.add_argument("--force_preprocess", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--update_interval", type=int, default=20)
    parser.add_argument("--actor_lr", type=float, default=0.00003)
    parser.add_argument("--critic_lr", type=float, default=0.00003)
    parser.add_argument("--lstm_steps", type=int, default=5)
    parser.add_argument("--set_dense_units", type=int, default=512)
    parser.add_argument("--set_lstm_units", type=int, default=512)
    parser.add_argument("--op_dense_units", type=int, default=512)
    parser.add_argument("--op_lstm_units", type=int, default=256)
    parser.add_argument("--critic_dense_units", default="512,256,128")
    parser.add_argument("--critic_lstm_units", type=int, default=128)
    parser.add_argument("--candidate_slots", type=int, default=10)
    parser.add_argument("--counter_curiosity_ratio", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--w_ext", type=float, default=None)
    parser.add_argument("--w_int", type=float, default=None)
    parser.add_argument("--w_mira", type=float, default=1.0)
    parser.add_argument("--w_coh", type=float, default=0.1)
    parser.add_argument("--w_div", type=float, default=0.1)
    parser.add_argument("--diversity_eta", type=float, default=0.1)
    parser.add_argument("--mira_latent_dim", type=int, default=16)
    parser.add_argument("--mira_lr", type=float, default=0.0003)
    parser.add_argument("--mira_dense_units", type=int, default=256)
    parser.add_argument("--mira_lambda_ana", type=float, default=1.0)
    parser.add_argument("--mira_mc_samples", type=int, default=4)
    parser.add_argument("--mira_bonus_clip", type=float, default=1.0)
    parser.add_argument("--mira_target_tau", type=float, default=0.01)
    parser.add_argument("--mira_phi_weight", type=float, default=1.0)
    parser.add_argument("--mira_dyn_weight", type=float, default=1.0)
    parser.add_argument("--mira_pair_warmup_episodes", type=int, default=0, choices=(0,))
    parser.add_argument("--mira_replay_buffer_size", type=int, default=50000)
    parser.add_argument("--mira_phi_batch_size", type=int, default=128)
    parser.add_argument("--mira_min_replay_size", type=int, default=256)
    parser.add_argument("--mira_orthogonal_prob", type=float, default=0.0, choices=(0.0,))
    parser.add_argument("--name", default="")
    parser.add_argument("--output_prefix", default=None)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--resume_model_dir", default=None)
    parser.add_argument("--resume_start_episode", type=int, default=0)
    parser.add_argument("--ray_address", default=None)
    parser.add_argument("--ray_temp_dir", default=None)
    return parser


@shared_query_run
def run(args):
    if args.baseline in MIRA_BASELINES:
        args.mira_core_version = "mira_reference_wasserstein_v3"
        if args.resume_model_dir or args.resume_start_episode:
            raise ValueError("MIRA v3 requires a fresh run; checkpoint continuation is not supported")
    validate_ablation_args(args)
    if args.baseline == "mira_no_ext" and args.resume_model_dir:
        raise ValueError('MIRA w/o Ext requires fresh weights; a checkpoint may contain task reward information')
    root_dir = Path(__file__).resolve().parents[1]
    if ablation_label(args):
        base = args.output_prefix or f"{args.baseline}_seed{args.seed}_full_a3c"
        args.output_prefix = base + "_" + ablation_label(args)
        args.name = args.output_prefix
        ablation_output_dir = Path(args.output_dir)
        if not ablation_output_dir.is_absolute():
            ablation_output_dir = root_dir / ablation_output_dir
        if (ablation_output_dir / args.output_prefix).exists():
            raise ValueError('Ablation output already exists; choose a fresh --output_prefix')
    if args.baseline not in BASELINES:
        raise ValueError(f"Missing or unsupported full A3C baseline: {args.baseline}. Use one of {sorted(BASELINES)}.")

    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = root_dir / csv_path

    if not args.name:
        args.name = args.output_prefix or f"{args.baseline}_seed{args.seed}_full_a3c"

    data = load_covertype(csv_path, n_bins=args.n_bins)
    from revision_support.query_clients import register_covertype_backend
    register_covertype_backend(data)
    target_items, target_path = resolve_target_set(
        data,
        root_dir=root_dir,
        target_set=args.target_set,
        target_seed=args.target_seed,
        target_size=args.target_size,
    )
    actions = build_action_space(n_continuous=data.continuous.shape[1], n_bins=data.n_bins)
    universe, universe_dir = ensure_fixed_universe(
        data=data,
        actions=actions,
        target_items=target_items,
        target_path=target_path,
        root_dir=root_dir,
        n_sets=args.n_sets,
        seed=args.seed,
        min_set_size=args.min_set_size,
        max_set_size=args.max_set_size,
        preprocess_name=args.preprocess_name,
        force=args.force_preprocess,
        csv_path=csv_path,
    )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    set_query_report_path(output_dir / (args.output_prefix or args.name) / "query_scheduler.json")
    if ablation_label(args):


        prepared = output_dir / (args.output_prefix or args.name)
        prepared.mkdir(parents=True, exist_ok=False)
        args._prepared_run_dir = str(prepared.resolve())
    args.csv_path = str(csv_path)
    args.target_path = str(target_path)
    args.fixed_universe_dir = str(universe_dir)
    args.fixed_universe_sets = int(universe.n_sets)
    args.target_items = int(len(target_items))

    dummy_env = CovertypeDualActorEnvironment(
        universe,
        actions,
        episode_steps=args.steps,
        seed=args.seed,
        candidate_slots=args.candidate_slots,
    )
    dummy_env.reset()
    args.candidate_capacity = int(dummy_env.candidate_slots)
    args.heuristic_candidate_limit = int(dummy_env.heuristic_candidate_limit)
    args.mira_statistic_indices = getattr(dummy_env, 'statistic_indices', [])
    args.base_set_state_dim = int(dummy_env.set_state_dim)
    args.base_operation_state_dim = int(dummy_env.operation_state_dim)
    set_state_dim = int(dummy_env.set_state_dim)
    operation_state_dim = int(dummy_env.operation_state_dim)
    if args.baseline in MIRA_BASELINES:
        set_state_dim += int(args.mira_latent_dim)
        operation_state_dim += int(args.mira_latent_dim)

    if not ray.is_initialized():
        ray_kwargs = {
            "address": args.ray_address,
            "ignore_reinit_error": True,
            "include_dashboard": False,
            "_system_config": {
                "health_check_failure_threshold": 20,
                "health_check_period_ms": 1000,
            },
        }
        if args.ray_temp_dir:
            ray_kwargs["_temp_dir"] = args.ray_temp_dir
        ray.init(**ray_kwargs)

    from revision_support.pca_training import prepare_in_budget
    pca_records = prepare_in_budget(dummy_env, args, 'covertype',
        output_dir / (args.output_prefix or args.name))
    args_dict = vars(args).copy()
    ps = query_actor(ParameterServer).remote(
        set_state_dim,
        operation_state_dim,
        dummy_env.set_action_dim,
        dummy_env.operation_action_dim,
        args_dict,
    )
    if pca_records:
        ray.get(ps.record_pca_collection.remote(pca_records))
        pca_records = []
    action_labels = [action.label for action in actions]
    workers = [
        query_actor(WorkerAgent).remote(str(universe_dir), action_labels, ps, worker_id, args_dict)
        for worker_id in range(max(1, int(args.workers)))
    ]
    complete_training(ray, ps, workers, [worker.train_loop.remote() for worker in workers])
    return ray.get(ps.output_paths.remote())


def main():
    paths = run(build_parser().parse_args())
    print("Wrote:")
    for key, value in paths.items():
        print(f"  {key}: {value}")


def _configure_tf():
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except RuntimeError:
                pass


def _masked_probs(probs, mask):
    probs = np.asarray(probs, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    masked = np.where(mask > 0.0, probs, 0.0)
    total = float(masked.sum())
    if total <= 0.0 or not np.isfinite(total):
        valid = np.flatnonzero(mask > 0.0)
        fallback = np.zeros_like(masked)
        if valid.size == 0:
            fallback[:] = 1.0 / float(fallback.size)
        else:
            fallback[valid] = 1.0 / float(valid.size)
        return fallback.astype(np.float64)
    return (masked / total).astype(np.float64)


def _actions_from_labels(labels):
    from .actions import Action

    actions = []
    for label in labels:
        parts = str(label).split(":")
        op = parts[0]
        if op in {"by_facet_cont", "by_superset_cont", "by_neighbors_cont"}:
            feature = int(parts[1]) if len(parts) > 1 else -1
            value = int(parts[2]) if len(parts) > 2 else -1
            delta = int(parts[3]) if len(parts) > 3 else 0
            actions.append(Action(op, feature=feature, value=value, delta=delta))
        elif op in {"by_facet_cover", "by_facet_wilderness", "by_facet_soil"}:
            value = int(parts[1]) if len(parts) > 1 else -1
            actions.append(Action(op, value=value))
        elif op == "by_distribution":
            delta = int(parts[1]) if len(parts) > 1 else 0
            actions.append(Action(op, delta=delta))
        else:
            actions.append(Action(op))
    return actions


def _parse_units(value):
    if isinstance(value, (list, tuple)):
        units = [int(item) for item in value]
    else:
        units = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(units) != 3:
        raise ValueError("--critic_dense_units must contain exactly three comma-separated integers, e.g. 512,256,128")
    return tuple(units)


def _counter_curiosity(pair, episode_counts, global_counts, episode_steps):
    episode_counts[pair] = episode_counts.get(pair, 0) + 1
    count = episode_counts[pair] + global_counts.get(pair, 0)
    return (100.0 / float(max(episode_steps, 1))) / float(max(count, 1))


def _coherency(phi_current, phi_next):
    if np.allclose(phi_current, phi_next, atol=1e-5):
        return 0.0
    denom = float(np.linalg.norm(phi_current) * np.linalg.norm(phi_next))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(phi_current, phi_next) / denom)


def _episode_diversity(phi_next, history, eta=0.1):
    if not history:
        return 1.0
    distances = [np.linalg.norm(phi_next - item) for item in history]
    min_distance = float(np.min(distances)) if distances else 0.0
    return float(1.0 - np.exp(-float(eta) * min_distance))


def _training_reward(
    baseline,
    r_ext,
    r_int,
    counter_curiosity,
    r_coh,
    r_div,
    mira_bonus,
    args,
):
    if baseline == "pure_a3c":
        return float(args.w_ext) * float(r_ext)
    if baseline == "dora":
        ratio = float(args.counter_curiosity_ratio)
        return (1.0 - ratio) * float(r_ext) + ratio * float(counter_curiosity)
    if baseline == "atena_no_ext":
        return float(args.w_int) * float(r_int) + float(args.w_coh) * float(r_coh) + float(args.w_div) * float(r_div)
    if baseline == "atena":
        return (
            float(args.w_ext) * float(r_ext)
            + float(args.w_int) * float(r_int)
            + float(args.w_coh) * float(r_coh)
            + float(args.w_div) * float(r_div)
        )
    if baseline == "mira":
        return float(args.w_ext) * float(r_ext) + float(args.alpha) * float(args.w_mira) * float(mira_bonus)
    if baseline == "mira_no_ext":
        return float(args.alpha) * float(args.w_mira) * float(mira_bonus)
    raise ValueError(f"Unsupported baseline: {baseline}")


def _zero_totals():
    return {
        "extrinsic_reward": 0.0,
        "target_hits": 0,
        "interestingness": 0.0,
        "familiarity": 0.0,
        "counter_curiosity": 0.0,
        "coherency": 0.0,
        "diversity": 0.0,
        "total_reward": 0.0,
        "valid_steps": 0,
        "mira_bonus": 0.0,
        "exploration_reward": 0.0,
        "mira_phi_loss": 0.0,
        "mira_dyn_loss": 0.0,
        "mira_prediction_error": 0.0,
        "mira_mean_d_phi": 0.0,
        "mira_mean_metric_target": 0.0,
        "mira_update_count": 0,
        "mira_z_source": "",
        "mira_pair_source": "",
        "mira_replay_size": 0,
    }


def _add_totals(
    totals,
    metrics,
    r_ext,
    r_int,
    counter_curiosity,
    r_coh,
    r_div,
    step_reward,
    mira_bonus=0.0,
    exploration_reward=0.0,
):
    totals["extrinsic_reward"] += float(r_ext)
    totals["target_hits"] += int(metrics.get("target_hits", 0))
    totals["interestingness"] += float(r_int)
    totals["familiarity"] += float(r_ext)
    totals["counter_curiosity"] += float(counter_curiosity)
    totals["coherency"] += float(r_coh)
    totals["diversity"] += float(r_div)
    totals["total_reward"] += float(step_reward)
    totals["valid_steps"] += int(metrics.get("valid", 0))
    totals["mira_bonus"] += float(mira_bonus)
    totals["exploration_reward"] += float(exploration_reward)


if __name__ == "__main__":
    main()
