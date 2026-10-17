from app.pipelines.predicateitem import PredicateItem
from app.pipelines.dataset import Dataset
from app.pipelines import pipeline
import gym
import numpy as np
import json
import random
from gym import spaces
from os import listdir

import numpy as np
piecewise = np.piecewise
from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from .state_encoder import StateEncoder
from .target_set_generator import TargetSetGenerator
from .action_manager import ActionManager
from revision_support.resources import flush_events, query_step
from revision_support.galaxy_execution import execute_galaxy_operator


class PipelineEnvironment(gym.Env):
    def __init__(self, pipeline: PipelineWithPrecalculatedSets, mode="simple", target_set_name=None, number_of_examples=3, agentId=-1, episode_steps=50, target_items=None, operators=[], target_seed=None, target_samples_per_file=100, observation_only=False):
        self.observation_only = bool(observation_only)
        self.pipeline = pipeline
        self.mode = mode
        self.target_set_name = target_set_name
        self.target_seed = target_seed
        self.target_samples_per_file = target_samples_per_file
        self.number_of_examples = number_of_examples
        self.episode_steps = episode_steps
        self.agentId = agentId
        self.systemRandom = random.SystemRandom()
        self.exploration_dimensions = self.pipeline.exploration_columns
        self.target_set_index = -1


        if self.target_set_name not in [None, "None"] and target_items is None:
            with open(f"./rl/targets/{self.target_set_name}.json") as f:
                exact_target_items = set([int(x) for x in json.load(f)])
                self.target_items = exact_target_items
                self.state_encoder = StateEncoder(
                    pipeline, target_items=exact_target_items, target_set_size=2000)
            print(f"Loaded fixed target set '{self.target_set_name}' with {len(exact_target_items)} objects.")
        elif self.mode == "scattered" and target_items is None:

            raw_17k_targets = TargetSetGenerator.get_diverse_target_set(
                number_of_samples=self.target_samples_per_file,
                seed=self.target_seed
            )


            exact_17k_targets = set([int(x) for x in raw_17k_targets])


            sample_target = list(exact_17k_targets)[0]

            df_type = pipeline.initial_collection["galaxies.objID"].dtype 

            print(f"Loaded STRICT Paper Baseline Target Set with {len(exact_17k_targets)} objects!")
            print(f"[Target Protocol] seed={self.target_seed} samples_per_file={self.target_samples_per_file}")
            print(f"[🔬 Type Diagnostic] Target type: {type(sample_target)} | DataFrame objID type: {df_type}")


            self.target_items = exact_17k_targets
            self.state_encoder = StateEncoder(
                pipeline,
                target_items=exact_17k_targets,
                target_set_size=2000
            )


        else:
            self.target_items = set(target_items) if target_items is not None else set()
            self.state_encoder = StateEncoder(
                pipeline, target_items=target_items)

        self.action_manager = ActionManager(pipeline, operators=operators)
        self.set_action_space = spaces.Discrete(self.pipeline.discrete_categories_count)
        self.operation_action_space = spaces.Discrete(len(self.action_manager.set_action_types))


        self.set_state_dim = self.pipeline.discrete_categories_count * len(self.state_encoder.set_description)
        if self.mode == "by_example":
            self.set_state_dim += len(self.state_encoder.set_description)
        region_dim = len(self.state_encoder.set_description)
        self.statistic_indices = [offset + index
            for offset in range(0, self.set_state_dim, region_dim)
            for index in self.state_encoder.statistic_indices]
        self.operation_state_dim = self.set_state_dim + len(self.state_encoder.set_description)

        self.reset()

    def get_set_state(self):

        encoded_sets, r_ext, r_int_js = self.state_encoder.encode_datasets(
            datasets=self.datasets, parent_dataset=self.input_set,
            get_reward=not self.observation_only)
        if not self.observation_only:
            self._cache_latest_dataset_states(encoded_sets)
        state = []
        if self.mode == 'by_example':
            state += self.example_state


        descriptor = state + encoded_sets
        raw_state_array = np.array(descriptor, dtype=np.float32)
        safe_state_array = np.nan_to_num(raw_state_array, nan=0.0, posinf=0.0, neginf=0.0)

        return safe_state_array, r_ext, r_int_js

    def _safe_set_id(self, set_id):
        try:
            parsed = int(set_id)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    def _cache_latest_dataset_states(self, encoded_sets):
        self.latest_dataset_state_rows = []
        if not self.datasets:
            return

        state_dim = len(self.state_encoder.set_description)
        for index, dataset in enumerate(self.datasets):
            set_id = self._safe_set_id(dataset.set_id)
            if set_id is None:
                continue
            start = index * state_dim
            end = start + state_dim
            if end > len(encoded_sets):
                continue
            state = np.nan_to_num(
                np.array(encoded_sets[start:end], dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(float).tolist()
            self.latest_dataset_state_rows.append({
                "set_id": set_id,
                "state": state,
            })

    def _record_exploration_step(self, r_ext, r_int_js, set_action_array, set_index, operation_action):
        if not self.latest_dataset_state_rows:
            return

        operator = set_action_array[0] if len(set_action_array) > 0 else None
        parameter = set_action_array[1] if len(set_action_array) > 1 else None
        input_set_id = self._safe_set_id(self.input_set.set_id if self.input_set is not None else None)
        for row in self.latest_dataset_state_rows:
            set_id = row["set_id"]
            self.exploration_trace_rows.append({
                "agent_id": int(self.agentId),
                "step": int(self.step_count),
                "set_id": set_id,
                "step_extrinsic_reward": float(r_ext),
                "step_interestingness": float(r_int_js),
                "operator": operator,
                "parameter": parameter,
                "input_set_id": input_set_id if input_set_id is not None else -1,
                "operation_action": int(operation_action),
            })
            if set_id not in self.visited_set_state_rows:
                self.visited_set_state_rows[set_id] = row["state"]

    def consume_exploration_logs(self):
        flush_events()
        trace_rows = list(self.exploration_trace_rows)
        state_rows = [
            {"set_id": set_id, "state": state}
            for set_id, state in self.visited_set_state_rows.items()
        ]
        return trace_rows, state_rows

    def get_operation_state(self, set_index):
        if len(self.datasets) == 0:
            dataset = self.pipeline.get_dataset()
        else:
            dataset = self.datasets[set_index]

        encoded_set, _, _ = self.state_encoder.encode_dataset(dataset, get_reward=not self.observation_only)

        operation_state = np.concatenate([
            self.set_state,
            np.array(encoded_set, dtype=np.float32)
        ])


        safe_operation_state = np.nan_to_num(operation_state, nan=0.0, posinf=0.0, neginf=0.0)

        return safe_operation_state

    def reset(self):
        self.step_count = 0
        self.datasets = []
        self.input_set = self.pipeline.get_dataset()
        self.set_review_counter = 0
        self.sets_viewed = set()
        self.episode_info = []
        self.exploration_trace_rows = []
        self.visited_set_state_rows = {}
        self.latest_dataset_state_rows = []
        self.state_encoder.reset()

        if self.mode == "by_example":

            target_list = list(self.state_encoder.target_items)
            target_set_id = self.systemRandom.sample(target_list, 1)[0]
            example_dataset = self.pipeline.get_dataset(set_id=target_set_id)
            self.example_state, _, _ = self.state_encoder.encode_dataset(example_dataset)

        self.set_state, r_ext, r_int_js = self.get_set_state()
        return self.set_state

    def fix_possible_set_action_probs(self, probs):
        return self.action_manager.fix_possible_set_action_probs(self.datasets, probs)

    def fix_possible_operation_action_probs(self, set_index, probs):
        if len(self.datasets) == 0:
            dataset = self.pipeline.get_dataset()
        else:
            dataset = self.datasets[set_index]
        return self.action_manager.fix_possible_operation_action_probs(dataset, probs)

    def get_random_operation(self, set_index):
        probs = self.fix_possible_operation_action_probs(set_index, [1.0/self.operation_action_space.n]*self.operation_action_space.n)
        return np.random.choice(self.operation_action_space.n, p=probs)

    @query_step
    def step(self, set_action, operation_action):
        self.step_count += 1
        set_action_array = self.action_manager.set_action_types[operation_action].split("-&-")

        if len(self.datasets) == 0:
            self.input_set = self.pipeline.get_dataset()
            set_index = -1
        else:
            self.input_set = self.datasets[set_action]
            set_index = set_action

        original_datasets = self.datasets
        try:
            returned_datasets = execute_galaxy_operator(
                self.pipeline, self.input_set,
                self.action_manager.set_action_types[operation_action])
        except Exception as error:
            raise RuntimeError(
                f"Galaxy operator failed: worker={self.agentId}, step={self.step_count}, "
                f"action={self.action_manager.set_action_types[operation_action]}, "
                f"input_set_id={self.input_set.set_id}"
            ) from error

        self.datasets = returned_datasets
        done = self.step_count >= self.episode_steps
        if len(self.datasets) == 0:
            self.datasets = original_datasets
            self.latest_dataset_state_rows = []
            r_ext, r_int_js = 0.0, 0.0
        else:
            result_set_ids = set(map(lambda x: x.set_id, self.datasets))
            self.set_review_counter += len(result_set_ids & self.sets_viewed)
            self.sets_viewed.update(result_set_ids)


            self.set_state, r_ext, r_int_js = self.get_set_state()

        if self.observation_only:
            return self.set_state, r_ext, r_int_js, done, f"{set_index}-{operation_action}"

        self.episode_info.append({
            "input_set_index": set_index,
            "input_set_size": len(self.input_set.data),
            "input_set_id": self.input_set.set_id if self.input_set.set_id != None else -1,
            "operator": set_action_array[0],
            "parameter": set_action_array[1] if len(set_action_array) > 1 else None,
            "output_set_count": len(returned_datasets),
            "output_set_average_size": sum(len(x.data) for x in returned_datasets)/len(returned_datasets) if returned_datasets else 0,
            "reward_extrinsic": r_ext,
            "reward_intrinsic_js": r_int_js,
            "sets_viewed": len(self.sets_viewed)
        })
        if returned_datasets:
            self._record_exploration_step(r_ext, r_int_js, set_action_array, set_index, operation_action)
        else:

            self.exploration_trace_rows.append({
                "agent_id": int(self.agentId), "step": int(self.step_count),
                "set_id": -1, "step_extrinsic_reward": 0.0,
                "step_interestingness": 0.0, "operator": set_action_array[0],
                "parameter": set_action_array[1] if len(set_action_array) > 1 else None,
                "input_set_id": self.input_set.set_id if self.input_set.set_id is not None else -1,
                "operation_action": int(operation_action),
            })
        return self.set_state, r_ext, r_int_js, done, f"{set_index}-{operation_action}"
