import math
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


if not hasattr(pd.Series, "iteritems"):
    pd.Series.iteritems = pd.Series.items

ROOT_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT_DIR / "ATENA-A-EDA" / "benchmark"
sys.path.insert(0, str(ROOT_DIR.parent))
from revision_support.resources import flush_events
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from atena.evaluation.distance import display_distance  # noqa: E402
sys.path.insert(0, str(ROOT_DIR))
from atena_common.task_reward import TaskRewardMixin, combine_task_reward
from atena_common.costs import QueryCosts
from atena_common.simulator import CountedSimulator

from atena.simulation.actions import (  # noqa: E402
    ActionType,
    AggregationFunction,
    BackAction,
    FilterAction,
    FilterOperator,
    GroupAction,
)
from atena.simulation.dataset import (  # noqa: E402
    CyberDatasetName,
    Dataset,
    DatasetMeta,
    FlightsDatasetName,
    SchemaName,
)
from atena.simulation.tokenization import tokenize_column  # noqa: E402


def dataset_enum(schema: str, dataset_number: int):
    schema_name = SchemaName(schema.lower())
    if schema_name is SchemaName.FLIGHTS:
        return schema_name, FlightsDatasetName(int(dataset_number))
    if schema_name is SchemaName.CYBER:
        return schema_name, CyberDatasetName(int(dataset_number))
    raise ValueError(f"Unsupported official A-EDA schema: {schema}")


class AtenaActionSpace:
    """Finite action index wrapper over ATENA's parameterized actions."""

    def __init__(
        self,
        dataset: Dataset,
        max_terms_per_column: int = 10,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.rng = random.Random(int(seed))
        self.actions = [BackAction()]
        self.action_kinds = ["BACK"]
        self.column_term_map: Dict[str, List[str]] = {}

        for column in self.dataset.columns:
            tokens = tokenize_column(self.dataset.dataset_df, column)
            tokens = sorted(tokens, key=lambda value: str(value))
            if max_terms_per_column > 0:
                # Follow ATENA's bounded filter-term spirit: keep frequent tokens first.
                counts = self.dataset.dataset_df[column].astype(str).value_counts(dropna=False)
                tokens = sorted(tokens, key=lambda value: (-int(counts.get(str(value), 0)), str(value)))
                tokens = tokens[: int(max_terms_per_column)]
            if not tokens:
                tokens = ["<UNK>"]
            self.column_term_map[column] = tokens

            for op in FilterOperator:
                for term in tokens:
                    self.actions.append(FilterAction(column, op, term))
                    self.action_kinds.append("FILTER")

        for grouped_column in self.dataset.columns:
            for aggregated_column in self.dataset.primary_key_columns:
                self.actions.append(
                    GroupAction(grouped_column, aggregated_column, AggregationFunction.COUNT)
                )
                self.action_kinds.append("GROUP")

    def __len__(self):
        return len(self.actions)

    def get(self, index: int):
        return self.actions[int(index)]

    def kind(self, index: int) -> str:
        return self.action_kinds[int(index)]

    def indices_for_kind(self, kind: str) -> List[int]:
        return [idx for idx, value in enumerate(self.action_kinds) if value == kind]


class AtenaEDAEnv(TaskRewardMixin):
    """Official ATENA benchmark environment with task rewards for training."""

    def __init__(
        self,
        schema: str,
        dataset_number: int,
        episode_length: int = 12,
        max_terms_per_column: int = 10,
        seed: int = 0,
        reward_mode: str = "official_compound",
        w_interestingness: float = 1.0,
        w_diversity: float = 2.0,
        w_coherency: float = 1.0,
        w_kl: float = 1.5,
        w_compaction: float = 2.0,
        w_official_diversity: float = 2.0,
        empty_penalty: float = -1.0,
        back_penalty: float = -0.2,
        repeat_penalty: float = -1.0,
        dora_curiosity_ratio: float = 0.25,
        dora_target_size: int = 12,
        dora_target_seed: int = 0,
    ):
        self.schema_name, self.dataset_name = dataset_enum(schema, dataset_number)
        self.dataset_meta = DatasetMeta(self.schema_name, self.dataset_name)
        self.dataset = Dataset(self.dataset_meta)

        self.query_costs = QueryCosts()
        self.simulator = CountedSimulator(self.dataset, self)
        self.action_space = AtenaActionSpace(
            self.dataset,
            max_terms_per_column=max_terms_per_column,
            seed=seed,
        )
        self.episode_length = int(episode_length)
        self.reward_mode = reward_mode
        self.w_interestingness = float(w_interestingness)
        self.w_diversity = float(w_diversity)
        self.w_coherency = float(w_coherency)
        self.w_kl = float(w_kl)
        self.w_compaction = float(w_compaction)
        self.w_official_diversity = float(w_official_diversity)
        self.empty_penalty = float(empty_penalty)
        self.back_penalty = float(back_penalty)
        self.repeat_penalty = float(repeat_penalty)
        self.rng = np.random.default_rng(int(seed))
        self.dora_curiosity_ratio = float(dora_curiosity_ratio)
        self.dora_target_size = int(dora_target_size)
        self.dora_target_seed = int(dora_target_seed)
        self.dora_target_indices = self._build_dora_target_indices()
        self.dora_target_signatures = {
            self._action_signature(self.action_space.get(index))
            for index in self.dora_target_indices
        }
        self.dora_target_columns = {
            (self._action_signature(self.action_space.get(index))[0], self._action_column(self.action_space.get(index)))
            for index in self.dora_target_indices
        }
        self.dora_target_kinds = {
            self._action_signature(self.action_space.get(index))[0]
            for index in self.dora_target_indices
        }
        self.visit_counts = defaultdict(int)
        self.state_dim = self._feature_dim()
        self.reset()

    @property
    def action_dim(self) -> int:
        return len(self.action_space)

    def reset(self):
        self.simulator.reset()
        self.actions = []
        self.step_index = 0
        self.displays = list(self.simulator.simulation_state.displays_history)
        self.display_strings = {str(self.displays[-1])}
        self.prev_action_kind = "START"
        return self._encode_current_state()

    def legal_action_mask(self) -> np.ndarray:
        mask = np.ones(self.action_dim, dtype=np.float32)
        if len(self.simulator.simulation_state.states_stack) <= 1:
            mask[0] = 0.0
        return mask

    def step(self, action_index: int):
        action_index = int(action_index)
        mask = self.legal_action_mask()
        if action_index < 0 or action_index >= self.action_dim or mask[action_index] <= 0:
            action_index = int(self.rng.choice(np.flatnonzero(mask > 0)))

        state_before = self._encode_current_state()
        action = self.action_space.get(action_index)
        prev_display = self.simulator.simulation_state.displays_history[-1]
        step_info = self.simulator.execute_action(action)
        current_display = step_info.display
        self.actions.append(action)
        self.step_index += 1
        self.displays.append(current_display)

        reward_info = self._compute_rewards(action_index, prev_display, current_display, step_info)
        task_reward = self._combine_reward(reward_info)
        done = self.step_index >= self.episode_length
        if done:
            flush_events()
        next_state = self._encode_current_state()

        return next_state, float(task_reward), done, {
            **reward_info,
            "action_index": action_index,
            "action_kind": self.action_space.kind(action_index),
            "action_repr": repr(action),
            "state_before": state_before,
            "state_after": next_state,
        }

    def preview_step(self, action_index: int) -> Tuple[float, Dict[str, object]]:
        action_index = int(action_index)
        mask = self.legal_action_mask()
        if action_index < 0 or action_index >= self.action_dim or mask[action_index] <= 0:
            raise ValueError(f"Action index {action_index} is not legal in the current state")

        snapshot = self._snapshot_mutable_state()
        was_preview = getattr(self, "_profiling_preview", False)
        self._profiling_preview = True
        try:
            _, reward, _, info = self.step(action_index)
            return float(reward), dict(info)
        finally:
            self._profiling_preview = was_preview
            self._restore_mutable_state(snapshot)

    def _snapshot_mutable_state(self) -> Dict[str, object]:
        simulation_state = self.simulator.simulation_state
        return {
            "states_history": list(simulation_state.states_history),
            "states_stack": list(simulation_state.states_stack),
            "displays_history": list(simulation_state.displays_history),
            "actions": list(self.actions),
            "step_index": int(self.step_index),
            "displays": list(self.displays),
            "display_strings": set(self.display_strings),
            "prev_action_kind": self.prev_action_kind,
            "visit_counts": dict(self.visit_counts),
            "rng_state": deepcopy(self.rng.bit_generator.state),
        }

    def _restore_mutable_state(self, snapshot: Mapping[str, object]) -> None:
        simulation_state = self.simulator.simulation_state
        simulation_state.states_history = list(snapshot["states_history"])
        simulation_state.states_stack = list(snapshot["states_stack"])
        simulation_state.displays_history = list(snapshot["displays_history"])
        self.actions = list(snapshot["actions"])
        self.step_index = int(snapshot["step_index"])
        self.displays = list(snapshot["displays"])
        self.display_strings = set(snapshot["display_strings"])
        self.prev_action_kind = str(snapshot["prev_action_kind"])
        self.visit_counts = defaultdict(int, snapshot["visit_counts"])
        self.rng.bit_generator.state = deepcopy(snapshot["rng_state"])

    def sample_legal_action(self) -> int:
        valid = np.flatnonzero(self.legal_action_mask() > 0)
        return int(self.rng.choice(valid))

    def _compute_rewards(self, action_index, prev_display, current_display, step_info):
        action_kind = self.action_space.kind(action_index)
        action = self.action_space.get(action_index)
        filtered_df = step_info._filtered_df
        aggregated_df = step_info._aggregated_df
        display_key = str(current_display)

        is_grouped = bool(step_info.state.is_grouped())
        invalid = len(filtered_df) == 0 or (is_grouped and aggregated_df is None)
        repeated = display_key in self.display_strings
        self.display_strings.add(display_key)

        if action_kind == "BACK":
            interestingness = 0.0
            coherency = self.back_penalty
        elif invalid:
            interestingness = self.empty_penalty
            coherency = self.empty_penalty
        else:
            interestingness = self._interestingness(step_info)
            coherency = self._coherency(action_kind)

        if repeated and action_kind != "BACK":
            diversity = self.repeat_penalty
        else:
            from atena.evaluation.distance import display_distance

            distances = [
                display_distance(history_display, current_display).display_distance
                for history_display in self.displays[:-1]
            ]
            diversity = float(min(distances)) if distances else 1.0

        state_hash = hash(self._encode_current_state().tobytes())
        self.visit_counts[state_hash] += 1
        dora_curiosity = 1.0 / math.sqrt(float(self.visit_counts[state_hash]))
        dora_familiarity, dora_target_hit = self._dora_familiarity(action)
        common = self.common_reward_components(action_kind, step_info, invalid, repeated, diversity=diversity)

        return {
            "interestingness": float(interestingness),
            "diversity": float(diversity),
            "coherency": float(coherency),
            **common,
            "dora_curiosity": float(dora_curiosity),
            "dora_familiarity": float(dora_familiarity),
            "dora_target_hit": float(dora_target_hit),
            "invalid": float(invalid),
            "repeated": float(repeated),
        }

    def _combine_reward(self, reward_info):
        if self.reward_mode == "interestingness":
            return reward_info["interestingness"]
        if self.reward_mode == "dora":
            ratio = min(max(self.dora_curiosity_ratio, 0.0), 1.0)
            return (
                (1.0 - ratio) * reward_info["dora_familiarity"]
                + ratio * reward_info["dora_curiosity"]
            )
        if self.reward_mode == "compound":
            return (
                self.w_interestingness * reward_info["interestingness"]
                + self.w_diversity * reward_info["diversity"]
                + self.w_coherency * reward_info["coherency"]
            )
        if self.reward_mode == "task_only":
            return (
                self.w_interestingness * reward_info["interestingness"]
                + self.w_diversity * reward_info["diversity"]
                + self.w_coherency * reward_info["coherency"]
            )
        if self.reward_mode == "official_compound":
            return self._official_compound_reward(reward_info)
        raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

    def _official_compound_reward(self, reward_info):
        return combine_task_reward(self, reward_info)

    def _build_dora_target_indices(self):
        target_size = max(1, int(self.dora_target_size))
        schema_offset = 0 if self.schema_name is SchemaName.FLIGHTS else 100000
        seed = int(self.dora_target_seed) + schema_offset + (1009 * int(self.dataset_name.value))
        rng = np.random.default_rng(seed)

        filter_indices = self.action_space.indices_for_kind("FILTER")
        group_indices = self.action_space.indices_for_kind("GROUP")
        group_quota = min(len(group_indices), target_size // 2)
        filter_quota = min(len(filter_indices), target_size - group_quota)
        remaining = target_size - group_quota - filter_quota
        if remaining > 0:
            extra_groups = min(len(group_indices) - group_quota, remaining)
            group_quota += extra_groups
            remaining -= extra_groups
        if remaining > 0:
            filter_quota += min(len(filter_indices) - filter_quota, remaining)

        selected = []
        if filter_quota > 0:
            selected.extend(rng.choice(filter_indices, size=filter_quota, replace=False).astype(int).tolist())
        if group_quota > 0:
            selected.extend(rng.choice(group_indices, size=group_quota, replace=False).astype(int).tolist())
        rng.shuffle(selected)
        return selected

    def dora_target_actions_repr(self):
        return [repr(self.action_space.get(index)) for index in self.dora_target_indices]

    def _dora_familiarity(self, action):
        signature = self._action_signature(action)
        kind = signature[0]
        column = self._action_column(action)
        if signature in self.dora_target_signatures:
            return 1.0, 1.0
        if (kind, column) in self.dora_target_columns:
            return 0.5, 0.0
        if kind in self.dora_target_kinds:
            return 0.25, 0.0
        return 0.0, 0.0

    def _action_signature(self, action):
        if isinstance(action, FilterAction):
            return (
                "FILTER",
                str(action.filtered_column),
                str(action.filter_operator.name),
                str(action.filter_term),
            )
        if isinstance(action, GroupAction):
            return (
                "GROUP",
                str(action.grouped_column),
                str(action.aggregated_column),
                str(action.aggregation_function.name),
            )
        return ("BACK",)

    def _action_column(self, action):
        if isinstance(action, FilterAction):
            return str(action.filtered_column)
        if isinstance(action, GroupAction):
            return str(action.grouped_column)
        return ""







    def _interestingness(self, step_info):
        if step_info.state.is_grouped() and step_info._aggregated_df is not None:
            groups = max(len(step_info._aggregated_df), 1)
            total = max(len(step_info._filtered_df), 1)
            compact = 1.0 - min(groups / total, 1.0)
            return float(compact)

        parent_size = max(len(self.dataset.dataset_df), 1)
        current_size = max(len(step_info._filtered_df), 0)
        selectivity = 1.0 - min(current_size / parent_size, 1.0)
        non_empty_bonus = 0.25 if current_size > 0 else -1.0
        return float(selectivity + non_empty_bonus)

    def _coherency(self, action_kind: str):
        if action_kind == "BACK":
            return self.back_penalty
        if self.prev_action_kind == "START":
            self.prev_action_kind = action_kind
            return 0.25
        coherent_pairs = {
            ("GROUP", "FILTER"),
            ("FILTER", "GROUP"),
            ("FILTER", "FILTER"),
            ("GROUP", "GROUP"),
        }
        score = 0.5 if (self.prev_action_kind, action_kind) in coherent_pairs else 0.0
        self.prev_action_kind = action_kind
        return float(score)

    def _feature_dim(self):
        columns = len(self.dataset.columns)
        return (columns * 3) + 5 + 3 + 1

    def _encode_current_state(self):
        display = self.simulator.simulation_state.displays_history[-1]
        data_layer = display["data_layer"]
        features: List[float] = []
        for column in self.dataset.columns:
            col = data_layer[column]
            features.extend([
                float(col["unique"]),
                float(col["nulls"]),
                float(col["entropy"]),
            ])

        gran = display["granularity_layer"]
        if gran is None:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([
                float(len(gran["group_attrs"]) / max(len(self.dataset.columns), 1)),
                float(len(gran["agg_attrs"]) / max(len(self.dataset.primary_key_columns), 1)),
                float(gran["inverse_ngroups"]),
                float(gran["site_std"]),
                float(gran["inverse_size_mean"]),
            ])

        action_one_hot = [0.0, 0.0, 0.0]
        if self.prev_action_kind == "BACK":
            action_one_hot[0] = 1.0
        elif self.prev_action_kind == "FILTER":
            action_one_hot[1] = 1.0
        elif self.prev_action_kind == "GROUP":
            action_one_hot[2] = 1.0
        features.extend(action_one_hot)
        features.append(float(self.step_index / max(self.episode_length, 1)))
        return np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=5.0, neginf=-5.0)


def make_env(schema, dataset_number, seed, args, reward_mode: Optional[str] = None):
    return AtenaEDAEnv(
        schema=schema,
        dataset_number=dataset_number,
        episode_length=args.episode_length,
        max_terms_per_column=args.max_terms_per_column,
        seed=seed,
        reward_mode=reward_mode or args.reward_mode,
        w_interestingness=args.w_interestingness,
        w_diversity=args.w_diversity,
        w_coherency=args.w_coherency,
        w_kl=getattr(args, "w_kl", 1.5),
        w_compaction=getattr(args, "w_compaction", 2.0),
        w_official_diversity=getattr(args, "w_official_diversity", 2.0),
        empty_penalty=args.empty_penalty,
        back_penalty=args.back_penalty,
        repeat_penalty=args.repeat_penalty,
        dora_curiosity_ratio=getattr(args, "dora_curiosity_ratio", 0.25),
        dora_target_size=getattr(args, "dora_target_size", args.episode_length),
        dora_target_seed=getattr(args, "dora_target_seed", 0),
    )
