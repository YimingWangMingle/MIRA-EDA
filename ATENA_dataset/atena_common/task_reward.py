import math
from collections import Counter
from atena.evaluation.distance import display_distance
from .simulator import CountedSimulator


def combine_task_reward(env, info):
    return (env.w_kl * info["official_kl"] + env.w_compaction * info["official_compaction"]
            + env.w_official_diversity * info["official_diversity"])


class TaskRewardMixin:
    def common_reward_components(self, action_kind, step_info, invalid, repeated, diversity=None):
        if invalid:
            diversity = self.empty_penalty
        elif repeated and action_kind != "BACK":
            diversity = self.repeat_penalty
        elif diversity is None:
            distances = [display_distance(old, step_info.display).display_distance for old in self.displays[:-1]]
            diversity = float(min(distances)) if distances else 1.0
        kl, compaction = self._official_reward_components(action_kind, step_info, invalid)
        result = dict(official_kl=float(kl), official_compaction=float(compaction), official_diversity=float(diversity))
        result["common_task_reward"] = float(combine_task_reward(self, result))
        return result

    def _official_reward_components(self, action_kind: str, step_info, invalid: bool):
        if action_kind == "BACK" or invalid:
            return 0.0, 0.0
        if action_kind == "FILTER":
            return self._official_kl(step_info), 0.0
        if action_kind == "GROUP":
            return 0.0, self._official_compaction(step_info)
        return 0.0, 0.0

    def _official_kl(self, step_info):
        previous_df = self._previous_filtered_df()
        current_df = step_info._filtered_df
        if len(previous_df) <= 0 or len(current_df) <= 0:
            return self.empty_penalty
        scores = []
        epsilon = 0.2 / max(len(previous_df), 1)
        for column in self.dataset.columns:
            p_counts = self._value_counts(previous_df[column])
            q_counts = self._value_counts(current_df[column])
            keys = set(p_counts) | set(q_counts)
            p_total = sum(p_counts.values()) + (epsilon * len(keys))
            q_total = sum(q_counts.values()) + (epsilon * len(keys))
            kl = 0.0
            for key in keys:
                p = (p_counts.get(key, 0.0) + epsilon) / p_total
                q = (q_counts.get(key, 0.0) + epsilon) / q_total
                kl += p * math.log(p / q)
            scores.append(kl)
        if not scores:
            return 0.0
        return float(1.0 / (1.0 + math.exp(-((max(scores) / 2.0) - 3.0))))

    def _official_compaction(self, step_info):
        if not step_info.state.is_grouped() or step_info._aggregated_df is None:
            return 0.0
        groups = len(step_info._aggregated_df)
        if groups <= 1:
            return self.empty_penalty
        filtered_rows = max(len(step_info._filtered_df), 1)
        grouped_columns = max(len(step_info.state["grouping"]), 1)
        denominator_epsilon = 1e-5
        compact_display = self._normalized_sigmoid(
            center=0.5,
            steepness=17.0,
            value=1.0 - 1.0 / math.log(8.0 + (groups * grouped_columns) + denominator_epsilon, 8.0),
        )
        compact_data = 1.0 - self._normalized_sigmoid(
            center=0.5,
            steepness=17.0,
            value=1.0 - 1.0 / math.log(7.0 + filtered_rows + denominator_epsilon, 7.0),
        )
        return float(compact_display * compact_data)

    def _previous_filtered_df(self):
        if not self.actions:
            return self.dataset.dataset_df
        previous_actions = self.actions[:-1]

        simulator = CountedSimulator(self.dataset, self, replay=True)
        if not previous_actions:
            return self.dataset.dataset_df
        steps = simulator.run_actions(previous_actions)
        if not steps:
            return self.dataset.dataset_df
        return steps[-1]._filtered_df

    @staticmethod
    def _value_counts(series):
        counts = Counter()
        for value in series:
            counts[str(value)] += 1
        return counts

    @staticmethod
    def _normalized_sigmoid(center: float, steepness: float, value: float):
        raw = 1.0 / (1.0 + math.exp(-float(steepness) * (float(value) - float(center))))
        low = 1.0 / (1.0 + math.exp(float(steepness) * float(center)))
        high = 1.0 / (1.0 + math.exp(-float(steepness) * (1.0 - float(center))))
        if abs(high - low) <= 1e-12:
            return raw
        return (raw - low) / (high - low)
