import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from typing import List
from app.pipelines.pipeline_precalculated_sets import PipelineWithPrecalculatedSets
from app.pipelines.dataset import Dataset


class StateEncoder:
    def __init__(self, pipeline: PipelineWithPrecalculatedSets, target_items=None, found_items_with_ratio=None, target_set_size=2000):
        self.pipeline = pipeline

        # 原论文外部奖励参数
        self.target_ratio = 0.1
        self.target_max_reward = 100
        self.target_set_size = target_set_size
        if target_items is not None:
            self.target_items = set(target_items)
            self.reward_multiplier = self.target_max_reward / (len(self.target_items) * self.target_ratio)
        else:
            self.target_items = []
            self.reward_multiplier = 0

        self.found_items_with_ratio = found_items_with_ratio if found_items_with_ratio is not None else {}

        self.set_description = ["item count"]
        for column in self.pipeline.exploration_columns:
            self.set_description.append(f"description {column}")
            self.set_description.append(f"distinct {column}")
            self.set_description.append(f"entropy {column}")

    def reset(self):
        self.found_items_with_ratio = {}

    def encode_datasets(self, datasets: List[Dataset], parent_dataset: Dataset = None):
        encoded_sets = []
        total_extrinsic_reward = 0.0
        total_interestingness_reward = 0.0

        for dataset in datasets:
            encoded_set, r_ext, r_int_js = self.encode_dataset(dataset, parent_dataset)
            encoded_sets.extend(encoded_set)
            total_extrinsic_reward += r_ext
            total_interestingness_reward += r_int_js

        if len(datasets) > 0:
            total_interestingness_reward = float(total_interestingness_reward / len(datasets))

        if len(datasets) < 10:
            encoded_sets += [0.0] * (10 - len(datasets)) * len(self.set_description)

        return encoded_sets, total_extrinsic_reward, total_interestingness_reward

    def encode_dataset(self, dataset: Dataset, parent_dataset: Dataset = None, get_reward=True):
        encoded_set = []
        r_ext = 0.0
        r_int_js = 0.0
        data = dataset.data

        # 对齐原论文的外部奖励定义：
        # 1) 命中依据用 galaxies.objID
        # 2) 条件要求 get_reward and dataset.set_id is not None
        # 3) reward_set_size_ratio 用原论文公式
        if get_reward and dataset.set_id is not None and not data.empty and len(self.target_items) > 0:
            dataset_obj_ids = set(map(int, data["galaxies.objID"].tolist()))
            target_items_int = set(map(int, self.target_items))

            original_target_found_in_dataset = dataset_obj_ids & target_items_int
            new_target_found_in_dataset = original_target_found_in_dataset - set(map(int, self.found_items_with_ratio.keys()))

            reward_set_size_ratio = (
                len(original_target_found_in_dataset) / len(data)
            ) * (
                self.target_set_size / len(data)
            )

            if len(new_target_found_in_dataset) > 0:
                r_ext += len(new_target_found_in_dataset) * reward_set_size_ratio * self.reward_multiplier
                ratio_item_dict = dict(
                    zip(
                        map(str, new_target_found_in_dataset),
                        [reward_set_size_ratio] * len(new_target_found_in_dataset)
                    )
                )
                self.found_items_with_ratio.update(ratio_item_dict)

            old_target_found_in_dataset = original_target_found_in_dataset - new_target_found_in_dataset
            better_ratio_items = list(filter(
                lambda x: int(x) in old_target_found_in_dataset and self.found_items_with_ratio[x] < reward_set_size_ratio,
                self.found_items_with_ratio
            ))

            if len(better_ratio_items) > 0:
                r_ext += len(better_ratio_items) * reward_set_size_ratio * self.reward_multiplier
                ratio_item_dict = dict(
                    zip(better_ratio_items, [reward_set_size_ratio] * len(better_ratio_items))
                )
                self.found_items_with_ratio.update(ratio_item_dict)

        # 保留优化版内部奖励（JS divergence）
        if parent_dataset is not None and not parent_dataset.data.empty and not data.empty:
            divergence_sum = 0.0
            valid_dims = 0
            for dim in self.pipeline.exploration_columns:
                p_counts = parent_dataset.data[dim].value_counts(normalize=True)
                q_counts = data[dim].value_counts(normalize=True)
                all_keys = set(p_counts.keys()).union(set(q_counts.keys()))
                if not all_keys:
                    continue

                p_dist = np.array([p_counts.get(k, 0.0) for k in all_keys])
                q_dist = np.array([q_counts.get(k, 0.0) for k in all_keys])
                js_distance = jensenshannon(p_dist, q_dist)

                if not np.isnan(js_distance):
                    divergence_sum += (js_distance ** 2)
                    valid_dims += 1

            if valid_dims > 0:
                avg_divergence = divergence_sum / valid_dims
                min_size = min(len(data), len(parent_dataset.data))
                max_size = max(len(data), len(parent_dataset.data))
                ratio = min_size / max_size if max_size > 0 else 0.0
                compactness_weight = max(0.0, 1.0 - abs(ratio - 0.5) * 2.0)
                r_int_js = float(avg_divergence * compactness_weight)

        # 状态提取
        encoded_set.append(float(np.log10(len(data) + 1)))
        for dimension in self.pipeline.exploration_columns:
            predicate_item = next((x for x in dataset.predicate.components if x.attribute == dimension), None)
            if predicate_item is not None:
                encoded_set.append(float(self.pipeline.ordered_dimensions[dimension].index(str(predicate_item.value))))
            else:
                encoded_set.append(0.0)

            encoded_set.append(float(np.log10(data[dimension].nunique() + 1)))
            counts = data[dimension].value_counts()
            encoded_set.append(float(entropy(counts) if not counts.empty else 0.0))

        return encoded_set, r_ext, r_int_js
