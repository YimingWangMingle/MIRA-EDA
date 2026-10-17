import math
import os
import sys
import unittest
import logging
from collections import deque
from functools import reduce, lru_cache

import numpy as np
from scipy.stats import entropy

import Utilities.Configuration.config as cfg
from Utilities.Collections.Counter_Without_Nans import CounterWithoutNanKeys
from arguments import FilterTermsBinsSizes
from gym_atena.lib.helpers import (
    DisplayTuple
)

logger = logging.getLogger(__name__)


class EnvDatasetProp(object):


    def __init__(self,
                 repo,
                 KEYS,
                 KEYS_ANALYST_STR,
                 FILTER_COLS,
                 GROUP_COLS,
                 AGG_KEYS,
                 AGG_KEYS_ANALYST_STR,
                 NUMERIC_KEYS,
                 FILTER_LIST=None,
                 FILTER_BY_FIELD_DICT=None,
                 DONT_FILTER_FIELDS=None
                 ):

        self.repo = repo

        self.KEYS = KEYS
        self.KEYS_ANALYST_STR = KEYS_ANALYST_STR
        self.FILTER_COLS = FILTER_COLS
        self.GROUP_COLS = GROUP_COLS
        self.AGG_KEYS = AGG_KEYS
        self.AGG_KEYS_ANALYST_STR = AGG_KEYS_ANALYST_STR
        self.NUMERIC_KEYS = NUMERIC_KEYS
        self.FILTER_LIST = FILTER_LIST if FILTER_LIST is not None else []
        self.FILTER_BY_FIELD_DICT = FILTER_BY_FIELD_DICT if FILTER_BY_FIELD_DICT is not None else {}
        self.DONT_FILTER_FIELDS = DONT_FILTER_FIELDS if DONT_FILTER_FIELDS is not None else {}

        self.KEYS_MAP_ANALYST_STR = {col: col_str for col, col_str in zip(self.KEYS, self.KEYS_ANALYST_STR)}
        self.ANALYST_STR_MAP_KEYS = {col_str: col for col, col_str in zip(self.KEYS, self.KEYS_ANALYST_STR)}
        self.FILTER_COLS_MAP = {col: idx for idx, col in enumerate(self.FILTER_COLS)}

        self.COLS_NO = len(self.KEYS)
        self.AGG_COLS_NO = len(self.AGG_KEYS)
        self.FILTER_TERMS_NO = len(self.FILTER_LIST)


class EnvProp(object):
    ACTION_RANGE = 6.0

    def __init__(self,
                 OPERATOR_TYPE_LOOKUP,
                 INT_OPERATOR_MAP_ATENA,
                 AGG_MAP_ATENA,
                 env_dataset_prop,
                 ):


        self.OPERATOR_TYPE_LOOKUP = OPERATOR_TYPE_LOOKUP
        self.INT_OPERATOR_MAP_ATENA = INT_OPERATOR_MAP_ATENA
        self.AGG_MAP_ATENA = AGG_MAP_ATENA
        self.env_dataset_prop = env_dataset_prop

        self.ACTION_TYPES_NO = len(self.OPERATOR_TYPE_LOOKUP)
        self.FILTER_OPS = len(self.INT_OPERATOR_MAP_ATENA)
        self.AGG_FUNCS_NO = len(self.AGG_MAP_ATENA)


        self.MAX_FILTER_TERMS_BY_FIELD_NO = 1.0
        self.bins = self.DISCRETE_FILTER_TERM_BINS_NUM = None
        self.create_filter_term_bins()


        self.MAP_PARAMETRIC_SOFMAX_IDX_TO_DISCRETE_ACTION, self.MAP_PARAMETRIC_SOFMAX_DISCRETE_ACTION_TO_IDX = (
            self.get_parametric_softmax_idx_action_mapping())

    def create_filter_term_bins(self):

        filter_terms_bin_sizes = FilterTermsBinsSizes(cfg.bins_sizes)
        if filter_terms_bin_sizes is FilterTermsBinsSizes.EQUAL_WIDTH:
            self.DISCRETE_FILTER_TERM_BINS_NUM = 11
        elif filter_terms_bin_sizes is FilterTermsBinsSizes.CUSTOM_WIDTH:
            self.bins = [0] + [0.1 / 2 ** i for i in range(11, 0, -1)] + [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                                                                          0.5, 0.55,
                                                                          0.6, 0.7, 0.8, 0.9, 1.0]
            self.DISCRETE_FILTER_TERM_BINS_NUM = len(self.bins) - 1
        elif filter_terms_bin_sizes is FilterTermsBinsSizes.EXPONENTIAL:
            self.bins = [None] * cfg.exponential_sizes_num_of_bins
            self.DISCRETE_FILTER_TERM_BINS_NUM = cfg.exponential_sizes_num_of_bins
        else:
            raise NotImplementedError

    def compressed2full_range(self, action_vec, continuous_filter_term=True):


        RANGE = self.ACTION_RANGE
        entries_ranges = np.array([self.ACTION_TYPES_NO,
                                   self.env_dataset_prop.COLS_NO,
                                   self.FILTER_OPS,
                                   self.MAX_FILTER_TERMS_BY_FIELD_NO,
                                   self.env_dataset_prop.AGG_COLS_NO,
                                   self.AGG_FUNCS_NO])
        full_range = np.multiply(np.array((action_vec + RANGE / 2) / RANGE), entries_ranges) - 0.5
        full_range_filter_term = full_range[3]
        clipped = np.clip(full_range, np.zeros(6), entries_ranges - 1)
        if continuous_filter_term:
            clipped[3] = full_range_filter_term
        return clipped

    def get_parametric_segments(self):
        raise NotImplementedError

    def get_pre_output_layer_size(self):

        filter_terms_num = self.DISCRETE_FILTER_TERM_BINS_NUM

        return (len(self.OPERATOR_TYPE_LOOKUP) +
                len(self.env_dataset_prop.GROUP_COLS) +
                len(self.env_dataset_prop.FILTER_COLS) +
                len(self.INT_OPERATOR_MAP_ATENA) // 3 +
                filter_terms_num)

    @staticmethod
    def get_seg_size(seg):
        if not seg:
            return 1
        else:
            return reduce(lambda x, y: x * y, seg, 1)

    def get_parametric_softmax_segments_sizes(self):
        parametric_segments = self.get_parametric_segments()
        return [self.get_seg_size(seg) for seg in parametric_segments]

    def get_parametric_softmax_output_layer_size(self):
        return sum(self.get_parametric_softmax_segments_sizes())

    def parametric_softmax_idx_to_discrete_action(self, idx, segs):
        raise NotImplementedError

    def get_softmax_layer_size(self):
        raise NotImplementedError

    def get_parametric_softmax_idx_action_mapping(self):
        parametric_segments = self.get_parametric_segments()
        output_layer_size = self.get_parametric_softmax_output_layer_size()
        idx_to_action = {idx: self.parametric_softmax_idx_to_discrete_action(idx, parametric_segments) for
                         idx in range(output_layer_size)}
        action_to_idx = {act: idx for idx, act in idx_to_action.items()}
        return idx_to_action, action_to_idx

    def static_param_softmax_idx_to_action_type(self, idx):

        result = np.zeros(1, dtype=np.float32)
        (action_type, parameters) = self.MAP_PARAMETRIC_SOFMAX_IDX_TO_DISCRETE_ACTION[idx]

        result[0] = action_type

        return result

    def get_filtered_df(self, dataset_df, filter_tpl):


        if not filter_tpl:
            return dataset_df

        df = dataset_df.copy()

        for filt in filter_tpl:
            field = filt.field
            op_num = filt.condition
            value = filt.term

            opr = self.INT_OPERATOR_MAP_ATENA.get(op_num)
            if opr is not None:
                try:
                    if (df[field].dtype == 'f8' or df[field].dtype == 'u4' or df[field].dtype == 'int64') and value == '<UNK>':
                        value = math.nan
                    value = float(value) if str(df[field].dtype) not in ['object', 'category'] and value != '<UNK>' else value


                    df = df[opr(df[field], value)]

                except:
                    logger.warning(f"Filter on column {field} with operator {opr} and value {value} is emtpy")
                    return df.truncate(after=-1)
            else:


                try:

                    if op_num in [6, 7, 8]:
                        if df[field].dtype == 'O' or str(df[field].dtype) == 'category':
                            df = df[df[field].str.contains(value, na=False, regex=False)]
                        elif df[field].dtype == 'f8' or df[field].dtype == 'u4' or df[field].dtype == 'int64':
                            df = df[df[field].astype(str).str.contains(str(value), na=False, regex=False)]
                        else:
                            logger.warning(f"Filter on column {field} with operator Contains and value {value} is emtpy")
                            raise NotImplementedError
                    elif op_num == 7:
                        df = df[df[field].str.startswith(value, na=False)]
                    elif op_num == 8:
                        df = df[df[field].str.endswith(value, na=False)]
                    else:
                        logger.warning(f"Filter on column {field} with operator number {op_num} and value {value} raised NotImplementedError and will be emtpy")
                        raise NotImplementedError
                except NotImplementedError:
                    return df.truncate(after=-1)

        return df

    def get_groupby_df(self, df, groupings, aggregations):

        if not groupings:
            return None, None

        df_gb = df.groupby(list(groupings), observed=True)


        agg_dict = {}

        for agg in aggregations:
            agg_dict[agg.field] = self.AGG_MAP_ATENA.get(agg.type)

        try:
            agg_df = df_gb.agg(agg_dict)
        except:
            return None, None
        return df_gb, agg_df

    def get_data_column_measures(self, column):

        B = 20
        size = len(column)
        if size == 0:
            return {"unique": 0.0, "nulls": 1.0, "entropy": 0.0}
        column_na_value_counts = CounterWithoutNanKeys(column)
        column_na_value_counts_values = column_na_value_counts.values()
        cna_size = sum(column_na_value_counts_values)

        n = size - cna_size


        u = len(column_na_value_counts.keys())

        u_n = u / cna_size if u != 0 else 0

        if column.name not in self.env_dataset_prop.NUMERIC_KEYS:

            h = entropy(list(column_na_value_counts_values))
            h = h / math.log(cna_size) if cna_size > 1 else 0.0
        else:
            h = entropy(np.histogram(column_na_value_counts.non_nan_iterable, bins=B)[0]) / math.log(
                B) if cna_size > 1 else 0.0

        return {"unique": u_n, "nulls": n / size, "entropy": h}

    def calc_data_layer(self, df):
        if len(df) == 0:
            ret_dict = {}
            for k in self.env_dataset_prop.KEYS:
                ret_dict[k] = {"unique": 0.0, "nulls": 1.0, "entropy": 0.0}
            return ret_dict
        else:

            return {key: self.get_data_column_measures(df[key]) for key in self.env_dataset_prop.KEYS}

    def get_grouping_measures(self, group_obj, agg_df):
        if group_obj is None or agg_df is None:
            return None

        B = 20
        groups_num = group_obj.ngroups
        if groups_num == 0:
            site_std = 0.0
            inverse_size_mean = 0
            inverse_ngroups = 0
        else:
            sizes = group_obj.size()
            sizes_sum = sizes.sum()
            nsizes = sizes / sizes_sum
            site_std = nsizes.std(ddof=0)

            sizes_mean = sizes.mean()
            inverse_size_mean = 1 / sizes_mean

            if sizes_sum > 0:


                inverse_ngroups = 1 / groups_num
            else:
                inverse_ngroups = 0
                inverse_size_mean = 0

        group_keys = group_obj.keys
        agg_keys = list(agg_df.keys())
        agg_nve_dict = {}
        if agg_keys is not None:
            for ak in agg_keys:
                column = agg_df[ak]
                column_na = column.dropna()
                cna_size = len(column_na)
                if cna_size <= 1:
                    agg_nve_dict[ak] = 0.0

                elif agg_df[ak].dtype == 'O' or str(agg_df[ak].dtype) == 'category':
                    h = entropy(column_na.value_counts().values)
                    agg_nve_dict[ak] = h / math.log(cna_size)
                else:
                    agg_nve_dict[ak] = entropy(np.histogram(column_na, bins=B)[0]) / math.log(B)
        return {"group_attrs": group_keys, "agg_attrs": agg_nve_dict, "inverse_ngroups": inverse_ngroups,
                "site_std": site_std, "inverse_size_mean": inverse_size_mean}

    def calc_gran_layer(self, group_obj, agg_df):

        return self.get_grouping_measures(group_obj, agg_df)

    def get_raw_data(self, dataset_df, state):

        filtered_df = self.get_filtered_df(dataset_df, state["filtering"])
        gdf, agg_df = self.get_groupby_df(filtered_df, state["grouping"], state["aggregations"])
        return filtered_df, gdf, agg_df

    def get_state_dfs(self, dataset_df, state, memo=None, dataset_number=None):
        return self.calc_display_vector(dataset_df, state, memo, dataset_number)[2]

    def calc_display_vector(self, dataset_df, state, memo=None, dataset_number=None, step_number=None,
                            states_hist=None, obs_hist=None, len_single_display_vec=None):
        if memo is not None:
            result = memo.get((dataset_number, state))
            if result is not None:
                obs, disp_tpl, dfs = result


                if states_hist is not None:
                    if len(obs_hist) > 0 and cfg.stack_obs_num > 1:
                        obs_to_stack = obs_hist[-1][-len_single_display_vec * (cfg.stack_obs_num - 1):]
                    else:
                        obs_to_stack = np.empty(shape=0, dtype=np.float32)
                    obs = np.concatenate([obs_to_stack, obs])
                    if len(obs) < len_single_display_vec * cfg.stack_obs_num:
                        obs = np.concatenate(
                            [np.zeros(len_single_display_vec * (
                                    cfg.stack_obs_num - (len(obs) // len_single_display_vec)),
                                      dtype=np.float32), obs]
                        )


                if cfg.obs_with_step_num and step_number is not None:
                    step_vector = np.zeros(cfg.MAX_NUM_OF_STEPS, np.float32)
                    if step_number != 0:
                        step_vector[step_number - 1] = 1
                    obs = np.concatenate([step_vector, obs])
                result = obs, disp_tpl, dfs
                return result

        fdf, gdf, adf = self.get_raw_data(dataset_df, state)
        data_layer = self.calc_data_layer(fdf)
        gran_layer = self.calc_gran_layer(gdf, adf)

        vlist = []
        for d in data_layer.values():
            vlist += [d["unique"], d["nulls"], d["entropy"]]


        if gran_layer is None:


            vlist += [-1 for _ in self.env_dataset_prop.KEYS] + [0, 0, 0]
        else:
            for k in self.env_dataset_prop.KEYS:
                if k in gran_layer['agg_attrs'].keys():
                    vlist.append(gran_layer['agg_attrs'][k])
                elif k in gran_layer['group_attrs']:
                    vlist.append(2)
                else:
                    vlist.append(-1)

            vlist += [gran_layer['inverse_ngroups'], gran_layer['inverse_size_mean'], gran_layer['site_std']]

        state_obs = np.array(vlist, dtype=np.float32)
        disp_tpl = DisplayTuple(data_layer=data_layer, granularity_layer=gran_layer)

        obs = state_obs

        if states_hist is not None:
            if len(obs_hist) > 0 and cfg.stack_obs_num > 1:
                obs_to_stack = obs_hist[-1][-len_single_display_vec * (cfg.stack_obs_num - 1):]
            else:
                obs_to_stack = np.empty(shape=0, dtype=np.float32)
            obs = np.concatenate([obs_to_stack, obs])
            if len(obs) < len_single_display_vec * cfg.stack_obs_num:
                obs = np.concatenate(
                    [np.zeros(len_single_display_vec * (
                            cfg.stack_obs_num - (len(obs) // len_single_display_vec)),
                              dtype=np.float32), obs]
                )


        if cfg.obs_with_step_num and step_number is not None:
            step_vector = np.zeros(cfg.MAX_NUM_OF_STEPS, np.float32)
            if step_number != 0:
                step_vector[step_number - 1] = 1
            obs = np.concatenate([step_vector, state_obs])
        result = obs, disp_tpl, (fdf, adf)

        if memo is not None:
            memo[(dataset_number, state)] = state_obs, disp_tpl, (fdf, adf)

        return result


class BasicEnvProp(EnvProp):

    def __init__(self,
                 OPERATOR_TYPE_LOOKUP,
                 INT_OPERATOR_MAP_ATENA,
                 AGG_MAP_ATENA,
                 env_dataset_prop,
                 ):
        super().__init__(
            OPERATOR_TYPE_LOOKUP,
            INT_OPERATOR_MAP_ATENA,
            AGG_MAP_ATENA,
            env_dataset_prop,
        )

    def get_softmax_layer_size(self):
        back = 1

        filter_terms_num = self.DISCRETE_FILTER_TERM_BINS_NUM

        filter_ = (
                len(self.env_dataset_prop.FILTER_COLS) *
                len(self.INT_OPERATOR_MAP_ATENA) // 3 *
                filter_terms_num
        )
        group = len(self.env_dataset_prop.GROUP_COLS)

        return back + filter_ + group

    def get_parametric_segments(self):

        back = tuple()

        filter_terms_num = self.DISCRETE_FILTER_TERM_BINS_NUM
        filter_ = tuple([len(self.env_dataset_prop.FILTER_COLS), len(self.INT_OPERATOR_MAP_ATENA) // 3, filter_terms_num])
        group = tuple([len(self.env_dataset_prop.GROUP_COLS)])

        return back, filter_, group

    def parametric_softmax_idx_to_discrete_action(self, idx, segs):

        cur_idx = 0
        action_type_num = 0
        result = deque()

        for action_num, seg in enumerate(segs):
            seg_size = self.get_seg_size(seg)
            if cur_idx + seg_size > idx:
                in_seg_idx = idx - cur_idx
                action_type_num = action_num
                for i in range(len(seg) - 1, -1, -1):
                    result.appendleft(in_seg_idx % seg[i])
                    in_seg_idx //= seg[i]
                break
            else:
                cur_idx += seg_size
        else:
            raise ValueError("invalid index for the given segments")

        return action_type_num, tuple(result)
