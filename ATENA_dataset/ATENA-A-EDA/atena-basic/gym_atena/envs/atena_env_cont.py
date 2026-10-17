import json
import math
import logging
import random
from copy import deepcopy
from functools import lru_cache

import numpy as np

from cachetools import LRUCache

import gym
from gym import spaces

from scipy.stats import entropy
import scipy

from gym_atena.lib.tokenization import tokenize_column, get_nearest_neighbor_token, gep

from gym_atena.reactida.utils.distance import display_distance
import gym_atena.lib.helpers as ATENAUtils
from gym_atena.lib.helpers import (
    normalized_sigmoid_fkt,
    get_aggregate_attributes,
    empty_env_state,
    AggregationTuple,
    FilteringTuple,
)

from arguments import ArchName, FilterTermsBinsSizes
from Utilities.Collections.Counter_Without_Nans import CounterWithoutNanKeys
import Utilities.Configuration.config as cfg

logger = logging.getLogger(__name__)


class ATENAEnvCont(gym.Env):


    LOG_INTERVAL = 100

    NUM_OF_EPISODES = 0

    metadata = {
        'render.modes': ['human'],
    }


    STATE_DF_HISTORY = None


    COL_TOKENIZATION_HISTORY = None


    STATES_DISP_DISTANCE_HISTORY = None


    arch = ArchName(cfg.arch)


    len_single_display_vec = None


    static_env = None

    def __init__(self, max_steps=cfg.MAX_NUM_OF_STEPS, ret_df=False, gradual_training=False):

        self.env_prop = gep.update_global_env_prop_from_cfg()


        if self._is_caches_reset_needed():
            self.reset_caches()


        self.gradual_training = gradual_training
        self.max_steps = max_steps
        self.ret_df = ret_df
        self.env_dataset_prop = self.env_prop.env_dataset_prop


        self.repo = self.env_dataset_prop.repo


        self.action_space = spaces.Box(low=np.zeros(6) - 0.49,
                                       high=np.array([self.env_prop.ACTION_TYPES_NO,self.env_dataset_prop.COLS_NO,
                                                     self.env_prop.FILTER_OPS,self.env_dataset_prop.FILTER_TERMS_NO,
                                                     self.env_dataset_prop.AGG_COLS_NO, self.env_prop.AGG_FUNCS_NO]) - 0.51, dtype='float32')


        z = np.zeros(len(self.env_dataset_prop.KEYS) * 3)

        z2 = np.full(len(self.env_dataset_prop.KEYS), -1)
        z3 = np.zeros(3)

        low = np.tile(np.concatenate([z, z2, z3]), cfg.stack_obs_num)
        high = np.tile(np.concatenate([np.ones(len(z)), np.full(len(self.env_dataset_prop.KEYS), 2), np.ones(3)]), cfg.stack_obs_num)

        if cfg.obs_with_step_num:
            z_step_number = np.zeros(self.max_steps)
            low = np.concatenate([z_step_number, low])
            high = np.concatenate([np.ones(len(z_step_number)), high])


        self.observation_space = spaces.Box(low, high, dtype='float32')

        ATENAEnvCont.len_single_display_vec = len(z) + len(z2) + len(z3)


        if cfg.dataset_number is not None:
            print(self.repo.file_list[cfg.dataset_number])

        self._log = True

    @classmethod
    def reset_caches(cls):
        cls.STATE_DF_HISTORY = None
        if cfg.cache_dfs_size > -1:
            cls.STATE_DF_HISTORY = LRUCache(maxsize=cfg.cache_dfs_size)

        cls.COL_TOKENIZATION_HISTORY = None
        if cfg.cache_tokenization_size > -1:
            cls.COL_TOKENIZATION_HISTORY = LRUCache(maxsize=cfg.cache_tokenization_size)

        cls.STATES_DISP_DISTANCE_HISTORY = None
        if cfg.cache_tokenization_size > -1:
            cls.STATES_DISP_DISTANCE_HISTORY = LRUCache(maxsize=cfg.cache_distances_size)

    def _is_caches_reset_needed(self):
        if (self.STATES_DISP_DISTANCE_HISTORY is None
                and self.STATE_DF_HISTORY is None
                and self.COL_TOKENIZATION_HISTORY is None):
            return True
        return False


    def translate_action(self, act_vector, filter_by_field=True, filter_term=None):

        if type(act_vector) is not list:
            act_vector = self.cont2dis(act_vector)
        rtype = self.env_prop.OPERATOR_TYPE_LOOKUP.get(act_vector[0])
        if rtype == "back":
            return "Back"
        elif rtype == "filter":
            col = self.data.columns[act_vector[1]]
            cond = self.env_prop.INT_OPERATOR_MAP_ATENA.get(act_vector[2])
            if not cond:
                op_num = act_vector[2]
                if op_num in [6, 7, 8]:
                    cond = 'str.contains'
                elif op_num == 7:
                    cond = 'str.stratswith'
                elif op_num == 8:
                    cond = 'str.endswith'
            if not filter_by_field:
                term = self.env_dataset_prop.FILTER_LIST[act_vector[3]]
            else:

                term = filter_term
            return "Filter on Column '%s', using condition '%s', with term '%s'" % (col, str(cond), term)
        elif rtype == "group":
            col = self.data.columns[act_vector[1]]
            agg_col = self.data.columns[act_vector[4]]
            agg_func = self.env_prop.AGG_MAP_ATENA.get(act_vector[5])
            return "Group on Column '%s' and aggregate with '%s' on the column '%s'" % (col, str(agg_func), agg_col)
        else:
            raise NotImplementedError

    @lru_cache(maxsize=2048)
    def get_exponential_filter_term_bins(self, num_of_rows, num_of_bins):

        assert num_of_bins >= 1
        if num_of_rows == 0:
            return [0.0] * num_of_bins + [1.0]
        B_minus1 = num_of_bins - 1
        x = (num_of_rows ** (1 / B_minus1))
        single_row_contribution = 1 / num_of_rows
        bins = [0] + [round(x ** i) / num_of_rows for i in range(num_of_bins)]
        for i in range(1, len(bins)):
            if bins[i] <= bins[i - 1]:
                bins[i] = bins[i - 1] + single_row_contribution
        return bins

    @staticmethod
    def _param_softmax_idx_to_action_helper(idx):
        raise NotImplementedError

    def param_softmax_idx_to_action(self, idx):

        result = np.zeros(6, dtype=np.float32)
        (action_type, parameters) = self.env_prop.MAP_PARAMETRIC_SOFMAX_IDX_TO_DISCRETE_ACTION[idx]

        result[0] = action_type
        action_type_string = self.env_prop.OPERATOR_TYPE_LOOKUP[action_type]
        if action_type_string == "back":
            pass
        elif action_type_string == "filter":
            result[1] = parameters[0]

            result[2] = (parameters[1] + 1) * 3 - 1

            filter_terms_bin_sizes = FilterTermsBinsSizes(cfg.bins_sizes)
            if filter_terms_bin_sizes is FilterTermsBinsSizes.EQUAL_WIDTH:
                bin_size = 1 / (self.env_prop.DISCRETE_FILTER_TERM_BINS_NUM - 1)
                result[3] = parameters[2] * bin_size - 0.5 + random.uniform(-bin_size / 2, bin_size / 2)
            elif filter_terms_bin_sizes is FilterTermsBinsSizes.CUSTOM_WIDTH:
                lower_bin_edge = self.env_prop.bins[parameters[2]]
                upper_bin_edge = self.env_prop.bins[parameters[2] + 1]
                bin_size = upper_bin_edge - lower_bin_edge
                result[3] = lower_bin_edge - 0.5 + random.uniform(0, bin_size)
            elif filter_terms_bin_sizes is FilterTermsBinsSizes.EXPONENTIAL:
                num_of_rows = self.num_of_fdf_rows_hist[-1]
                bins = self.get_exponential_filter_term_bins(num_of_rows, cfg.exponential_sizes_num_of_bins)
                lower_bin_edge = bins[parameters[2]]
                upper_bin_edge = bins[parameters[2] + 1]
                bin_size = upper_bin_edge - lower_bin_edge
                result[3] = lower_bin_edge - 0.5 + random.uniform(0, bin_size)
            else:
                raise NotImplementedError

        elif action_type_string == "group":
            result[1] = parameters[0]
        else:
            raise ValueError("action_type should refer to back filter or group")

        return result

    @staticmethod
    def static_param_softmax_idx_to_action(idx):

        result, _, _ = ATENAEnvCont._param_softmax_idx_to_action_helper(idx)
        return result


    @staticmethod
    def cont2dis(c_vector):

        return list(np.array(np.round(c_vector), dtype=np.int))

    @staticmethod
    def _is_empty_display(obs):
        start_of_cur_display_suv_vec = len(obs) - ATENAEnvCont.len_single_display_vec
        return np.array_equal(
            obs[start_of_cur_display_suv_vec:3 + start_of_cur_display_suv_vec],
            [0, 1, 0])

    @staticmethod
    def _is_empty_groupings(obs):
        return np.array_equal(obs[-3:], [0, 0, 0])

    def reset(self, dataset_number=None):


        random.seed()
        scipy.random.seed()

        self.NUM_OF_EPISODES += 1
        self.step_num = 0
        if self.gradual_training:
            self.max_steps = random.randint(2, max(3, int(self.NUM_OF_EPISODES / 2500)))

        self._log = True if self.NUM_OF_EPISODES % self.LOG_INTERVAL == 0 else False


        if cfg.dataset_number is not None:
            dataset_number = cfg.dataset_number
        elif dataset_number is None:
            dataset_number = np.random.randint(len(self.repo.data))
        self.dataset_number = dataset_number
        if self._log:
            logger.info(f"Dataset number chosen is {dataset_number}, dataset name is {self.repo.file_list[dataset_number]}")

        self.data = self.repo.data[dataset_number][self.env_dataset_prop.KEYS]


        empty_state = empty_env_state

        self.history = [empty_state]

        self.states_hisotry = [empty_state]

        self.obs_hist = []

        self.obs_hist_all = []


        obs, disp, _ = self.env_prop.calc_display_vector(self.data,
                                           empty_state,
                                           memo=self.STATE_DF_HISTORY,
                                           dataset_number=self.dataset_number,
                                           step_number=self.step_num,
                                           states_hist=self.history,
                                           obs_hist=self.obs_hist,
                                           len_single_display_vec=self.len_single_display_vec
                                           )


        self.dhist = [disp]
        self.ahist = []
        self.obs_hist = [obs]
        self.obs_hist_all = [obs]
        self.filter_terms_hist = []
        self.num_of_rows_hist = [len(self.data)]
        self.num_of_fdf_rows_hist = [len(self.data)]


        self.num_of_immediate_action_rows_lst = [None]


        self.in_the_middle_of_empty_grouping = False
        self.in_the_middle_of_empty_grouping_steps = 0

        assert self.observation_space.contains(obs)
        return obs

    @property
    def arch(self):
        return ArchName(cfg.arch)

    def compute_interestingness_reward(self, dfs, state, last_action_type):
        kl_distance = compaction_gain = 0
        if self.env_prop.OPERATOR_TYPE_LOOKUP[last_action_type] == "filter":
            kl_distance = self.compute_interestingness_kl_divergence(dfs, state)
            kl_distance = kl_distance * cfg.kl_coeff
        elif self.env_prop.OPERATOR_TYPE_LOOKUP[last_action_type] == 'group':
            compaction_gain = self.compute_interestingness_compaction_gain(dfs, state)
            compaction_gain = compaction_gain * cfg.compaction_coeff

        return StepReward(kl_distance=kl_distance, compaction_gain=compaction_gain)


    def compute_interestingness_kl_divergence(self, dfs, state):

        kl_distances = []
        df_dt, is_grouping = self.get_filtered_only_or_grouped_data_frame(dfs)
        df_D = self.get_previous_df(past_steps=2)

        if is_grouping:


            aggeregate_attributes_list = get_aggregate_attributes(state)
            kl_attrs = aggeregate_attributes_list
            assert len(set(kl_attrs).difference(set(self.env_dataset_prop.AGG_KEYS))) == 0, "Only primary key is supported now, use gdf to support more aggregated attributes"
        else:

            KL_DIV_EPSILON = 2 / len(df_D) * 0.1
            kl_attrs = self.data.columns


        for attr in kl_attrs:


            attr_value_count1 = CounterWithoutNanKeys(df_D[attr].values)
            attr_value_count2 = CounterWithoutNanKeys(df_dt[attr].values)

            if is_grouping:
                KL_DIV_EPSILON = 2 / sum(attr_value_count1.elements()) * 0.1


            pk1 = []
            pk2 = []
            for key in attr_value_count1:
                pk1.append(attr_value_count1[key])
                if key in attr_value_count2:
                    pk2.append(attr_value_count2[key])
                else:
                    pk2.append(KL_DIV_EPSILON)


            for key in attr_value_count2:
                if key not in attr_value_count1:
                    pk2.append(attr_value_count2[key])
                    pk1.append(KL_DIV_EPSILON)


            attr_kl_div = entropy(pk1, pk2)
            kl_distances.append(attr_kl_div)

        return 1 / (1 + math.exp(-(max(kl_distances) / 2 - 3)))

    @staticmethod
    def get_filtered_only_or_grouped_data_frame(dfs):

        is_grouping = True if dfs[1] is not None else False
        if is_grouping:
            df_dt = dfs[1]
        else:
            df_dt = dfs[0]
        return df_dt, is_grouping


    def reward(self, obs, dfs, state, no_history_for_back):

        reward_info = StepReward()


        self.step_num += 1
        done = self.step_num >= self.max_steps


        last_action_type = self.ahist[-1][0]


        if no_history_for_back:
            r = -1.0 * cfg.humanity_coeff
            reward_info.back = r

            reward_info.is_back = True


        elif self.env_prop.OPERATOR_TYPE_LOOKUP[last_action_type] == 'back':
            r = 0
            is_data_driven = False

            r_humanity = 0

            if cfg.use_humans_reward:
                r = r_humanity

            reward_info.back = r
            reward_info.is_back = True

            if self.in_the_middle_of_empty_grouping:
                self.in_the_middle_of_empty_grouping_steps -= 1
            if self.in_the_middle_of_empty_grouping_steps == 0:
                self.in_the_middle_of_empty_grouping = False


        elif self._is_empty_display(obs):
            r = -1.0 * cfg.humanity_coeff

            reward_info.empty_display = r
            reward_info.interestingness = r


        elif (state['grouping'] and (dfs[1] is None or self._is_empty_groupings(obs)) or
              self.in_the_middle_of_empty_grouping):
            r = -1.0 * cfg.humanity_coeff
            self.in_the_middle_of_empty_grouping = True
            self.in_the_middle_of_empty_grouping_steps += 1

            reward_info.empty_groupings = r
            reward_info.interestingness = r


        else:

            r_div = self._compute_diversity_reward(last_action_type, reward_info)


            r_inter = 0
            r_humanity = 0
            if not reward_info.is_same_display_seen_already:

                interestingness_rewards = self.compute_interestingness_reward(dfs, state, last_action_type)
                reward_info.kl_distance = interestingness_rewards.kl_distance
                reward_info.compaction_gain = interestingness_rewards.compaction_gain
                r_inter = max(interestingness_rewards.values())
                reward_info.interestingness = r_inter


                reward_info.humanity = 0

            if cfg.no_diversity and not reward_info.is_same_display_seen_already:
                r_div = 0

            if cfg.no_interestingness:
                r_inter = 0

            if not cfg.use_humans_reward:
                r_humanity = 0

            r = r_div + r_inter + r_humanity
        return done, r, reward_info

    def _compute_diversity_reward(self, last_action_type, reward_info):
        last_display = self.dhist[-1]
        last_state = self.states_hisotry[-1]

        sim_vec = []


        for i, d in enumerate(self.dhist[:-1]):
            i_state = self.states_hisotry[i]
            state1 = i_state
            state2 = last_state
            if str(state1) > str(state2):
                state1, state2 = state2, state1
            display_distance_result_obj = None
            if self.STATES_DISP_DISTANCE_HISTORY is not None:
                display_distance_result_obj = self.STATES_DISP_DISTANCE_HISTORY.get(
                    (self.dataset_number, state1, state2))
            if display_distance_result_obj is None:
                display_distance_result_obj = display_distance(d, last_display)
                if self.STATES_DISP_DISTANCE_HISTORY is not None:
                    self.STATES_DISP_DISTANCE_HISTORY[
                        (self.dataset_number, state1, state2)] = display_distance_result_obj
            dist = display_distance_result_obj.display_distance


            if (dist == 0 or
                    (display_distance_result_obj.data_distance == 0 and
                     self.env_prop.OPERATOR_TYPE_LOOKUP[last_action_type] == 'filter'

                    )
            ):
                r = -1.0 * cfg.humanity_coeff

                reward_info.same_display_seen_already = r
                reward_info.diversity = r
                break

            else:
                sim_vec.append(dist)
        else:

            r = min(sim_vec) * cfg.diversity_coeff
            reward_info.diversity = r

        return r

    def compute_interestingness_compaction_gain(self, dfs, state):

        df_dt, is_grouping = self.get_filtered_only_or_grouped_data_frame(dfs)
        denominator_epsilon = 0.00001
        R = len(self.data)
        R_tag = len(df_dt)
        C = len(state["grouping"])
        if is_grouping:


            R_tag = len(dfs[0])
            G = len(df_dt)

            if G == 1:
                return -1


            compact_display_score = normalized_sigmoid_fkt(0.5, 17,
                                                           1 - 1 / math.log(8 + G * C + denominator_epsilon, 8))

            compact_data_score = 1 - normalized_sigmoid_fkt(0.5, 17,
                                                            1 - 1 / math.log(7 + R_tag + denominator_epsilon, 7))
            return compact_display_score * compact_data_score
        return 1 - math.log(R_tag + denominator_epsilon, R)

    def step(self, action, compressed=False, filter_by_field=True, continuous_filter_term=True, filter_term=None):


        prev_action = action
        action, filter_by_field = self.action_to_vec(action, compressed, continuous_filter_term, filter_by_field)
        operator_type = self.env_prop.OPERATOR_TYPE_LOOKUP.get(action[0])
        col = self.env_dataset_prop.KEYS[action[1]]
        no_history_for_back = False


        if operator_type == 'back':

            if len(self.history) > 1:
                self.obs_hist.pop()
                self.history.pop()
                new_state = self.history[-1]
            else:
                new_state = empty_env_state
                no_history_for_back = True

        elif operator_type == 'filter':

            condition = action[2]
            if filter_term is not None:
                pass
            elif not filter_by_field:
                filter_term = self.env_dataset_prop.FILTER_LIST[action[3]]
            else:

                filter_term = self.compute_nearest_neighbor_filter_term(action, col)

            filt_tpl = FilteringTuple(field=col, term=filter_term, condition=condition)

            new_state = self.history[-1]
            new_state = new_state.append_filtering(filt_tpl)
            self.history.append(new_state)

        elif operator_type == 'group':

            new_state = self.history[-1]
            if col not in new_state["grouping"]:
                new_state = new_state.append_grouping(col)
            agg_tpl = AggregationTuple(field=self.env_dataset_prop.AGG_KEYS[action[4]], type=action[5])
            if agg_tpl not in new_state["aggregations"]:
                new_state = new_state.append_aggregations(agg_tpl)
            self.history.append(new_state)
        else:
            raise Exception("unknown operator type: " + operator_type)

        self.states_hisotry.append(new_state)


        obs, disp, dfs = self.env_prop.calc_display_vector(self.data,
                                             new_state,
                                             memo=self.STATE_DF_HISTORY,
                                             dataset_number=self.dataset_number,
                                             step_number=self.step_num,
                                             states_hist=self.history,
                                             obs_hist=self.obs_hist,
                                             len_single_display_vec=self.len_single_display_vec
                                             )


        self.dhist.append(disp)
        self.ahist.append(action)
        self.obs_hist_all.append(obs)
        if operator_type != 'back':
            self.obs_hist.append(obs)

        self.filter_terms_hist.append(filter_term)
        self.num_of_rows_hist.append(len(self.get_previous_df()))
        self.num_of_fdf_rows_hist.append(len(self.get_previous_fdf()))
        self.num_of_immediate_action_rows_lst.append(self.get_num_of_immediate_action_rows_after_filter())


        done, reward, reward_info = self.reward(obs, dfs, new_state, no_history_for_back)


        assert obs.dtype == np.float32, 'obs.dtype must be np.float32'

        if done and self._log:
            logger.info('actions:%s' % str(self.ahist))


        if not self.ret_df:
            dfs = None

        return obs, reward, done, {"raw_action": action,
                                   "action": self.translate_action(action, filter_by_field, filter_term),
                                   "raw_display": dfs,
                                   "reward_info": reward_info,
                                   "state": new_state,
                                   "filter_term": filter_term,
                                   }

    def action_to_vec(self, action, compressed=False, continuous_filter_term=True, filter_by_field=True):
        if self.arch is ArchName.FF_PARAM_SOFTMAX or self.arch is ArchName.FF_SOFTMAX:
            compressed = False
            action = self.param_softmax_idx_to_action(action)
        if compressed:


            action = self.env_prop.compressed2full_range(action, continuous_filter_term)
        action_filter_term = action[3]
        action = self.cont2dis(action)
        if continuous_filter_term:
            action[3] = action_filter_term + 0.5
        return action, filter_by_field

    def compute_nearest_neighbor_filter_term(self, action, col):
        prev_state = self.history[-1]
        prev_state_without_group_and_agg = prev_state.reset_grouping_and_aggregations()
        if self.COL_TOKENIZATION_HISTORY is None or (
                (self.dataset_number, prev_state_without_group_and_agg,
                 col) not in self.COL_TOKENIZATION_HISTORY):
            prev_fdf = self.get_previous_fdf()
            sorted_by_freq_token_frequency_pairs, frequencies = tokenize_column(prev_fdf, col)


            if self.COL_TOKENIZATION_HISTORY is not None:
                self.COL_TOKENIZATION_HISTORY[(self.dataset_number,
                                               prev_state_without_group_and_agg,
                                               col)] = (sorted_by_freq_token_frequency_pairs, frequencies)
        else:
            sorted_by_freq_token_frequency_pairs, frequencies = self.COL_TOKENIZATION_HISTORY[
                (self.dataset_number,
                 prev_state_without_group_and_agg,
                 col)]
        filter_term = get_nearest_neighbor_token(sorted_by_freq_token_frequency_pairs, frequencies, action[3])
        return filter_term

    def get_num_of_immediate_action_rows_after_filter(self):
        if len(self.ahist) < 2:
            return None
        cur_action = self.ahist[-1]
        prev_action = self.ahist[-2]
        cur_action_type_str = self.env_prop.OPERATOR_TYPE_LOOKUP[cur_action[0]]
        prev_action_type_str = self.env_prop.OPERATOR_TYPE_LOOKUP[prev_action[0]]

        if prev_action_type_str == 'filter':
            if cur_action_type_str == 'back':
                return None
            else:

                state_before_filter = self.states_hisotry[-3]
                if cur_action_type_str == 'group':
                    grouped_column = self.env_dataset_prop.GROUP_COLS[cur_action[1]]
                    if grouped_column not in state_before_filter["grouping"]:
                        state_before_filter = state_before_filter.append_grouping(grouped_column)
                    agg_tpl = AggregationTuple(field=self.env_dataset_prop.AGG_KEYS[cur_action[4]], type=cur_action[5])
                    if agg_tpl not in state_before_filter["aggregations"]:
                        state_before_filter = state_before_filter.append_aggregations(agg_tpl)
                elif cur_action_type_str == 'filter':
                    filtered_column = self.env_dataset_prop.FILTER_COLS[cur_action[1]]
                    filt_tpl = FilteringTuple(field=filtered_column, term=self.filter_terms_hist[-1],
                                              condition=cur_action[2])
                    state_before_filter = state_before_filter.append_filtering(filt_tpl)

                dfs = self.env_prop.get_state_dfs(self.data,
                                    state_before_filter,
                                    memo=self.STATE_DF_HISTORY,
                                    dataset_number=self.dataset_number,
                                    )

                df_dt, is_grouping = self.get_filtered_only_or_grouped_data_frame(dfs)
                return len(df_dt)

        else:
            return None


    def get_previous_fdf(self, past_steps=1):
        prev_state = self.states_hisotry[-1 * past_steps]

        dfs = self.env_prop.get_state_dfs(self.data,
                            prev_state,
                            memo=self.STATE_DF_HISTORY,
                            dataset_number=self.dataset_number,
                            )

        return dfs[0]

    def get_previous_df(self, past_steps=1):

        prev_state = self.states_hisotry[-1 * past_steps]
        dfs = self.env_prop.get_state_dfs(self.data,
                            prev_state,
                            memo=self.STATE_DF_HISTORY,
                            dataset_number=self.dataset_number,
                            )

        df_dt, is_grouping = self.get_filtered_only_or_grouped_data_frame(dfs)

        return df_dt

    def render(self, mode='human', close=False):
        if close:
            return None
        self.ret_df = True

        return None

    @classmethod
    def get_static_env(cls, max_steps):

        if cls.static_env is None:
            cls.static_env = cls(max_steps=max_steps)
        else:
            cls.static_env.max_steps = max_steps
        return cls.static_env

    GET_SESSIONS_HISTS_CACHE = LRUCache(maxsize=300)

    @classmethod
    def get_sessions_hists(cls, actions_lst,
                           dataset_number,
                           compressed=False,
                           filter_by_field=True,
                           continuous_filter_term=True,
                           filter_terms_lst=None
                           ):


        prev_arch = cfg.arch
        cfg.arch = ArchName.FF_GAUSSIAN.value

        actions_tuple = ATENAUtils.lst_of_actions_to_tuple(actions_lst)
        if (dataset_number, actions_tuple) in cls.GET_SESSIONS_HISTS_CACHE:
            return cls.GET_SESSIONS_HISTS_CACHE[(dataset_number, actions_tuple)]

        if filter_terms_lst is not None:
            assert len(actions_lst) == len(filter_terms_lst)

        env = cls.get_static_env(max_steps=len(actions_lst))
        info_hist = []
        env.render()
        env.reset(dataset_number)
        for i, a in enumerate(actions_lst):
            s_, _, done, info = env.step(a,
                                         compressed=compressed,
                                         filter_by_field=filter_by_field,
                                         continuous_filter_term=continuous_filter_term,
                                         filter_term=None if filter_terms_lst is None else filter_terms_lst[i]
                                         )
            s = s_
            info_hist.append(info)
            if done:
                break
        dhist = env.dhist
        ahist = env.ahist
        cls.GET_SESSIONS_HISTS_CACHE[(dataset_number, actions_tuple)] = (dhist, ahist, info_hist)


        cfg.arch = prev_arch
        return dhist, ahist, info_hist

    @classmethod
    def get_greedy_max_reward_actions_lst(cls,
                                          dataset_number,
                                          episode_length,
                                          kl_coeff,
                                          compaction_coeff,
                                          diversity_coeff,
                                          humanity_coeff,
                                          verbose=False):

        cfg.kl_coeff = kl_coeff
        cfg.compaction_coeff = compaction_coeff
        cfg.diversity_coeff = diversity_coeff
        cfg.humanity_coeff = humanity_coeff

        cur_env = cls(max_steps=episode_length)
        cur_env.render()
        cur_env.reset(dataset_number)
        cur_env.arch = ArchName.FF_PARAM_SOFTMAX

        actions_lst = []
        total_reward = 0
        for step in range(episode_length):
            max_reward = -math.inf
            max_action = None
            max_env = None

            for act_idx in cur_env.env_prop.MAP_PARAMETRIC_SOFMAX_IDX_TO_DISCRETE_ACTION.keys():
                next_env = deepcopy(cur_env)
                obs, reward, done, info = next_env.step(act_idx)
                action_vec = info["raw_action"]
                action_vec[3] -= 0.5
                if reward > max_reward:
                    max_reward = reward
                    max_action = action_vec
                    max_env = next_env
            if verbose:
                print(max_reward)
            cur_env = max_env
            actions_lst.append(max_action)
            total_reward += max_reward
        if verbose:
            print(actions_lst)
            print(total_reward)
        return actions_lst, total_reward

    @classmethod
    def debug_actions(cls, actions_lst, dataset_number=None, compressed=False, filter_by_field=False,
                      continuous_filter_term=False, displays=False,
                      kl_coeff=cfg.kl_coeff,
                      compaction_coeff=cfg.compaction_coeff,
                      diversity_coeff=cfg.diversity_coeff,
                      humanity_coeff=cfg.humanity_coeff,
                      ):

        cfg.kl_coeff = kl_coeff
        cfg.compaction_coeff = compaction_coeff
        cfg.diversity_coeff = diversity_coeff
        cfg.humanity_coeff = humanity_coeff
        cfg.analysis_mode = True

        env = cls(max_steps=len(actions_lst))
        info_hist = []
        env.render()
        env.reset()
        for i in range(1):

            if isinstance(env, ATENAEnvCont):
                s = env.reset(dataset_number)
            elif isinstance(env, gym.wrappers.Monitor):
                s = env.env.env.reset(dataset_number)
            else:
                s = env.env.reset(dataset_number)
            r_sum = 0
            for a in actions_lst:
                print(a)
                s_, r, done, info = env.step(a,
                                             compressed=compressed,
                                             filter_by_field=filter_by_field,
                                             continuous_filter_term=continuous_filter_term)
                print(str(info["action"]))
                print("reward:" + str(r))
                print(str(info["reward_info"]))
                print()
                info_hist.append((info, r))
                s = s_
                r_sum += r
                print("")
                if displays:
                    f_df, a_df = info["raw_display"]
                    if a_df is not None:
                        print(a_df)
                    else:
                        print(f_df)
                print("---------------------------------------------------")
                if done:
                    break
        print(f"Total reward: {r_sum}")
        return info_hist, r_sum


class StepReward(object):


    def __init__(self,
                 empty_display=0,
                 empty_groupings=0,
                 same_display_seen_already=0,
                 back=0,
                 diversity=0,
                 interestingness=0,
                 kl_distance=0,
                 compaction_gain=0,
                 humanity=0
                 ):
        self.empty_display = empty_display
        self.empty_groupings = empty_groupings
        self.same_display_seen_already = same_display_seen_already
        self.back = back
        self.diversity = diversity
        self.interestingness = interestingness
        self.kl_distance = kl_distance
        self.compaction_gain = compaction_gain
        self.humanity = humanity

        self.is_back = False
        self.is_data_driven = False

    @property
    def is_same_display_seen_already(self):
        return self.same_display_seen_already < 0

    def items(self):

        result = deepcopy(self.__dict__)
        result.pop('is_back')
        result.pop('is_data_driven')
        return result.items()

    def values(self):

        result = deepcopy(self.__dict__)
        result.pop('is_back')
        result.pop('is_data_driven')
        return result.values()

    def __repr__(self):
        return self.items().__repr__()


if __name__ == '__main__':

    actions_lst = [
        [2, 1, 1, 0.3221167325973511, 0, 0],
        [2, 6, 4, 0.6, 0, 0],
        [0, 2, 8, 0.21705351769924164, 0, 0],
        [2, 2, 8, -0.49803635857105255, 0, 0],
        [2, 6, 7, -0.46546758365631, 0, 0],
        [1, 6, 4, 0.38086163997650146, 0, 0],
        [0, 3, 1, -0.40, 0, 0],
        [0, 6, 8, 0.444186806678772, 0, 0],
        [0, 6, 8, 0.444186806678772, 0, 0],
        [2, 5, 4, 0.6415030360221863, 0, 0],
        [2, 6, 4, 0.5431699156761169, 0, 0],
        [2, 2, 4, 0.8115951418876648, 0, 0],
    ]
    actions_lst = [np.array(act) for act in actions_lst]
    ATENAEnvCont.debug_actions(actions_lst, 3, compressed=False, filter_by_field=True, continuous_filter_term=True,
                               displays=True,
                               kl_coeff=2.5,
                               compaction_coeff=2.9,
                               diversity_coeff=6.0,
                               humanity_coeff=4.8,


                               )


    actions_lst = [
        [2, 6, 2, 0.3221167325973511, 0, 0],
        [2, 5, 1, -0.445, 0, 0],
    ]

    actions_lst2 = [
        [2, 5, 1, -0.445, 0, 0],
        [2, 6, 2, 0.3221167325973511, 0, 0],
    ]
