import sys
from collections import namedtuple, defaultdict
from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from gym_atena.envs.atena_env_cont import ATENAEnvCont
import gym_atena.lib.helpers as ATENAUtils
from gym_atena.lib.helpers import (
    ActionVectorEntry,
)

import Utilities.Configuration.config as cfg
from arguments import ArchName
from Utilities.Utility_Functions import initialize_agent_and_env
from models.Greedy.greedy_agent import GreedyAgent

import gym_atena.global_env_prop as gep

StepInfo = namedtuple('StepInfo', 'continuous_action reward info')

model_dir_path = None
command_args = None
results_path = None


CURRENT_PLOT_NUM = 1


PLOT = False


SAVE = True


def plot_and_save(dataset_number=None, path=None):

    global CURRENT_PLOT_NUM


    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    if path is None:
        if dataset_number is not None and dataset_number != "all":
            path = '%s/d%d/%d.png' % (results_path, dataset_number, CURRENT_PLOT_NUM)
        else:
            path = '%s/all/%d.png' % (results_path, CURRENT_PLOT_NUM)
        CURRENT_PLOT_NUM += 1
    fig = plt.gcf()

    if PLOT:
        plt.show()
    else:

        plt.show(block=False)
        plt.pause(0.3)
        plt.close()


    if SAVE:
        fig.savefig(path, dpi=100)


class AgentAnalyzer(object):


    def __init__(self, model_dir_path=model_dir_path, command_args=command_args, n_sessions_per_dataset=250, seed=0,
                 is_greedy=False):


        cfg.analysis_mode = True
        self.is_greedy = is_greedy

        if is_greedy:
            assert isinstance(command_args, dict)
            cfg.schema = command_args["schema"]
            self.env = ATENAEnvCont(max_steps=command_args["max_steps"])
            self.agent = GreedyAgent(
                kl_coeff=command_args["kl_coeff"],
                compaction_coeff=command_args["compaction_coeff"],
                diversity_coeff=command_args["diversity_coeff"],
                humanity_coeff=command_args["humanity_coeff"],
            )
            cfg.MAX_NUM_OF_STEPS = command_args["max_steps"]
            self.n_sessions_per_dataset = 1
            cfg.arch = ArchName.GREEDY.value
            path = '%s/config.txt' % results_path
            with open(path, "w") as config_file:
                config_text = "\n".join([f"{key}: {val}" for key, val in command_args.items()])
                config_file.write(config_text)

        else:
            command = "train.py --env ATENAcont-v0 --seed %d --demo %s --load %s" % (seed, command_args, model_dir_path)
            sys.argv = command.split()
            self.agent, self.env, self.args = initialize_agent_and_env()

            self.env = self.env.env
            self.n_sessions_per_dataset = n_sessions_per_dataset


        gep.update_global_env_prop_from_cfg()


        self.info_hists_of_datasets, self.r_sums_of_datasets = self.run_agent_on_all_datasets()


    def run_agent_session(self, dataset_number=None):

        info_hist = []
        self.env.render()
        self.env.reset()
        if isinstance(self.env, ATENAEnvCont):
            s = self.env.reset(dataset_number)


        else:
            raise TypeError("Only environment of type ATENAEnvCont is supported")

        if self.is_greedy:
            self.agent.begin_episode()
            self.agent.train(dataset_number=dataset_number, episode_length=cfg.MAX_NUM_OF_STEPS)

        r_sum = 0
        for ep_t in range(cfg.MAX_NUM_OF_STEPS):
            a = self.agent.act(s)
            next_s, r, done, info = self.env.step(a)
            if ArchName(cfg.arch) is ArchName.FF_PARAM_SOFTMAX:
                a = info["raw_action"]
            info_hist.append(StepInfo(continuous_action=a, reward=r, info=info))
            s = next_s
            r_sum += r
            if done:
                break
        return info_hist, r_sum

    def run_agent_on_dataset(self, dataset_number):

        info_hists = []
        r_sums = []
        for session in range(self.n_sessions_per_dataset):
            info_hist, r_sum = self.run_agent_session(dataset_number)
            info_hists.append(info_hist)
            r_sums.append(r_sum)

        return info_hists, r_sums

    def run_agent_on_all_datasets(self):

        info_hists_of_datasets = []
        r_sums_of_datasets = []
        for i in range(len(self.env.repo.data)):
            info_hists_of_dataset, r_sums_of_dataset = self.run_agent_on_dataset(i)
            info_hists_of_datasets.append(info_hists_of_dataset)
            r_sums_of_datasets.append(r_sums_of_dataset)

        return info_hists_of_datasets, r_sums_of_datasets

    def analyze_wrapper(self, analysis_func, dataset_number="all"):

        if analysis_func in [
            AgentAnalyzer.plot_action_entries_per_step,
            AgentAnalyzer.plot_ops_per_step,
            AgentAnalyzer.plot_col_id_per_op,
            AgentAnalyzer.plot_punishments_per_action_type,
            AgentAnalyzer.plot_col_id_per_filter_op,
            AgentAnalyzer.plot_reward_types_distribution_per_step,
            AgentAnalyzer.plot_reward_types_distribution,
            AgentAnalyzer.plot_avg_reward_per_step,
            AgentAnalyzer.plot_avg_reward_per_action_type,
            AgentAnalyzer.plot_punishment_distribution,
            AgentAnalyzer.plot_df_lens,
            AgentAnalyzer.plot_df_len_reward,
            AgentAnalyzer.plot_stacked_groups_per_step,
            AgentAnalyzer.get_best_session,
            AgentAnalyzer.plot_reward_types_distribution_per_action_type,
        ]:
            if dataset_number == "all":
                data = self.info_hists_of_datasets
                r_sums = self.r_sums_of_datasets
            else:
                data = [self.info_hists_of_datasets[dataset_number]]
                r_sums = [self.r_sums_of_datasets[dataset_number]]

            if analysis_func == AgentAnalyzer.plot_action_entries_per_step:
                data = self.get_action_vector_entries_data_from_info_hists(data)
            elif analysis_func == AgentAnalyzer.plot_col_id_per_op:
                data = self.get_col_id_per_op_from_info_hists(data)
            elif analysis_func == AgentAnalyzer.plot_col_id_per_filter_op:
                data = self.get_col_id_per_filter_op_from_info_hists(data)
            elif analysis_func == AgentAnalyzer.plot_punishments_per_action_type:
                data = self.get_punishments_per_action_type(data)
            elif analysis_func == AgentAnalyzer.plot_reward_types_distribution_per_step:
                data = self.reward_distribution_per_step_per_type(data)
            elif analysis_func == AgentAnalyzer.plot_reward_types_distribution:
                data = self.reward_distribution_per_step(data)
            elif analysis_func == AgentAnalyzer.get_best_session:
                data = (data, r_sums)
            elif analysis_func == AgentAnalyzer.plot_reward_types_distribution_per_action_type:
                data = self.reward_type_distribution_per_action_type(data)


            analysis_func(data, dataset_number)
        else:
            raise NotImplementedError

    def reward_distribution_per_step_per_dataset(self, dataset_number):
        return self.reward_distribution_per_step(
            [self.info_hists_of_datasets[dataset_number]])


    @staticmethod
    def repr_dataset_num(dataset_number):

        if dataset_number == "all":
            return ' for all datasets'

        return ' for dataset number %d' % dataset_number

    @staticmethod
    def get_vectors_of_cont_action_entries_in_each_step(info_hist):

        result_vectors = defaultdict(list)
        for step_info in info_hist:
            for i, entry_val in enumerate(step_info.continuous_action):
                result_vectors[ActionVectorEntry(i)].append(entry_val)

        return result_vectors

    @staticmethod
    def get_vector_of_cont_op_in_each_step(info_hist):

        result_vector = []
        for step_info in info_hist:
            result_vector.append(step_info.continuous_action[0])

        return result_vector

    @staticmethod
    def get_vector_of_disc_op_in_each_step(info_hist):

        result_vector = []
        for step_info in info_hist:
            result_vector.append(step_info.info["raw_action"][0])

        return result_vector

    @staticmethod
    def reward_distribution_per_step_per_type(info_hists_lst):

        step_rewards = [defaultdict(list) for _ in range(cfg.MAX_NUM_OF_STEPS)]


        reward_types = set()

        for info_hists in info_hists_lst:
            for info_hist in info_hists:
                for step, step_info in enumerate(info_hist):
                    for reward_type, value in step_info.info["reward_info"].items():
                        step_rewards[step][reward_type].append(abs(value))
                        reward_types.add(reward_type)


        reversed_dict = defaultdict(list)
        for step in range(len(step_rewards)):
            for reward_type in reward_types:
                reward_type_lst = step_rewards[step].get(reward_type, [0])
                reversed_dict[reward_type].append(sum(reward_type_lst) / len(reward_type_lst))

        return reversed_dict

    @staticmethod
    def reward_type_distribution_per_action_type(info_hists_lst):

        action_type_rewards = {action_type: defaultdict(list) for action_type in {'back', 'filter', 'group'}}


        reward_types = set()

        for info_hists in info_hists_lst:
            for info_hist in info_hists:
                for step, step_info in enumerate(info_hist):
                    action_type = ATENAUtils.OPERATOR_TYPE_LOOKUP[step_info.info["raw_action"][0]]
                    for reward_type, value in step_info.info["reward_info"].items():
                        action_type_rewards[action_type][reward_type].append(abs(value))
                        reward_types.add(reward_type)


        reversed_dict = defaultdict(float)
        for action_type in action_type_rewards.keys():
            for reward_type in reward_types:
                reward_type_lst = action_type_rewards[action_type].get(reward_type, [0])
                reversed_dict[(reward_type, action_type)] = sum(reward_type_lst) / len(reward_type_lst)

        return reversed_dict

    @staticmethod
    def reward_distribution_per_step(info_hists_lst):

        step_rewards = [defaultdict(list) for _ in range(cfg.MAX_NUM_OF_STEPS)]

        for info_hists in info_hists_lst:
            for info_hist in info_hists:
                for step, step_info in enumerate(info_hist):
                    for reward_type, value in step_info.info["reward_info"].items():
                        if value != 0:
                            step_rewards[step][reward_type].append(abs(value))

        return step_rewards

    @staticmethod
    def get_action_vector_entries_data_from_info_hists(data):

        cont_data_df = {entry.name.lower(): {"step": [], "entry_value": [], "category": []} for entry in
                        ActionVectorEntry}


        for dataset_data in data:
            for info_hist in dataset_data:
                for step, step_info in enumerate(info_hist):
                    cont_action_vect = step_info.continuous_action
                    if ArchName(cfg.arch) is ArchName.FF_GAUSSIAN:
                        disc_action_vect = gep.global_env_prop.compressed2full_range(cont_action_vect, True)
                    else:
                        disc_action_vect = cont_action_vect
                    disc_action_vect = ATENAEnvCont.cont2dis(disc_action_vect)
                    action_type = ATENAUtils.OPERATOR_TYPE_LOOKUP[disc_action_vect[0]]

                    for i, (cont_entry_val, disc_entry_val) in enumerate(zip(cont_action_vect, disc_action_vect)):
                        entry = ActionVectorEntry(i)


                        if action_type == "back":
                            if entry is not ActionVectorEntry.ACTION_TYPE:
                                continue

                        elif action_type == "filter":
                            if entry in [ActionVectorEntry.AGG_FUNC, ActionVectorEntry.AGG_COL_ID]:
                                continue

                        elif action_type == "group":
                            if entry in [ActionVectorEntry.FILTER_TERM, ActionVectorEntry.FILTER_OP]:
                                continue

                        entry_name = entry.name.lower()

                        if entry is ActionVectorEntry.ACTION_TYPE:
                            disc_cat = ATENAUtils.OPERATOR_TYPE_LOOKUP[disc_entry_val]

                        elif entry is ActionVectorEntry.COL_ID:
                            disc_cat = gep.global_env_prop.env_dataset_prop.KEYS[disc_entry_val]

                        elif entry is ActionVectorEntry.FILTER_OP:
                            disc_cat = ATENAUtils.INT_OPERATOR_MAP_ATENA_STR[disc_entry_val]

                        elif entry is ActionVectorEntry.FILTER_TERM:
                            disc_cat = "unknown"

                        elif entry is ActionVectorEntry.AGG_COL_ID:
                            disc_cat = gep.global_env_prop.env_dataset_prop.AGG_KEYS[disc_entry_val]

                        elif entry is ActionVectorEntry.AGG_FUNC:
                            disc_cat = "count"

                        if disc_cat == "packet_number":
                            disc_cat = "pkt_num"

                        cont_data_df[entry_name]["step"].append(step + 1)
                        cont_data_df[entry_name]["entry_value"].append(cont_entry_val)
                        cont_data_df[entry_name]["category"].append(disc_cat)


        cont_dfs = dict()
        for entry_name, entry_data_df in cont_data_df.items():
            cont_dfs[entry_name] = pd.DataFrame.from_dict(entry_data_df)
        return cont_dfs

    @staticmethod
    def get_col_id_per_op_from_info_hists(data):

        disc_data_df = {"op": [], "col_id": []}

        for dataset_data in data:
            for info_hist in dataset_data:
                for step, step_info in enumerate(info_hist):

                    cont_action_vect = step_info.continuous_action
                    if ArchName(cfg.arch) is ArchName.FF_GAUSSIAN:
                        disc_action_vect = gep.global_env_prop.compressed2full_range(cont_action_vect, True)
                    else:
                        disc_action_vect = cont_action_vect
                    disc_action_vect = ATENAEnvCont.cont2dis(disc_action_vect)
                    action_type = ATENAUtils.OPERATOR_TYPE_LOOKUP[disc_action_vect[0]]
                    col_id = gep.global_env_prop.env_dataset_prop.KEYS[disc_action_vect[1]]

                    if action_type == "back":
                        continue
                    else:
                        disc_data_df["op"].append(action_type)
                        disc_data_df["col_id"].append(col_id)

        return pd.DataFrame.from_dict(disc_data_df)

    @staticmethod
    def get_col_id_per_filter_op_from_info_hists(data):

        disc_data_df = {"filter_op": [], "col_id": []}

        for dataset_data in data:
            for info_hist in dataset_data:
                for step, step_info in enumerate(info_hist):

                    cont_action_vect = step_info.continuous_action
                    if ArchName(cfg.arch) is ArchName.FF_GAUSSIAN:
                        disc_action_vect = gep.global_env_prop.compressed2full_range(cont_action_vect, True)
                    else:
                        disc_action_vect = cont_action_vect
                    disc_action_vect = ATENAEnvCont.cont2dis(disc_action_vect)
                    action_type = ATENAUtils.OPERATOR_TYPE_LOOKUP[disc_action_vect[0]]
                    col_id = gep.global_env_prop.env_dataset_prop.KEYS[disc_action_vect[1]]

                    filter_op = ATENAUtils.INT_OPERATOR_MAP_ATENA_STR[disc_action_vect[2]]

                    if action_type != "filter":
                        continue
                    else:
                        disc_data_df["filter_op"].append(filter_op)
                        disc_data_df["col_id"].append(col_id)

        return pd.DataFrame.from_dict(disc_data_df)

    @staticmethod
    def get_punishments_per_action_type(data):

        x, y = 'action_type', 'punishment'
        data_df = {x: [], y: []}


        for dataset_data in data:
            for info_hist in dataset_data:
                for step_info in info_hist:
                    cont_action_vect = step_info.continuous_action
                    if ArchName(cfg.arch) is ArchName.FF_GAUSSIAN:
                        disc_action_vect = gep.global_env_prop.compressed2full_range(cont_action_vect, True)
                    else:
                        disc_action_vect = cont_action_vect
                    disc_action_vect = ATENAEnvCont.cont2dis(disc_action_vect)
                    action_type = ATENAUtils.OPERATOR_TYPE_LOOKUP[disc_action_vect[0]]

                    if step_info.info["reward_info"].empty_display < 0:
                        data_df[y].append("empty_display")

                    elif step_info.info["reward_info"].same_display_seen_already < 0:
                        data_df[y].append("same_display_already_seen")

                    elif step_info.info["reward_info"].back < 0:
                        data_df[y].append("back_with_no_history")

                    elif step_info.info["reward_info"].empty_groupings < 0:
                        data_df[y].append("empty_grouping")

                    else:
                        data_df[y].append("no_punishment")

                    data_df[x].append(action_type)

        return pd.DataFrame.from_dict(data_df)

    @staticmethod
    def create_percentage_plot(df, x, hue=None, title="", ax=None, dataset_number=None):


        plot = False
        if not ax:
            plot = True
            fig = plt.figure()
            ax = fig.add_subplot(111)


        x, y, hue = x, "prop", hue

        if hue:
            prop_df = (df[hue]
                       .groupby(df[x])
                       .value_counts(normalize=True)
                       .rename(y)
                       .reset_index())
        else:
            prop_df = (df[x]
                       .value_counts(normalize=True)
                       .rename(y)
                       .rename_axis(x)
                       .reset_index())


        sns.barplot(x=x, y=y, hue=hue, data=prop_df, ax=ax)


        ax.set_title('Proportion of ' + title)


        if plot:
            if hue:
                AgentAnalyzer.move_legend_to_the_right_of_ax(ax)
            plot_and_save(dataset_number=dataset_number)

    @staticmethod
    def move_legend_to_the_right_of_ax(ax):


        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])


        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    @staticmethod
    def get_vector_of_state_in_each_step(info_hist):

        result_vector = []
        for step_info in info_hist:
            result_vector.append(step_info.info["state"])

        return result_vector

    @staticmethod
    def get_vector_of_df_in_each_step(info_hist):

        result_vector = []
        for step_info in info_hist:
            df_dt, is_grouping = ATENAEnvCont.get_filtered_only_or_grouped_data_frame(step_info.info["raw_display"])
            result_vector.append((df_dt, is_grouping))

        return result_vector

    @staticmethod
    def get_vector_of_df_len_in_each_step(info_hist):

        result_vector = []
        for step_info in info_hist:
            df_dt, is_grouping = ATENAEnvCont.get_filtered_only_or_grouped_data_frame(step_info.info["raw_display"])
            result_vector.append((len(df_dt), is_grouping))

        return result_vector

    @staticmethod
    def get_vector_of_reward_in_each_step(info_hist):

        result_vector = []
        for step_info in info_hist:
            result_vector.append(step_info.reward)

        return result_vector

    @staticmethod
    def _plot_df_len_reward(df_lens_lst, rewards_lst, dataset_number):

        grouped_df_lens = []
        non_grouped_df_lens = []
        grouped_rewards = []
        non_grouped_rewards = []

        for i, (df_len, is_grouping) in enumerate(df_lens_lst):
            grouped_df_lens.append(df_len) if is_grouping else non_grouped_df_lens.append(df_len)
            grouped_rewards.append(rewards_lst[i]) if is_grouping else non_grouped_rewards.append(rewards_lst[i])

        fig, ax = plt.subplots()


        plt.title(r'Reward-Number of tuples \ groups (if grouped)')
        ax.scatter(grouped_df_lens, grouped_rewards, c='red', s=100.0, label='grouped', alpha=0.3, edgecolors='none')
        ax.scatter(non_grouped_df_lens, non_grouped_rewards, c='blue', s=100.0, label='non-grouped', alpha=0.3,
                   edgecolors='none')

        ax.set_ylabel('reward')
        ax.set_xlabel(r'df_len \ num of groups')

        ax.legend()

        plot_and_save(dataset_number=dataset_number)


    @classmethod
    def count_number_of_back_ops_in_session(cls, info_hist):
        return cls.count_number_of_op_in_session(info_hist=info_hist, op='back')

    @classmethod
    def count_number_of_filter_ops_in_session(cls, info_hist):
        return cls.count_number_of_op_in_session(info_hist=info_hist, op='filter')

    @classmethod
    def count_number_of_group_ops_in_session(cls, info_hist):
        return cls.count_number_of_op_in_session(info_hist=info_hist, op='group')

    @classmethod
    def count_number_of_op_in_session(cls, info_hist, op):

        op = ATENAUtils.INVERSE_OPERATOR_TYPE_LOOKUP(op)
        cntr = 0
        for step_info in info_hist:
            if step_info.info["raw_action"][0] == op:
                cntr += 1
        return cntr

    @classmethod
    def count_number_of_empty_display_punishments_in_session(cls, info_hist):
        return cls.count_number_of_punishments_in_session(
            info_hist=info_hist, punishment_type='empty_display')

    @classmethod
    def count_number_of_empty_grouping_punishments_in_session(cls, info_hist):
        return cls.count_number_of_punishments_in_session(
            info_hist=info_hist, punishment_type='empty_groupings')

    @classmethod
    def count_number_of_same_display_already_seen_punishments_in_session(cls, info_hist):
        return cls.count_number_of_punishments_in_session(
            info_hist=info_hist, punishment_type='same_display_seen_already')

    @classmethod
    def count_number_of_back_with_no_history_punishments_in_session(cls, info_hist):
        return cls.count_number_of_punishments_in_session(
            info_hist=info_hist, punishment_type='back')

    @classmethod
    def count_number_of_punishments_in_session(cls, info_hist, punishment_type):

        cntr = 0
        for step_info in info_hist:
            if step_info.info["reward_info"].__dict__[punishment_type] < 0:
                cntr += 1
        return cntr

    @classmethod
    def plot_action_entries_per_step(cls, cont_dfs, dataset_number="all"):

        sns.set()
        fig, axes = plt.subplots(nrows=2, ncols=3)

        for i, entry in enumerate(ActionVectorEntry):
            entry_name = entry.name.lower()
            cont_df = cont_dfs[entry_name]
            ax = axes[int((i + 3) / 6), i % 3]
            sns.swarmplot(x="step", y="entry_value", hue="category", data=cont_df, ax=ax)
            ax.set_title(entry_name)

            cls.move_legend_to_the_right_of_ax(ax)


        plt.subplots_adjust(hspace=0.35)
        plt.subplots_adjust(wspace=0.8)


        fig.suptitle("continuous values of all action vector entries in each step" + cls.repr_dataset_num(
            dataset_number), size=16)

        plot_and_save(dataset_number=dataset_number)

    @classmethod
    def plot_ops_per_step(cls, data, dataset_number="all"):

        disc_data_df = {"step": [], "op": []}

        for dataset_data in data:
            for info_hist in dataset_data:
                disc_ops_vector = cls.get_vector_of_disc_op_in_each_step(info_hist)
                for step in range(len(disc_ops_vector)):
                    disc_data_df["step"].append(step + 1)
                    disc_data_df["op"].append(disc_ops_vector[step])

        disc_df = pd.DataFrame.from_dict(disc_data_df)
        disc_df["op"] = disc_df["op"].map(ATENAUtils.OPERATOR_TYPE_LOOKUP)


        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        cls.create_percentage_plot(disc_df, "step", "op", 'operations per step' + cls.repr_dataset_num(dataset_number),
                                   ax1, dataset_number=dataset_number)
        cls.create_percentage_plot(disc_df, "op", None, 'operations' + cls.repr_dataset_num(dataset_number), ax2,
                                   dataset_number=dataset_number)


        cls.move_legend_to_the_right_of_ax(ax1)


        plt.subplots_adjust(hspace=0.35)


        plot_and_save(dataset_number=dataset_number)

    @classmethod
    def plot_col_id_per_op(cls, disc_df, dataset_number="all"):


        fig = plt.figure()
        ax = fig.add_subplot(111)

        cls.create_percentage_plot(disc_df, 'op', 'col_id', 'columns per operation', ax,
                                   dataset_number=dataset_number)


        cls.move_legend_to_the_right_of_ax(ax)


        plot_and_save(dataset_number=dataset_number)

    @classmethod
    def plot_col_id_per_filter_op(cls, disc_df, dataset_number="all"):


        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        cls.create_percentage_plot(disc_df, 'filter_op', None, 'filter_op' + cls.repr_dataset_num(dataset_number),
                                   ax1,
                                   dataset_number=dataset_number)
        cls.create_percentage_plot(disc_df, 'filter_op', 'col_id',
                                   'columns per filter_op' + cls.repr_dataset_num(dataset_number),
                                   ax2, dataset_number=dataset_number)


        cls.move_legend_to_the_right_of_ax(ax2)


        plt.subplots_adjust(hspace=0.35)


        plot_and_save(dataset_number=dataset_number)

    @classmethod
    def plot_punishments_per_action_type(cls, data_df, dataset_number="all"):

        cls.create_percentage_plot(data_df,
                                   'action_type',
                                   'punishment',
                                   'punishments per action type',
                                   dataset_number=dataset_number)

    @classmethod
    def plot_reward_types_distribution_per_step(cls, data, dataset_number="all"):

        columns = ["reward_type"] + [str(i + 1) for i in range(cfg.MAX_NUM_OF_STEPS)]
        data_df = [[reward_type] + reward_type_vec for reward_type, reward_type_vec in data.items() if
                   reward_type != "interestingness"]
        data_diversity_interestingness_df = [[reward_type] + reward_type_vec for reward_type, reward_type_vec in
                                             data.items() if reward_type in ["interestingness", "diversity", "humanity"]]

        df = pd.DataFrame(data=data_df, columns=columns)
        diversity_interestingness_df = pd.DataFrame(data=data_diversity_interestingness_df, columns=columns)
        sns.set()
        fig, axes = plt.subplots(nrows=2, ncols=1)


        df = df.set_index('reward_type').T
        df.plot(kind='bar', stacked=True, ax=axes[0])


        axes[0].set_title('Plot of average absolute value of each reward type per step')
        diversity_interestingness_df.set_index('reward_type').T.plot(kind='bar', stacked=True, ax=axes[1])

        for i in range(2):
            ax = axes[i]
            ax.set_xlabel('step')
            ax.set_ylabel('average_reward')


            cls.move_legend_to_the_right_of_ax(ax)


        plot_and_save(dataset_number=dataset_number)

    @classmethod
    def plot_reward_types_distribution_per_action_type(cls, data, dataset_number="all"):

        action_types = ['back', 'filter', 'group']
        reward_types = set()
        for reward_type, _ in data.keys():
            reward_types.add(reward_type)
        columns = ["reward_type"] + [action_type for action_type in action_types]
        data_df = [[reward_type] + [data[(reward_type, action_type)] for action_type in action_types] for
                   reward_type in reward_types if reward_type != "interestingness"]
        data_diversity_interestingness_humanity_df = [
            [reward_type] + [data[(reward_type, action_type)] for action_type in action_types] for reward_type
            in reward_types if reward_type in ["interestingness", "diversity", "humanity"]]

        df = pd.DataFrame(data=data_df, columns=columns)
        diversity_interestingness_df = pd.DataFrame(data=data_diversity_interestingness_humanity_df, columns=columns)
        sns.set()
        fig, axes = plt.subplots(nrows=2, ncols=1)


        df = df.set_index('reward_type').T
        df.plot(kind='bar', stacked=True, ax=axes[0])


        axes[0].set_title('Plot of average absolute value of each reward type per action type')
        diversity_interestingness_df.set_index('reward_type').T.plot(kind='bar', stacked=True, ax=axes[1])

        for i in range(2):
            ax = axes[i]
            ax.set_xlabel('action_type')
            ax.set_ylabel('average_reward')


            cls.move_legend_to_the_right_of_ax(ax)


        plot_and_save(dataset_number=dataset_number)

    @classmethod
    def plot_reward_types_distribution(cls, data, dataset_number="all"):

        data_df = {"reward_type": [], "reward": []}

        for step_data in data:
            for reward_column, rewards_list in step_data.items():
                for reward in rewards_list:

                    if reward_column == "humanity" and reward < 0:
                        reward_column = "humanity_punishment"
                    elif reward_column == "humanity" and reward >= 0:
                        reward_column = "humanity_gain"
                    data_df["reward_type"].append(reward_column)
                    data_df["reward"].append(reward)

        df = pd.DataFrame.from_dict(data_df)


        df["reward_bin"] = pd.cut(df["reward"].tolist(), 8)


        df["reward_bin_new"] = df["reward"]
        for i, line in df.iterrows():
            df.iloc[i, 3] = df.iloc[i, 2]

        x, y, hue = "reward_type", "proportion", "reward_bin"
        f, axes = plt.subplots(2, 1)

        df_plot = df.groupby(['reward_bin_new', 'reward_type']).size().reset_index().pivot(
            columns='reward_bin_new', index='reward_type', values=0).fillna(0).astype(int)

        df_plot.plot(kind='bar', stacked=True, rot=0, ax=axes[0])


        axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))


        hue = "reward_bin_new"
        prop_df = (df[hue]
                   .groupby(df[x])
                   .value_counts(normalize=True)
                   .rename(y)
                   .reset_index())

        sns.barplot(x=x, y=y, hue=hue, hue_order=sorted(df[hue].unique().tolist()),
                    order=sorted(df[x].unique().tolist()), data=prop_df, ax=axes[1])

        axes[0].set_ylabel('count')


        axes[0].set_title('Count of each reward absolute value occurrences (using equal-width bining) per reward type')
        axes[1].set_title(
            'Proportion of each absolute reward value occurrences (using equal-width bining) per reward type')

        for i in range(2):
            ax = axes[i]


            cls.move_legend_to_the_right_of_ax(ax)


        plt.subplots_adjust(hspace=0.35)


        plot_and_save(dataset_number=dataset_number)

    @classmethod
    def plot_avg_reward_per_step(cls, data, dataset_number="all"):

        x, y = "step", "reward"
        data_df = {x: [], y: []}

        for dataset_data in data:
            for info_hist in dataset_data:
                for step, step_info in enumerate(info_hist):
                    data_df[x].append(step + 1)
                    data_df[y].append(step_info.reward)

        df = pd.DataFrame.from_dict(data_df)
        ax = sns.barplot(x=x, y=y, data=df, palette="Set3")

        ax.set_xlabel('step')
        ax.set_ylabel('avg_reward')


        plt.title('Average reward per step' + cls.repr_dataset_num(dataset_number))


        plot_and_save(dataset_number=dataset_number)

    @classmethod
    def plot_avg_reward_per_action_type(cls, data, dataset_number="all"):

        x, y = "action_type", "reward"
        data_df = {x: [], y: []}

        for dataset_data in data:
            for info_hist in dataset_data:
                for step, step_info in enumerate(info_hist):
                    cont_action_vect = step_info.continuous_action
                    if ArchName(cfg.arch) is ArchName.FF_GAUSSIAN:
                        disc_action_vect = gep.global_env_prop.compressed2full_range(cont_action_vect, True)
                    else:
                        disc_action_vect = cont_action_vect
                    disc_action_vect = ATENAEnvCont.cont2dis(disc_action_vect)
                    action_type = ATENAUtils.OPERATOR_TYPE_LOOKUP[disc_action_vect[0]]
                    data_df[x].append(action_type)
                    data_df[y].append(step_info.reward)

        df = pd.DataFrame.from_dict(data_df)
        ax = sns.barplot(x=x, y=y, data=df, palette="Set3")

        ax.set_xlabel('action_type')
        ax.set_ylabel('avg_reward')


        plt.title('Average reward per action_type' + cls.repr_dataset_num(dataset_number))


        plot_and_save(dataset_number=dataset_number)

    @classmethod
    def plot_punishment_distribution(cls, data, dataset_number="all"):

        x, y = "punishment_type", "count_per_session"
        data_df = {x: [], y: []}


        for dataset_data in data:
            for info_hist in dataset_data:
                back_with_no_history_cnt = cls.count_number_of_back_with_no_history_punishments_in_session(info_hist)
                empty_display_cnt = cls.count_number_of_empty_display_punishments_in_session(info_hist)
                empty_grouping_cnt = cls.count_number_of_empty_grouping_punishments_in_session(info_hist)
                same_display_already_seen_cnt = cls.count_number_of_same_display_already_seen_punishments_in_session(
                    info_hist)

                data_df[x].append("back_with_no_history")
                data_df[y].append(back_with_no_history_cnt)
                data_df[x].append("empty_display")
                data_df[y].append(empty_display_cnt)
                data_df[x].append("empty_grouping")
                data_df[y].append(empty_grouping_cnt)
                data_df[x].append("same_display_already_seen")
                data_df[y].append(same_display_already_seen_cnt)

        df = pd.DataFrame.from_dict(data_df)
        ax = sns.barplot(x=x, y=y, data=df, palette="Set3")


        plt.title('Average number of occurrences of each punishment type in a session')


        plot_and_save(dataset_number=dataset_number)

    @classmethod
    def plot_df_lens(cls, data, dataset_number="all"):

        x, y = "is_grpuped", "df_len"
        data_df = {x: [], y: []}

        for dataset_data in data:
            for info_hist in dataset_data:
                df_len_is_grouped_pair_vector = cls.get_vector_of_df_len_in_each_step(info_hist)
                df_len_vector = [df_len for df_len, is_grouped in df_len_is_grouped_pair_vector]
                df_is_grouped_vector = ["grouped" if is_grouped else "non-grouped" for df_len, is_grouped in
                                        df_len_is_grouped_pair_vector]
                data_df[x].extend(df_is_grouped_vector)
                data_df[y].extend(df_len_vector)

        df = pd.DataFrame.from_dict(data_df)


        hue, new_hue = "df_len_bin", "df_len_bin_new"

        df[hue] = pd.cut(df[y].tolist(), [-0.1, 0, 1, 1.99, 4, 10, 20, 50, 100, 350, 700, 1500, 5000, 15000])


        df[new_hue] = df[hue]
        for i, line in df.iterrows():
            df.iloc[i, 3] = df.iloc[i, 2]

        f, axes = plt.subplots(2, 1)

        df_plot = df.groupby([new_hue, x]).size().reset_index().pivot(
            columns=new_hue, index=x, values=0).fillna(0).astype(int)

        df_plot.plot(kind='bar', stacked=True, rot=0, ax=axes[0])


        axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))

        axes[0].set_ylabel('count')


        hue = new_hue
        y = 'prop'
        prop_df = (df[hue]
                   .groupby(df[x])
                   .value_counts(normalize=True)
                   .rename(y)
                   .reset_index())

        sns.barplot(x=x, y=y, hue=hue, hue_order=sorted(df[hue].unique().tolist()),
                    order=sorted(df[x].unique().tolist()), data=prop_df, ax=axes[1])


        axes[0].set_title('Count of each DataFrames length for non-grouped and group numbers for grouped')
        axes[1].set_title('Proportion of each DataFrames length for non-grouped and group numbers for grouped')

        for i in range(2):
            ax = axes[i]


            cls.move_legend_to_the_right_of_ax(ax)


        plt.subplots_adjust(hspace=0.35)


        plot_and_save(dataset_number=dataset_number)

    @classmethod
    def plot_df_len_reward(cls, data, dataset_number="all"):

        df_lens_lst = []
        rewards_lst = []

        for dataset_data in data:
            for info_hist in dataset_data:
                df_len_vector = cls.get_vector_of_df_len_in_each_step(info_hist)
                reward_vector = cls.get_vector_of_reward_in_each_step(info_hist)
                df_lens_lst.extend(df_len_vector)
                rewards_lst.extend(reward_vector)


        cls._plot_df_len_reward(df_lens_lst, rewards_lst, dataset_number)

    @classmethod
    def plot_stacked_groups_per_step(cls, data, dataset_number="all"):

        x, hue = "step", "stacked_groups_num"
        data_df = {x: [], hue: []}

        for dataset_data in data:
            for info_hist in dataset_data:
                states_vector = cls.get_vector_of_state_in_each_step(info_hist)
                for i, state in enumerate(states_vector):
                    data_df[hue].append(len(state['grouping']))
                    data_df[x].append(i+1)

        df = pd.DataFrame.from_dict(data_df)
        cls.create_percentage_plot(df, x, hue, 'stacked_groups_num per step' + cls.repr_dataset_num(dataset_number),
                                   dataset_number=dataset_number)

    @classmethod
    def get_best_session(cls, data, dataset_number="all"):

        info_hists_lst = data[0]
        r_sums_lst = data[1]
        max_r_sum = -100000
        max_r_sum_session = None

        for lst_idx, r_sums in enumerate(r_sums_lst):
            for intra_lst_idx, r_sum in enumerate(r_sums):
                if r_sum > max_r_sum:
                    max_r_sum = r_sum
                    max_r_sum_session = info_hists_lst[lst_idx][intra_lst_idx]


        max_r_sum_raw_actions = [step_info.info["raw_action"] for step_info in max_r_sum_session]
        max_r_sum_actions = [step_info.info["action"] for step_info in max_r_sum_session]
        for raw_action in max_r_sum_raw_actions:
            raw_action[3] -= 0.5

        filter_terms = [step_info.info["filter_term"] for step_info in max_r_sum_session]


        if dataset_number is not None and dataset_number != "all":
            path = '%s/d%d/max_r_sum_actions.txt' % (results_path, dataset_number)
        else:
            path = '%s/all/max_r_sum_actions.txt' % results_path

        if SAVE:
            with open(path, 'w') as f:
                for action in max_r_sum_actions:
                    f.write(str(action) + ',\n')
                for raw_action in max_r_sum_raw_actions:
                    f.write(str(raw_action) + ',\n')
                f.write('\n')
                for filter_term in filter_terms:
                    if filter_term is not None:
                        f.write(f"'{str(filter_term)}'" + ',\n')
                    else:
                        f.write(str(filter_term) + ',\n')
                f.write('r_sum is: %f' % max_r_sum)


        return max_r_sum, max_r_sum_session


if __name__ == '__main__':
    agent_analyzer = AgentAnalyzer(model_dir_path=model_dir_path,
                                   command_args=command_args,
                                   n_sessions_per_dataset=100,
                                   seed=0)


    print(agent_analyzer.reward_distribution_per_step_per_dataset(0))
    print(agent_analyzer.reward_distribution_per_step_per_dataset(1))
    print(agent_analyzer.reward_distribution_per_step_per_dataset(2))
    print(agent_analyzer.reward_distribution_per_step_per_dataset(3))


    for i in list(range(len(agent_analyzer.env.repo.data)))+["all"]:
        CURRENT_PLOT_NUM = 1


        agent_analyzer.analyze_wrapper(AgentAnalyzer.get_best_session, dataset_number=i)
