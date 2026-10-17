MAX_NUM_OF_STEPS = 12


outdir = ""


num_envs = 1


n_hidden_channels = 64
arch = 'FFGaussian'
beta = 1.0


schema = 'NETWORKING'


dataset_number = None


no_back = False


obs_with_step_num = False


stack_obs_num = 1


bins_sizes = 'CUSTOM_WIDTH'
exponential_sizes_num_of_bins = 35


use_humans_reward = True
humans_reward_interval = 64
humanity_coeff = 1.0


no_diversity = False
diversity_coeff = 1.0


no_interestingness = False
kl_coeff = 1.0
compaction_coeff = 1.0
count_data_driven = False


log_interval = 1000


max_nn_tokens = 12
cache_dfs_size = 750
cache_tokenization_size = 10000
cache_distances_size = 750
