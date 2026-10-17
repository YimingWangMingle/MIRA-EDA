from functools import lru_cache

from gym_atena.envs.atena_env_cont import ATENAEnvCont


class GreedyAgent(object):

    def __init__(self,
                 kl_coeff,
                 compaction_coeff,
                 diversity_coeff,
                 humanity_coeff,
                 ):
        self.kl_coeff = kl_coeff
        self.compaction_coeff = compaction_coeff
        self.diversity_coeff = diversity_coeff
        self.humanity_coeff = humanity_coeff
        self.acts_lst = []
        self.step_num = 0

    @lru_cache(maxsize=32)
    def train(self, dataset_number, episode_length, verbose=True):

        if verbose:
            print(f'Training greedy algorithm on dataset No. {dataset_number} for {episode_length} steps')
        acts_lst, _ = ATENAEnvCont.get_greedy_max_reward_actions_lst(
            dataset_number=dataset_number,
            episode_length=episode_length,
            kl_coeff=self.kl_coeff,
            compaction_coeff=self.compaction_coeff,
            diversity_coeff=self.diversity_coeff,
            humanity_coeff=self.humanity_coeff,
            verbose=True,
        )
        self.acts_lst = acts_lst

    def act(self, dummy=None):

        try:
            action = self.acts_lst[self.step_num]
            self.step_num += 1
            return action
        except IndexError:
            raise IndexError("Did you forget to train greedy agent with the correct number of steps?")

    def begin_episode(self):

        self.step_num = 0
