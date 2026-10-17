import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='ATENAld-v0',
    entry_point='gym_ianna.envs:ATENAEnv',


)

register(
    id='ATENAcont-v0',
    entry_point='gym_atena.envs.atena_env_cont:ATENAEnvCont',


)
