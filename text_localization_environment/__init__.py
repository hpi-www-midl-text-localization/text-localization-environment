from gym.envs.registration import register
from text_localization_environment.TextLocEnv import TextLocEnv

register(
    id='TextLocEnv-v0',
    entry_point='text_localization_environment.environment:TextLocEnv',
)
