from gym.envs.registration import register

register(
    id='campus-v0',
    entry_point='campus_gym.envs:CampusGymEnv',
)
register(
    id='campus-extrahard-v0',
    entry_point='campus_gym.envs:CampusExtraHardEnv',
)