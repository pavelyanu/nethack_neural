from gym.spaces.dict import Dict
class EnvSpecs:
    def __init__(self) -> None:
        self.observation_shape = None
        self.action_shape = None
        self.num_actions = None
        self.num_envs = None

    def init_with_gym_env(self, env, num_envs=1):
        if isinstance(env.observation_space, Dict):
            self.observation_shape = {k: env.observation_space[k].shape for k in env.observation_space.spaces}
        self.action_shape = env.action_space.shape
        self.num_actions = env.action_space.n
        self.num_envs = num_envs