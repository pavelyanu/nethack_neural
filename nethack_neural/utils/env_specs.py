from gym.spaces.dict import Dict
class EnvSpecs:
    """A class that holds specifications of an environment.

    This class stores the shapes of observation and action spaces, the number of possible actions, 
    and the number of environments (in case of multiple parallel environments).

    Attributes:
        observation_shape (tuple or dict): The shape of the observation space. 
            For environments with a Dict observation space, this will be a dictionary mapping 
            each key in the space to its shape.
        action_shape (tuple): The shape of the action space.
        num_actions (int): The number of possible actions in the action space.
        num_envs (int): The number of environments in use.
    """
    def __init__(self) -> None:
        self.observation_shape = None
        self.action_shape = None
        self.num_actions = None
        self.num_envs = None

    def init_with_gym_env(self, env, num_envs=1):
        """Initialize the environment specifications using a gym environment.

        Args:
            env (gym.Env): The gym environment to initialize from.
            num_envs (int): The number of environments in use. Default is 1.
        """
        if isinstance(env.observation_space, Dict):
            self.observation_shape = {k: env.observation_space[k].shape for k in env.observation_space.spaces}
        else:
            self.observation_shape = env.observation_space.shape
        self.action_shape = env.action_space.shape
        self.num_actions = env.action_space.n
        self.num_envs = num_envs