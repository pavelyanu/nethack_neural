from abc import ABC, abstractmethod
from numpy import ndarray
from datetime import datetime
from nethack_neural.agents.abstract_agent import AbstractAgent

class AbstractRunner(ABC):
    """Abstract base class for running reinforcement learning experiments.

    This class provides a generic runner for conducting experiments in a reinforcement learning environment. 
    It handles the interaction between the agent and the environment and provides logging capabilities.

    Args:
        env (gym.Env): The gym environment in which to run the experiments.
        agent (AbstractAgent): The agent that will interact with the environment.
        loggers (list): A list of loggers for logging experiment results.
    """
    @abstractmethod
    def __init__(self, env, agent: AbstractAgent, loggers=[]):
        self._env = env
        self._agent = agent
        if loggers is None:
            loggers = []
        elif (isinstance(loggers, list) == False):
            loggers = [loggers]
        self._loggers = loggers

    @property
    def env(self):
        """The gym environment in which to run the experiments."""
        return self._env

    @property
    def agent(self):
        """The agent that will interact with the environment."""
        return self._agent

    @abstractmethod
    def run(self, *args, **kwargs):
        """Runs the experiment. This method must be implemented by subclasses."""
        pass

    def log_message(self, msg):
        """Logs a message.

        Args:
            msg (str): The message to log.
        """
        time = datetime.now()
        for logger in self._loggers:
            logger.log_message(time, msg)
    
    def log_stats(self, *, timestep, reward, length):
        """Logs statistics about the experiment.

        Args:
            timestep (int): The current timestep of the experiment.
            reward (float): The reward obtained in the current timestep.
            length (int): The length of the experiment.
        """
        time = datetime.now()
        for logger in self._loggers:
            logger.log_stats(time=time, timestep=timestep, reward=reward, length=length)

    def evaluate(self, timestep, num_episodes=10, render=False, agent=None, env=None):
        """Evaluates the performance of the agent.

        This method runs a number of episodes in the environment using the agent's policy and logs the results.

        Args:
            timestep (int): The current timestep of the experiment.
            num_episodes (int): The number of episodes to run for the evaluation.
            render (bool): Whether to render the environment.
            agent (AbstractAgent): The agent to use for the evaluation. If None, uses the runner's agent.
            env (gym.Env): The environment to use for the evaluation. If None, uses the runner's environment.
        """
        if agent is None:
            agent = self.agent
        if env is None:
            env = self.env
        total_reward = 0
        length = 0
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                if render:
                    env.render()
                action = agent.act(state, train=False)
                try:
                    while len(action) > 1:
                        action = action[0]
                except:
                    pass
                try:
                    if isinstance(action, tuple) or isinstance(action, list) or isinstance(action, ndarray):
                        action = action[0]
                except:
                    pass
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
                length += 1
        self.log_message(f"Evaluation reward: {total_reward / num_episodes}, Average length: {length / num_episodes}")
        self.log_stats(timestep=timestep, reward=total_reward / num_episodes, length=length / num_episodes)