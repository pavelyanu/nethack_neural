from abc import ABC, abstractmethod
from numpy import ndarray
from datetime import datetime

class AbstractRunner(ABC):
    @abstractmethod
    def __init__(self, env, agent, loggers=[]):
        self._env = env
        self._agent = agent
        if loggers is None:
            loggers = []
        elif (isinstance(loggers, list) == False):
            loggers = [loggers]
        self._loggers = loggers

    @property
    def env(self):
        return self._env

    @property
    def agent(self):
        return self._agent

    def log(self, msg):
        for logger in self._loggers:
            logger.log(msg)

    def evaluate(self, num_episodes=10, render=False, agent=None, env=None):
        if agent is None:
            agent = self.agent
        if env is None:
            env = self.env
        total_reward = 0
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                if render:
                    env.render()
                state = agent.preprocess(state, add_batch_dim=True)
                action = agent.act(state, train=False)
                while len(action) > 1:
                    action = action[0]
                if isinstance(action, tuple) or isinstance(action, list) or isinstance(action, ndarray):
                    action = action[0]
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
        self.log("Time: {} Reward: {}".format(datetime.now().strftime("%H:%M:%S"), total_reward / num_episodes))