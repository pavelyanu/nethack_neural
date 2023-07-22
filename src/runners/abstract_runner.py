from abc import ABC, abstractmethod

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

    @abstractmethod
    def run(self, num_episodes=100, render=False):
        pass

    def log(self, msg):
        for logger in self._loggers:
            logger.log(msg)

    def evaluate(self, num_episodes=100, render=False, agent=None, env=None):
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
                action = agent.act(state, train=False)
                if isinstance(action, tuple):
                    action = action[0]
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
        self.log("Reward: {}".format(total_reward / num_episodes))