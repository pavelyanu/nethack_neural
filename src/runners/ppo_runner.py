from src.runners.abstract_runner import AbstractRunner
from src.agents.ppo_agent import PPOAgent

class PPORunner(AbstractRunner):
    def __init__(self, env, agent, loggers):
        super().__init__(env, agent,loggers)

    def run(self, num_episodes=1000, render=False):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                if render:
                    self.env.render()
                action, logprob = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.save_transition(state, action, logprob, reward, next_state, done)
                state = next_state
            self.agent.train()
            self.evaluate(render=render)
        self.agent.save('models/')