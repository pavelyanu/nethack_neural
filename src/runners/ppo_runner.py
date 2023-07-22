from src.runners.abstract_runner import AbstractRunner
from src.agents.ppo_agent import AbstractPPOAgent

class PPORunner(AbstractRunner):
    def __init__(self, env, agent, loggers):
        super().__init__(env, agent,loggers)

    def run(self, num_episodes=1000, render=False):
        timesteps = 0
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
                timesteps += 1
            self.agent.train()
            self.evaluate(render=render)
            self.log("Episode {} complete. Total timesteps: {}".format(episode, timesteps))
        self.agent.save('models/')

    def run_vectorized(self, num_envs, num_episodes=1000, render=False):
        timesteps = 0
        for episode in range(num_episodes):
            states = self.env.reset(num_envs)
            done = [False] * num_envs
            while not all(done):
                if render:
                    self.env.render()
                actions, logprobs = self.agent.act(states)
                next_states, rewards, done, _ = self.env.step(actions)
                self.agent.save_transition(states, actions, logprobs, rewards, next_states, done)
                states = next_states
                timesteps += num_envs
            self.agent.train()
            self.evaluate(render=render)
            self.log("Episode {} complete. Total timesteps: {}".format(episode, timesteps))
        self.agent.save('models/')
