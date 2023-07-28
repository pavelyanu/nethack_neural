import torch

from src.runners.abstract_runner import AbstractRunner
from src.agents.ppo_agent import AbstractPPOAgent
from src.utils.transition import Transition

class PPORunner(AbstractRunner):
    def __init__(self, env, agent, loggers):
        super().__init__(env, agent,loggers)

    def run_vectorized(self, num_envs, eval_env, num_steps=10000, train_step=2000, eval_step=2000, render=False):
        timesteps = 0
        states = self.env.reset()
        states = self.agent.preprocess(states)
        while timesteps < num_steps:
            actions, logprobs = self.agent.act(states)
            next_states, rewards, done, _ = self.env.step(actions)
            self.agent.save_transition(
                state=states,
                action=actions,
                reward=rewards,
                logprob=logprobs,
                done=done
            )
            states = next_states
            states = self.agent.preprocess(states)
            timesteps += num_envs

            if timesteps % train_step == 0:
                self.agent.last_state(states)
                self.agent.train()

            if timesteps % eval_step == 0:
                self.evaluate(render=render, env=eval_env, num_episodes=5)
                self.log(f"Total timesteps: {timesteps}")