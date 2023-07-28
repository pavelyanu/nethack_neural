import torch

from src.runners.abstract_runner import AbstractRunner
from src.agents.ppo_agent import AbstractPPOAgent
from src.utils.transition import Transition

class PPORunner(AbstractRunner):
    def __init__(self, env, agent, loggers):
        super().__init__(env, agent,loggers)

    def run_vectorized(self, num_envs, eval_env, total_steps=10000, worker_steps=2000, evaluation_period=2000, evaluation_steps=10, render=False):
        timesteps = 0
        train_counter = 0
        eval_counter = 0
        states = self.env.reset()
        states = self.agent.preprocess(states)
        while timesteps < total_steps:
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
            train_counter += num_envs
            eval_counter += num_envs

            if train_counter >= worker_steps:
                self.agent.last_state(states)
                self.agent.train()
                train_counter = 0

            if eval_counter >= evaluation_period:
                self.evaluate(render=render, env=eval_env, num_episodes=evaluation_steps)
                self.log(f"Total timesteps: {timesteps}")
                eval_counter = 0