from nethack_neural.runners.abstract_runner import AbstractRunner
import pandas as pd
import tqdm
import plotext as plt
import threading
import time
from datetime import datetime
import numpy as np
from numpy import ndarray

class PPOFullRunner(AbstractRunner):
    def __init__(self, env, agent, loggers, use_tqdm=False, use_visualization=False):
        super().__init__(env, agent, loggers)
        self.use_tqdm = use_tqdm
        self.use_visualization = use_visualization

        # PPORunner attributes
        self.progress_bar = None
        
        # PPOVisualRunner attributes
        if self.use_visualization:
            self.running_data = pd.DataFrame(columns=['time', 'timestep', 'reward', 'length'])
            self.evaluation_data = pd.DataFrame(columns=['time', 'timestep', 'reward', 'length'])
            self.current_replay = []
            self.pending_replay = []
            self.terminal_width = plt.terminal_width()
            self.terminal_height = plt.terminal_height()
            self.plot_width = self.terminal_width
            self.plot_height = self.terminal_height - 27
            if self.plot_height < 10:
                raise Exception("Terminal height too small for full visualization.")
            self.running_data_buffer = None
            self.replay_index = 0
            self.last_frame_time = time.time()
            self.last_visualization_time = time.time()
            self.plot_refresh_rate = 0.5
            self.replay_refresh_rate = 0.1
            self.plot_string = ""
            self.timesteps = 0
            self.progress_bar_steps = 0

    def run(self, num_envs, eval_env, total_steps=10000, worker_steps=2000, evaluation_period=2000, evaluation_steps=10, render=False):
        if self.use_visualization:
            self._run_visual(num_envs, eval_env, total_steps, worker_steps, evaluation_period, evaluation_steps, render)
        else:
            self._run_non_visual(num_envs, eval_env, total_steps, worker_steps, evaluation_period, evaluation_steps, render)

    def _run_non_visual(self, num_envs, eval_env, total_steps=10000, worker_steps=2000, evaluation_period=2000, evaluation_steps=10, render=False):
        env_rewards = [0 for _ in range(num_envs)]
        reward_log = []
        if self.use_tqdm:
            self.progress_bar = tqdm.tqdm(total=total_steps)
            self.progress_bar.update(0)

        timesteps = 0
        train_counter = 0
        eval_counter = 0
        states = self.env.reset()
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
            timesteps += num_envs
            train_counter += num_envs
            eval_counter += num_envs
            if self.use_tqdm:
                self.progress_bar.update(num_envs)

            if self.use_tqdm:
                for i in range(num_envs):
                    env_rewards[i] += rewards[i]
                    if done[i]:
                        reward_log.append(env_rewards[i])
                        env_rewards[i] = 0
                        if len(reward_log) > 100:
                            reward_log.pop(0)
                        self.progress_bar.set_postfix_str(f"Mean reward: {sum(reward_log) / len(reward_log)}")

            if train_counter >= worker_steps:
                self.agent.last_state(states)
                self.agent.train()
                train_counter = 0

            if eval_counter >= evaluation_period:
                self.evaluate(timestep=timesteps, render=render, env=eval_env, num_episodes=evaluation_steps)
                self.log_message(f"Total timesteps: {timesteps}")
                eval_counter = 0

    def _run_visual(self, num_envs, eval_env, total_steps=10000, worker_steps=2000, evaluation_period=2000, evaluation_steps=10, render=False):
        self.timesteps = 0
        train_counter = 0
        eval_counter = 0
        visual_counter = 0
        self.progress_bar = tqdm.tqdm(total=total_steps)
        self.running_data_buffer = [{'reward': 0, 'length': 0} for _ in range(num_envs)]
        states = self.env.reset()
        while self.timesteps < total_steps:
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
            self.timesteps += num_envs
            train_counter += num_envs
            eval_counter += num_envs
            visual_counter += 1

            for i in range(num_envs):
                self.running_data_buffer[i]['reward'] += rewards[i]
                self.running_data_buffer[i]['length'] += 1
                if done[i]:
                    data = {'time': datetime.now(),
                            'timestep': self.timesteps,
                            'reward': self.running_data_buffer[i]['reward'],
                            'length': self.running_data_buffer[i]['length']}
                    self.running_data = pd.concat([self.running_data, pd.DataFrame(data, index=[0])], ignore_index=True)
                    self.running_data_buffer[i]['reward'] = 0
                    self.running_data_buffer[i]['length'] = 0

            if train_counter >= worker_steps:
                self.agent.last_state(states)
                thread = threading.Thread(target=self.agent.train)
                # self.agent.train()
                thread.start()
                while thread.is_alive():
                    self.visualize()
                    time.sleep(min(self.plot_refresh_rate, self.replay_refresh_rate))
                thread.join()
                train_counter = 0

            if eval_counter >= evaluation_period:
                self.evaluate(timestep=self.timesteps, render=render, env=eval_env, num_episodes=evaluation_steps)
                self.log_message(f"Total timesteps: {self.timesteps}")
                eval_counter = 0

            self.visualize()

    def evaluate(self, timestep, num_episodes=10, render=False, agent=None, env=None):
        if self.use_visualization:
            self._evaluate_visual(timestep, num_episodes, render, agent, env)
        else:
            super().evaluate(timestep, num_episodes, render, agent, env)

    def _evaluate_visual(self, timestep, num_episodes=10, render=False, agent=None, env=None):
        if agent is None:
            agent = self.agent
        if env is None:
            env = self.env
        total_reward = 0
        length = 0
        max_reward = -np.inf
        final_replay =[]
        for episode in range(num_episodes):
            episode_reward = 0
            frames = []
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
                episode_reward += reward
                state = next_state
                length += 1
                frames.append(env.render(mode='ansi'))
            if episode_reward > max_reward:
                max_reward = episode_reward
                final_replay = frames
            self.visualize()

        if self.current_replay:
            self.pending_replay = final_replay
        else:
            self.current_replay = final_replay
            self.replay_index = 0
        self.log_message(f"Evaluation reward: {total_reward / num_episodes}, Average length: {length / num_episodes}")
        self.log_stats(timestep=timestep, reward=total_reward / num_episodes, length=length / num_episodes)
        data = {'time': datetime.now(),
                'timestep': timestep,
                'reward': total_reward / num_episodes,
                'length': length / num_episodes}
        self.evaluation_data = pd.concat([self.evaluation_data, pd.DataFrame(data, index=[0])], ignore_index=True)

    def visualize(self):
        current_time = time.time()
        if current_time - self.last_visualization_time > self.plot_refresh_rate or \
           current_time - self.last_frame_time > self.replay_refresh_rate:
            print("\033c", end="")
            if current_time - self.last_visualization_time > self.plot_refresh_rate:
                self.update_plot()
                self.last_visualization_time = current_time
            print(self.plot_string)
            print('-' * self.terminal_width)
            print("Episode Replay:")
            if self.current_replay:
                if current_time - self.last_frame_time > self.replay_refresh_rate:  # Change frame every 0.1 second
                    self.update_replay()
                    self.last_frame_time = current_time
                print(self.current_replay[self.replay_index])
            self.progress_bar.update(self.timesteps - self.progress_bar_steps)
            self.progress_bar.refresh()
            self.progress_bar_steps = self.timesteps

    def update_replay(self):
        self.replay_index += 1
        if self.replay_index >= len(self.current_replay):
            self.replay_index = 0
            if self.pending_replay:
                self.current_replay = self.pending_replay
                self.pending_replay = []

    def update_plot(self):
        plt.clf()
        plt.clc()
        plt.theme("pro")
        plt.plot_size(width=self.plot_width, height=self.plot_height)
        if len(self.running_data) > 0:
            plt.plot(self.running_data["timestep"], self.running_data["reward"], label="Running reward", color="blue")
        if len(self.evaluation_data) > 0:
            plt.plot(self.evaluation_data["timestep"], self.evaluation_data["reward"], label="Evaluation reward", color="red")
        plt.title("Training Progress")
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        if len(self.running_data) > 0 or len(self.evaluation_data) > 0:
            self.plot_string = plt.build()

