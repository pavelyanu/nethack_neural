from nethack_neural.runners.abstract_runner import AbstractRunner
import tqdm

class PPORunner(AbstractRunner):
    """Runner class for Proximal Policy Optimization (PPO) algorithms.

    The PPORunner class extends the AbstractRunner base class. It is designed to handle the 
    interaction between a PPO agent and the environment, manage the training and evaluation processes, 
    and provide logging capabilities.

    Args:
        env (gym.Env): The gym environment in which to run the experiments.
        agent (AbstractPPOAgent): The PPO agent that will interact with the environment.
        loggers (list): A list of loggers for logging experiment results.
        use_tqdm (bool, optional): Whether to use tqdm to display a progress bar. Defaults to False.
    """
    def __init__(self, env, agent, loggers, use_tqdm=False):
        super().__init__(env, agent,loggers)
        self.progress_bar = None
        self.use_tqdm = use_tqdm

    def run(self, num_envs, eval_env, total_steps=10000, worker_steps=2000, evaluation_period=2000, evaluation_steps=10, render=False):
        """Runs the PPO training and evaluation loop.

        This method controls the interaction between the agent and the environment, collects the
        experiences, updates the agent's policy, and periodically evaluates the agent's performance.

        Args:
            num_envs (int): Number of environments to run in parallel.
            eval_env (gym.Env): The gym environment in which to evaluate the agent's performance.
            total_steps (int, optional): Total number of steps to run the training. Defaults to 10000.
            worker_steps (int, optional): Number of steps after which the policy will be updated. Defaults to 2000.
            evaluation_period (int, optional): Number of steps after which the agent's performance will be evaluated. Defaults to 2000.
            evaluation_steps (int, optional): Number of episodes to run for each evaluation. Defaults to 10.
            render (bool, optional): Whether to render the environment during evaluation. Defaults to False.
        """
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