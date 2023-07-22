import argparse

import gym
import minihack

import torch
import torch.nn as nn
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from agents.ppo_agent import GlyphBlstatsPPOAgent
from agents.random_agent import RandomAgent
from loggers.file_logger import FileLogger
from runners.ppo_runner import PPORunner
from runners.default_runner import Runner
from utils.env_specs import EnvSpecs

parser = argparse.ArgumentParser(description='Train a PPO agent on MiniHack')
parser.add_argument('--num-episodes', type=int, default=1000, metavar='N',)
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--log', type=str, default='logs/ppo_training.log')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--num_envs', type=int, default=2)
parser.add_argument('--stable', action='store_true', default=False)


def main(args):
    if args.stable:
        run_stable()
    if args.num_envs > 1:
        train_vectorized(args)
    else:
        train(args)

def train(args):
    env = gym.make("MiniHack-Room-Monster-15x15-v0", observation_keys=( 'glyphs', 'blstats' ))
    env_specs = EnvSpecs()
    env_specs.init_with_gym_env(env)
    agent = GlyphBlstatsPPOAgent(
        env_specs
    )
    loggers = [FileLogger(args.log)]
    runner = PPORunner(env, agent, loggers)
    runner.run(num_episodes=args.num_episodes, render=args.render)
    loggers[0].log("Training complete.")
    random_agent = RandomAgent(
        env_specs
    )
    runner = Runner(env, random_agent, loggers)
    runner.evaluate(render=args.render)

def train_vectorized(args):
    env = gym.make("MiniHack-Room-Monster-15x15-v0", observation_keys=( 'glyphs', 'blstats' ))
    env_specs = EnvSpecs()
    env_specs.init_with_gym_env(env, num_envs=args.num_envs)
    venv = gym.vector.make(
        env.spec.id,
        num_envs=args.num_envs,
        observation_keys=('glyphs', 'blstats'))
    agent = GlyphBlstatsPPOAgent(env_specs)
    loggers = [FileLogger(args.log)]
    runner = PPORunner(venv, agent, loggers)
    runner.run_vectorized(args.num_envs, env)


def run_stable():
    env = gym.make("MiniHack-Room-Monster-15x15-v0", observation_keys=( 'glyphs', 'blstats' ))
    # env = MinihackWrapper(gym.make("MiniHack-Room-Monster-15x15-v0", observation_keys=(['glyphs'])))
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)