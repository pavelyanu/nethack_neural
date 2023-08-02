import argparse
import cProfile

import gym
# import gymnasium as gym
import minihack
import nle

import torch
import torch.nn as nn
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from agents.ppo_agent import GlyphBlstatsPPOAgent, CartPolePPOAgent
from agents.random_agent import RandomAgent
from loggers.file_logger import FileLogger
from runners.ppo_runner import PPORunner
from runners.default_runner import Runner
from utils.env_specs import EnvSpecs

parser = argparse.ArgumentParser(description='Train a PPO agent on MiniHack')
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--log', type=str, default='logs/ppo_training.log')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--num_envs', type=int, default=1)
parser.add_argument('--total_steps', type=int, default=10000)
parser.add_argument('--worker_steps', type=int, default=2000)
parser.add_argument('--evaluation_period', type=int, default=5000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--hidden_layer', type=int, default=64)
parser.add_argument('--stable', action='store_true', default=False)
parser.add_argument('--profile', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run on')
parser.add_argument('--save-path', type=str, default='models')


def main(args):
    if args.stable:
        run_stable(args)
    else:
        train_vectorized(args)

def _train_vectorized(args):
    env = gym.make("NetHackScore-v0", observation_keys=( 'glyphs', 'blstats' ))
    # env = gym.make("NetHackScore-v0", observation_keys=( ['glyphs']))
    env_specs = EnvSpecs()
    env_specs.init_with_gym_env(env, num_envs=args.num_envs)
    venv = gym.vector.make(
        env.spec.id,
        num_envs=args.num_envs,
        observation_keys=('glyphs', 'blstats'))
        # observation_keys=(['glyphs']))
    agent = GlyphBlstatsPPOAgent(
        env_specs,
        batch_size=args.batch_size,
        buffer_size=args.worker_steps,
        storage_device=args.device,
        training_device=args.device,
        hidden_layer=args.hidden_layer,
    )
    loggers = [FileLogger(args.log)]
    runner = PPORunner(venv, agent, loggers)
    runner.run_vectorized(
        args.num_envs,
        env,
        total_steps=args.total_steps,
        worker_steps=args.worker_steps,
        evaluation_period=args.evaluation_period,
        render=args.render)
    agent.save(args.save_path)

def train_vectorized(args):
    env = gym.make("CartPole-v1")
    env_specs = EnvSpecs()
    env_specs.init_with_gym_env(env, num_envs=args.num_envs)
    venv = gym.vector.make(
        env.spec.id,
        num_envs=args.num_envs,
    )
    agent = CartPolePPOAgent(
        env_specs,
        batch_size=args.batch_size,
        buffer_size=args.worker_steps,
        storage_device=args.device,
        training_device=args.device,
        hidden_layer=args.hidden_layer,
    )
    loggers = [FileLogger(args.log)]
    runner = PPORunner(venv, agent, loggers)
    runner.run_vectorized(
        args.num_envs,
        env,
        total_steps=args.total_steps,
        worker_steps=args.worker_steps,
        evaluation_period=args.evaluation_period,
        render=args.render)
    agent.save(args.save_path)

def run_stable(args):
    env = gym.make("CartPole-v1")
    # env = gym.make("NetHackScore-v0", observation_keys=( 'glyphs', 'blstats' ))
    model = PPO("MlpPolicy", env, verbose=1, n_steps=args.worker_steps, batch_size=args.batch_size)
    # model = PPO("MultiInputPolicy", env, verbose=1, n_steps=args.worker_steps, batch_size=args.batch_size)
    model.learn(total_timesteps=args.total_steps)
    model.save(args.save_path)



if __name__ == '__main__':
    args = parser.parse_args()
    if args.profile:
        if args.stable:
            cProfile.run('main(args)', filename='stable.prof')
        else:
            if args.device == 'cuda':
                torch.cuda.set_device(0)
                cProfile.run('main(args)', filename='ppo.cuda.prof')
            else:
                torch.cuda.set_device(-1)
                cProfile.run('main(args)', filename='ppo.cpu.prof')
    else:
        main(args) 