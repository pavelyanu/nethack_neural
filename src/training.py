import argparse
import cProfile

import gym
import minihack
import nle

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
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--log', type=str, default='logs/ppo_training.log')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--num_envs', type=int, default=16)
parser.add_argument('--total_steps', type=int, default=10000)
parser.add_argument('--worker_steps', type=int, default=1000)
parser.add_argument('--eval_step', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--hidden_layer', type=int, default=128)
parser.add_argument('--stable', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run on')
parser.add_argument('--save-path', type=str, default='models')


def main(args):
    if args.stable:
        run_stable(args)
    else:
        train_vectorized(args)

def train_vectorized(args):
    # env = gym.make("MiniHack-Room-Monster-15x15-v0", observation_keys=( 'glyphs', 'blstats' ))
    env = gym.make("NetHackScore-v0", observation_keys=( 'glyphs', 'blstats' ))
    env_specs = EnvSpecs()
    env_specs.init_with_gym_env(env, num_envs=args.num_envs)
    venv = gym.vector.make(
        env.spec.id,
        num_envs=args.num_envs,
        observation_keys=('glyphs', 'blstats'))
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
        evaluation_period=args.eval_step,
        render=args.render)
    


def run_stable(args):
    env = gym.make("NetHackScore-v0", observation_keys=( 'glyphs', 'blstats' ))
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)



if __name__ == '__main__':
    args = parser.parse_args()
    # main(args)
    if args.stable:
        cProfile.run('main(args)', filename='stable.prof')
    else:
        if args.device == 'cuda':
            torch.cuda.set_device(0)
            cProfile.run('main(args)', filename='ppo.cuda.prof')
        else:
            torch.cuda.set_device(-1)
            cProfile.run('main(args)', filename='ppo.cpu.prof')