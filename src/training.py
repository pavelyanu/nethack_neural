import argparse

import gym
import minihack

import torch
import torch.nn as nn
import numpy as np

from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent
from loggers.file_logger import FileLogger
from utils.wrapper import MinihackWrapper, MinihackTensorDictWrapper
from runners.ppo_runner import PPORunner
from runners.default_runner import Runner

parser = argparse.ArgumentParser(description='Train a PPO agent on MiniHack')
parser.add_argument('--num-episodes', type=int, default=1000, metavar='N',)
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--log', type=str, default='logs/ppo_training.log')
parser.add_argument('--model', type=str, default=None)


def main(args):
    env = MinihackWrapper(gym.make("MiniHack-River-MonsterLava-v0", observation_keys=( 'glyphs', 'blstats' )))
    # env = MinihackWrapper(gym.make("MiniHack-CorridorBattle-v0", observation_keys=( 'glyphs', 'blstats' )))
    observation_space = env.observation_space
    action_space = env.action_space
    agent = PPOAgent(
        observation_space=observation_space,
        action_space=action_space,
    )
    loggers = [FileLogger(args.log)]
    runner = PPORunner(env, agent, loggers)
    runner.run(num_episodes=args.num_episodes, render=args.render)
    loggers[0].log("Training complete.")
    random_agent = RandomAgent(
        observation_space=observation_space,
        action_space=action_space,
    )
    runner = Runner(env, random_agent, loggers)
    runner.evaluate(render=args.render)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)