import argparse

import gym
import minihack

import torch
import torch.nn as nn
import numpy as np

from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent
from runner import Runner
from loggers.file_logger import FileLogger
from utils.wrapper import MinihackWrapper

parser = argparse.ArgumentParser(description='Train a PPO agent on MiniHack')
parser.add_argument('--num-episodes', type=int, default=1000, metavar='N',)
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--log', type=str, default='logs/ppo_training.log')
parser.add_argument('--model', type=str, default=None)


def main(args):
    env = MinihackWrapper(gym.make("MiniHack-CorridorBattle-v0"))
    agent = PPOAgent(env.observation_space.spaces["glyphs"].shape, env.action_space.n)
    runner = Runner(
        env=env,
        agent=agent,
        num_episodes=args.num_episodes,
        loggers=[FileLogger(args.log)],
    )
    runner.train(render=args.render)
    runner.evaluate(render=args.render)
    runner = Runner(
        env=env,
        agent=RandomAgent(env.observation_space.spaces["glyphs"].shape, env.action_space.n),
        num_episodes=args.num_episodes,
        loggers=[FileLogger(args.log)],
    )
    runner.evaluate(render=args.render)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)