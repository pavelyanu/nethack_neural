import gym
import nle
import torch
import torchrl

def test_environment():
    env = gym.make("NetHackScore-v0")
    env.reset()  # each reset generates a new dungeon
    env.step(1)  # move agent '@' north

if __name__ == "__main__":
    test_environment()
