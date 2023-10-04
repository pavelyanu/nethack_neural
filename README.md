# NetHack-Neural

# Problem Statement

This project aims to create a Proximal Policy Optimization (PPO) agent that can solve various environments based on the game NetHack. The program is developed using Python 3 and relies on popular libraries like PyTorch, Gym, Click, and MiniHack. The development environment consists of VS Code for script editing and debugging. The program is compatible with Linux.

# Detailed Specification

### Analysis of Existing Programs
There are several reinforcement learning agents available that can handle gym-like environments. However, the PPO_NetHack agent aims to specialize in the NetHack-based environments, providing a robust framework for training, evaluating, and visualizing the agent's progress in a variety of NetHack scenarios.

### Description of Intended Functionality
The PPO_NetHack agent will have the following key features:

1. **Versatility:** The agent will be able to handle different NetHack environments, allowing users to train the agent on a wide variety of tasks.
2. **Modular Design:** The project will be structured in a modular way, separating the agent, environment handling, logging, and command-line interface into different modules.
3. **Logging:** The agent will provide logging functionality, allowing users to track the training progress and performance of the agent.
4. **Visualization:** The agent will provide a real-time visualization of the training progress and the agent's performance in the environment.
5. **Command-line Interface:** The agent will provide a command-line interface for easy configuration of the training process.

### Structure of the Program
The program will be divided into several modules:

1. **Agent module:** This module will contain the implementation of the PPO agent. It will handle the policy network, value network, and the training process.
2. **Environment module:** This module will handle the interaction with the NetHack environments, including setting up the environments and translating the environments' states to the agent.
3. **Logger module:** This module will provide logging functionality, writing detailed logs of the agent's training progress and performance.
4. **Command-line Interface module:** This module will provide a command-line interface for easy configuration of the training process.
5. **Runner module:** This module will control the overall training process, coordinating the agent, environments, and logger.

### Development Environment
The agent is developed using Python 3, with the use of libraries such as PyTorch for the implementation of the PPO algorithm, Gym and MiniHack for the environments, Click for the command-line interface, and pandas and tqdm for logging and visualization. The development environment consists of VS Code for script editing and debugging.

All the source code will be documented, and a user guide will be provided to help users set up and use the PPO_NetHack agent.

# Installation Manual

I'm still figuring out a portable way to install the program. For now, you can clone the repository and run `python main.py` in the terminal to start the program.

## Dependencies

In addition to Python 3.x and libraries listed in requirements.txt, the program requires the following due to Nethack Learning Environment: [ NLE Installation ]( https://github.com/facebookresearch/nle#installation )

# User's Guide

## Overview

NetHack-Neural is a Python-based tool designed to train and evaluate Proximal Policy Optimization (PPO) agents in environments based on the game NetHack. Utilizing libraries like PyTorch, Gym, Click, and MiniHack, the tool provides a robust framework for conducting experiments in Reinforcement Learning (RL).

## Features

- **Versatile Environments**: Supports multiple NetHack-based environments.
- **Modular Design**: Segregates agent, environment, logging, and CLI into separate modules.
- **Logging**: Comprehensive logging to track training and performance.
- **Visualization**: Real-time training progress and performance visualization.
- **Customization**: CLI for easy configuration of training, logging, and environment settings.

## Basic Usage

Run `python main.py` in the terminal to start the program. You will be prompted to configure various settings, including:

- Environment type and name
- Observation keys
- Training parameters
- Agent parameters
- Logger type and location
- Training and storage devices
- Visualization type

## Configuration Files

NetHack-Neural uses YAML-based configuration files stored in the `run_configs` directory. Predefined configurations can be used, or custom ones can be saved for later use.

### Example Configuration

```yaml
actor_lr: 0.0003
batch_size: 64
critic_lr: 0.0003
env_name: MiniHack-Room-5x5-v0
epochs: 10
eps_clip: 0.2
evaluation_length: 5
evaluation_period: 500
gae_lambda: 0.95
gamma: 0.99
hidden_layer_size: 64
load_model: null
loggers: []
num_envs: 4
observation_keys:
- glyphs
- blstats
save_model: null
total_steps: 100000
training_device: auto
visualization: full
worker_steps: 1000
```

## Command-line Interface (CLI)

The CLI provided by Click and simple-term-menu allows for easy navigation and configuration. Follow the prompts to set up your experiment.

## Logging

You can choose between terminal-based (`stdout`) and file-based logging. File-based logging can also generate plots to visualize the agent's performance over time.

## Advanced Usage

### Custom Environments

NetHack-Neural supports the addition of custom Gym-compatible environments. These can be added to the `environments.yaml` file.

### Custom Agents

While the tool comes with predefined PPO agents, you can integrate custom agents by inheriting from the base agent class and implementing the required methods.

## Technical Details

### Agent Types

- `GlyphPPOAgent`: Uses only map tiles (glyphs) for observations.
- `BlstatPPOAgent`: Uses only player stats (blstats) for observations.
- `GlyphBlstatsPPOAgent`: Uses both glyphs and blstats for observations.

### Logger Types

- `StdoutLogger`: Logs to the terminal.
- `FileLogger`: Logs to a specified file and optionally generates performance plots.

### Training Device Options

- `auto`: Automatically selects between CPU and GPU based on availability.
- `cpu`: Forces CPU usage.
- `gpu`: Forces GPU usage (if available).

### Visualization Types

- `none`: No visualization.
- `bar`: Displays a tqdm progress bar.
- `full`: Displays tqdm progress bar, rewards plot, and last evaluation episode in the terminal.

## Developer Documentation

(To be added)
