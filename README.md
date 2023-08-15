# NetHack-Neural

## Problem Statement

This project aims to create a Proximal Policy Optimization (PPO) agent that can solve various environments based on the game NetHack. The program is developed using Python 3 and relies on popular libraries like PyTorch, Gym, Click, and MiniHack. The development environment consists of a combination of Jupyter notebooks for interactive development and VS Code for script editing and debugging. The program is compatible with Linux.

## Detailed Specification

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
The agent is developed using Python 3, with the use of libraries such as PyTorch for the implementation of the PPO algorithm, Gym and MiniHack for the environments, Click for the command-line interface, and pandas and tqdm for logging and visualization. The development environment includes Jupyter notebooks for interactive development and VS Code for script editing and debugging.

All the source code will be documented, and a user guide will be provided to help users set up and use the PPO_NetHack agent.

## Installation Manual

To install `nethack_neural`, simply use pip:

```sh
pip install nethack_neural
```

This will install the package and all its dependencies. 

## User's Guide

### Launching the Program

Once installed, you can launch the program from the command line using the command `nethack_neural`. By default, this will run the command with preset parameters.

You can also run specific commands, such as `nethack_neural specialized` or `nethack_neural fully_specialized`.

### Differences between Commands

The `nethack_neural command` (default run) will run the program using the default parameters specified in the script.

The `nethack_neural fully_specialized` command, on the other hand, will prompt the user to input all parameters manually, providing maximum customization for each run.

The `nethack_neural specialized` command provides a middle ground. It is a simplified version of the fully_specialized command, offering fewer customization options. These options focus on the environment, number of environments, total steps, worker steps, evaluation period, evaluation length, and visualization type.

### Parameters

Here are the parameters you can set for the `fully_specialized` command:

- `environment`: Choose an environment from the list or input a custom one.
- `observation_keys`: Choose observation keys. 'gb' for glyphs and blstats, 'g' for glyphs.
- `critic_lr`: Enter critic learning rate (default: 0.0003).
- `actor_lr`: Enter actor learning rate (default: 0.0003).
- `eps_clip`: Enter eps clip (default: 0.2).
- `hidden_layer_size`: Enter hidden layer size (default: 64).
- `num_envs`: Enter number of environments (default: 4).
- `total_steps`: Enter total steps (default: 100000).
- `worker_steps`: Enter worker steps (default: 1000).
- `evaluation_period`: Enter evaluation period (default: 500).
- `evaluation_length`: Enter evaluation length (default: 5).
- `batch_size`: Enter batch size (default: 64).
- `epochs`: Enter number of epochs (default: 10).
- `training_device`: Choose training device from 'cpu', 'gpu', or 'auto' (default: 'auto').
- `logger`: Choose logger type from 'stdout', 'file', or 'none' (default: 'none').
- `visualization`: Choose visualization type from 'none', 'full', or 'bar' (default: 'full').
- `save_model`: Choose whether to save the model (default: False).
- `load_model`: Choose whether to load an existing model (default: False).

### Visualization

The visualization option controls the display of the training process in the terminal. 

- 'none': No visualization.
- 'full': real-time reward graph with a progress bar and replay.
- 'bar': A progress bar only.

### Saving and Loading Models

With `save_model`, you can choose whether to save the trained model at the end of the training process. You will be prompted for a directory to save the model.

With `load_model`, you can choose whether to load a pre-trained model at the beginning of the training process. You will be prompted for a directory from which to load the model.

## Developer Documentation

(To be added)
