import os
import datetime
import yaml
import re
import subprocess
import tempfile
import sys
import curses

from nethack_neural.agents.ppo_agent import GlyphBlstatsPPOAgent, GlyphPPOAgent, BlstatPPOAgent
from nethack_neural.runners.ppo_runner import PPORunner
from nethack_neural.runners.ppo_visual_runner import PPOVisualRunner
from nethack_neural.loggers.file_logger import FileLogger
from nethack_neural.loggers.stdout_logger import StdoutLogger
from nethack_neural.utils.env_specs import EnvSpecs

import click
import minihack
import gym
import torch

from simple_term_menu import TerminalMenu

"""This module uses the following configuration format (here is the default configuration):

actor_lr: 0.0003
batch_size: 64
critic_lr: 0.0003
environment: MiniHack-Room-5x5-v0
epochs: 10
eps_clip: 0.2
evaluation_length: 5
evaluation_period: 500
gae_lambda: 0.95
gamma: 0.99
hidden_layer_size: 64
load_model: null
logger: none
num_envs: 4
observation_keys:
- glyphs
- blstats
save_model: null
total_steps: 100000
training_device: auto
visualization: full
worker_steps: 1000
"""

available_vis = [
    'none',
    'bar',
    'full'
]

vis_to_runners = {
    'none': PPORunner,
    'bar': PPORunner,
    'full': PPOVisualRunner,
}

vis_expansions = {
    'none': 'No visualization',
    'bar': 'Only tqdm progress bar with mean of 100 latest rewards',
    'full': 'Full visualization. Includes tqdm progress bar, plot of rewards, and last evaluation episode running in the terminal.',
}

available_keys = [
    'glyphs',
    'blstats',
]

key_explanations = {
    'blstats': "The 'blstats' are the player's stats, such as HP, strength, etc.",
    'glyphs': "The 'glyphs' are the map tiles around the player.",
}

keys_to_heads = {
    'glyphs': GlyphPPOAgent,
    'blstats': BlstatPPOAgent,
    'glyphs+blstats': GlyphBlstatsPPOAgent,
}

training_parameters = [
    'num_envs',
    'total_steps',
    'worker_steps',
    'evaluation_period',
    'evaluation_length',
]

training_parameters_to_types = {
    'num_envs': int,
    'total_steps': int,
    'worker_steps': int,
    'evaluation_period': int,
    'evaluation_length': int,
}

training_parameters_to_explanations = {
    'num_envs': "The number of environments to run in parallel.",
    'total_steps': "The total number of steps to train for.",
    'worker_steps': "The number of steps to run in each environment before updating the model.",
    'evaluation_period': "The number of steps between evaluations.",
    'evaluation_length': "The number of episodes to run during each evaluation.",
}

agent_parameters = [
    'critic_lr',
    'actor_lr',
    'gamma',
    'gae_lambda',
    'eps_clip',
    'hidden_layer_size',
    'batch_size',
    'epochs',
]

agent_parameters_to_types = {
    'critic_lr': float,
    'actor_lr': float,
    'gamma': float,
    'gae_lambda': float,
    'eps_clip': float,
    'hidden_layer_size': int,
    'batch_size': int,
    'epochs': int,
}

agent_parameters_to_explanations = {
    'critic_lr': "The learning rate for the critic.",
    'actor_lr': "The learning rate for the actor.",
    'gamma': "The discount factor.",
    'gae_lambda': "The lambda parameter for generalized advantage estimation.",
    'eps_clip': "The epsilon parameter for PPO.",
    'hidden_layer_size': "The size of the hidden layer.",
    'batch_size': "The batch size for training.",
    'epochs': "The number of epochs to train for.",
}

available_loggers = [
    'stdout',
    'file',
]

logger_explanations = {
    'stdout': "Log to the terminal.",
    'file': "Log to a file.",
}

loggers_to_classes = {
    'stdout': StdoutLogger,
    'file': FileLogger,
}


def get_project_root():
    """Get the root path of the project."""
    current_path = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(current_path))

def get_config_path():
    """Get the path to the run_configs directory."""
    return os.path.join(get_project_root(), 'run_configs')

def get_default_config():
    """Load the default configuration from the YAML file."""
    with open(os.path.join(get_config_path(), 'default.yaml'), 'r') as f:
        default_config = yaml.safe_load(f)
    return default_config

def load_environments():
    """Load environments from the YAML configuration and validate each environment name."""
    with open(os.path.join(get_project_root(), 'environments.yaml'), 'r') as f:
        envs = yaml.safe_load(f)

    for env_type, env_list in envs.items():
        for idx, env_name in enumerate(env_list):
            env_code = env_name.split(" ")[0]
            try:
                gym.make(env_code)
            except gym.error.NameNotFound as e:
                match = re.search(r'Did you mean: `(.*?)`', str(e))
                if match:
                    suggested_name = match.group(1) + "-v0"
                    env_description = env_name[len(env_code):].strip()
                    envs[env_type][idx] = suggested_name + " " + env_description
    return envs

def preview_environment(env_name):
    """Render a preview of the environment using Gym."""
    env = gym.make(env_name)
    env.reset()
    observation = env.render(mode="ansi")
    return observation


def pretty_list_print(lst):
    """Format a list into a pretty string."""
    return "\n".join(lst)


def file_browser(stdscr, start_directory='.', to_choose='f'):
    curses.curs_set(0)
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK) 
    curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)

    current_directory = start_directory
    selected_idx = 0

    while True:
        stdscr.clear()
        
        entries = ['..'] + os.listdir(current_directory)
        h, w = stdscr.getmaxyx()

        stdscr.addstr(0, 0, f"Current directory: {current_directory}".ljust(w), curses.A_BOLD)
        
        for i, entry in enumerate(entries):
            full_path = os.path.join(current_directory, entry)
            if os.path.isdir(full_path):
                if i == selected_idx:
                    stdscr.addstr(i + 2, 0, entry.ljust(w), curses.color_pair(1) | curses.A_BOLD)
                else:
                    stdscr.addstr(i + 2, 0, entry.ljust(w), curses.color_pair(1))
            else:
                if i == selected_idx:
                    stdscr.addstr(i + 2, 0, entry.ljust(w), curses.color_pair(2) | curses.A_BOLD)
                else:
                    stdscr.addstr(i + 2, 0, entry.ljust(w), curses.color_pair(2))
        
        instructions = "Arrows or vim keys to navigate, Enter to choose, Right arrow or l to enter directory, Left arrow or h to go up."
        stdscr.addstr(h-2, 0, instructions.ljust(w), curses.A_DIM)
        
        key = stdscr.getch()
        
        if key in [curses.KEY_UP, ord('k')] and selected_idx > 0:
            selected_idx -= 1
        elif key in [curses.KEY_DOWN, ord('j')] and selected_idx < len(entries) - 1:
            selected_idx += 1
        elif key == curses.KEY_RIGHT or key == ord('l'):
            chosen_entry = entries[selected_idx]
            full_path = os.path.join(current_directory, chosen_entry)
            if os.path.isdir(full_path):
                current_directory = full_path
                selected_idx = 0
        elif key == curses.KEY_LEFT or key == ord('h'):
            current_directory = os.path.dirname(current_directory)
            selected_idx = 0
        elif key in [curses.KEY_ENTER, 10, 13]:
            chosen_entry = entries[selected_idx]
            full_path = os.path.join(current_directory, chosen_entry)
            if os.path.isdir(full_path) and to_choose == 'd':
                return full_path
            elif os.path.isfile(full_path) and to_choose == 'f':
                return full_path
        elif key == ord('q'):
            return None

def choose_environment(config: dict):
    environments = load_environments()

    preview_command = lambda env_name: pretty_list_print(environments[env_name])
    
    while True:
        types_menu = TerminalMenu(
            list(environments.keys()),
            title="Choose an environment type",
            preview_command=preview_command,
            preview_size=0.5,
            clear_screen=True,
            show_search_hint=True,
            cycle_cursor=True
        )
        
        env_type = types_menu.show()
        if env_type is None:
            return
        
        env_type_name = list(environments.keys())[env_type]

        envs_list = [env.split(" ")[0] for env in environments[env_type_name]]
        specific_menu = TerminalMenu(
            envs_list,
            title=f"Choose a {env_type_name} environment",
            preview_command=preview_environment,
            preview_size=0.5,
        )
        
        chosen_env = specific_menu.show()
        if chosen_env is not None:
            config['env_name'] = environments[env_type_name][chosen_env]
            return

def choose_observation_keys(config: dict):
    """Choose the observation keys for the environment."""
    observation_menu = TerminalMenu(
        available_keys,
        title="Choose the observation keys",
        clear_screen=True,
        cycle_cursor=True,
        multi_select=True,
        show_multi_select_hint=True,
        preview_command=lambda key: key_explanations[key],
    )
    keys = observation_menu.show()
    if keys is None:
        return
    keys = [available_keys[key] for key in keys]
    config['observation_keys'] = keys

def choose_save_model(config):
    """Choose whether to save the model."""
    save_menu = TerminalMenu(
        ["No", "Yes"],
        title="Save the model?",
        clear_screen=True,
        cycle_cursor=True,
        multi_select=False,
        show_multi_select_hint=False,
    )
    save = save_menu.show()
    if save is None:
        return
    save = bool(save)
    if not save:
        config['save_model'] = None
        return
    model_save_path = curses.wrapper(file_browser, to_choose='d')
    if model_save_path is None:
        print("No model save path chosen. The model will not be saved.")
        config['save_model'] = None
    config['save_model'] = model_save_path

def choose_load_model(config: dict):
    """Choose whether to load a model."""
    load_menu = TerminalMenu(
        ["No", "Yes"],
        title="Load a model?",
        clear_screen=True,
        cycle_cursor=True,
        multi_select=False,
        show_multi_select_hint=False,
    )
    load = load_menu.show()
    if load is None:
        return
    load = bool(load)
    if not load:
        config['load_model'] = None
        return
    model_load_path = curses.wrapper(file_browser, to_choose='f')
    if model_load_path is None:
        print("No model load path chosen. The model will not be loaded.")
        config['load_model'] = None
    config['load_model'] = model_load_path

def choose_training_parameters(config):
    """Choose the training parameters."""
    parameter_menu = TerminalMenu(
        ["No", "Yes"],
        title="Choose training parameters?",
        clear_screen=True,
        cycle_cursor=True,
        multi_select=False,
        show_multi_select_hint=False,
    )
    choose_parameters = parameter_menu.show()
    choose = bool(choose_parameters)
    if not choose:
        return
    for parameter in training_parameters:
        parameter_type = training_parameters_to_types[parameter]
        parameter_explanation = training_parameters_to_explanations[parameter]
        parameter_value = click.prompt(parameter_explanation, type=parameter_type, default=config[parameter])
        config[parameter] = parameter_value

def choose_agent_parameters(config):
    """Choose the agent parameters."""
    parameter_menu = TerminalMenu(
        ["No", "Yes"],
        title="Choose agent parameters?",
        clear_screen=True,
        cycle_cursor=True,
        multi_select=False,
        show_multi_select_hint=False,
    )
    choose_parameters = parameter_menu.show()
    choose = bool(choose_parameters)
    if not choose:
        return
    for parameter in agent_parameters:
        parameter_type = agent_parameters_to_types[parameter]
        parameter_explanation = agent_parameters_to_explanations[parameter]
        parameter_value = click.prompt(parameter_explanation, type=parameter_type, default=config[parameter])
        config[parameter] = parameter_value

def choose_loggers(config: dict):
    """Choose the loggers for the application."""
    loggers = []
    logger_paths = set()

    # Ask if the user wants to add loggers
    add_logger_menu = TerminalMenu(
        ["No", "Yes"],
        title="Do you want to add loggers?",
        clear_screen=True,
        cycle_cursor=True,
        multi_select=False
    )
    add_logger = add_logger_menu.show()
    if add_logger is None or add_logger == 0:
        return
    stdout_chosen = False
    while True:
        # Select type of logger
        logger_type_menu = TerminalMenu(
            ["File Logger", "Stdout Logger"],
            title="Choose a logger type",
            clear_screen=True,
            cycle_cursor=True,
            multi_select=False
        )
        logger_type = logger_type_menu.show()
        if logger_type is None:
            break
        # Configure chosen logger
        if logger_type == 0:  # File Logger
            # Ensure unique paths
            while True:
                message_path = click.prompt("Enter path to log messages")
                stats_path = click.prompt("Enter path to log stats")
                if message_path not in logger_paths and stats_path not in logger_paths:
                    logger_paths.update([message_path, stats_path])
                    break
                print("Error: Paths must be unique. Please enter different paths.")
            
            # Ask about plotting options
            plot_option_menu = TerminalMenu(
                ["None", "Save", "Show"],
                title="Plotting options for File Logger",
                clear_screen=True,
                cycle_cursor=True
            )
            plot_option = plot_option_menu.show()
            if plot_option == 1:
                loggers.append({"type": "FileLogger", "message_path": message_path, "stats_path": stats_path, "plot_option": "save"})
            elif plot_option == 2:
                loggers.append({"type": "FileLogger", "message_path": message_path, "stats_path": stats_path, "plot_option": "show"})
            else:
                loggers.append({"type": "FileLogger", "message_path": message_path, "stats_path": stats_path})

        elif logger_type == 1 and not stdout_chosen:  # Stdout Logger
            loggers.append({"type": "StdoutLogger"})
            stdout_chosen = True
        elif stdout_chosen:
            print("You can have only one StdoutLogger.")

        # Ask if the user wants to add another logger
        add_another_logger_menu = TerminalMenu(
            ["No", "Yes"],
            title="Do you want to add another logger?",
            clear_screen=True,
            cycle_cursor=True
        )
        add_another_logger = add_another_logger_menu.show()
        if add_another_logger == 0:
            break

    config["loggers"] = loggers

def choose_config():
    """Choose a configuration file to run."""
    config_menu = TerminalMenu(
        ["Pre-generated", "Custom"],
        title="Choose a configuration",
        clear_screen=True,
        cycle_cursor=True,
        multi_select=False,
        show_multi_select_hint=False,
    )
    config_type = config_menu.show()
    if config_type is None:
        return get_default_config()
    custom = bool(config_type)
    if custom:
        message = "Choose a base configuration to modify"
    else:
        message = "Choose a configuration to run"
    config_files = os.listdir(get_config_path())
    config_files = [config_file for config_file in config_files if config_file.endswith(".yaml")]
    config_menu = TerminalMenu(
        config_files,
        title=message,
        clear_screen=True,
        cycle_cursor=True,
        multi_select=False,
        show_multi_select_hint=False,
    )
    config_file = config_menu.show()
    if config_file is None:
        print("No configuration file chosen. Using default configuration.")
        config = get_default_config()
    else:
        config_file = config_files[config_file]
        with open(os.path.join(get_config_path(), config_file), 'r') as f:
            config = yaml.safe_load(f)
    if not custom:
        return config
    choose_save_model(config)
    choose_load_model(config)
    choose_environment(config)
    choose_observation_keys(config)
    choose_training_parameters(config)
    choose_agent_parameters(config)
    save_menu = TerminalMenu(
        ["No", "Yes"],
        title="Save the configuration?",
        clear_screen=True,
        cycle_cursor=True,
        multi_select=False,
        show_multi_select_hint=False,
    )
    save = save_menu.show()
    if save is None:
        return config
    save = bool(save)
    if not save:
        return config
    while True:
        config_name = click.prompt("Enter a name for the configuration")
        if config_name.endswith(".yaml"):
            config_name = config_name[:-5]
        if config_name == 'default':
            print("The name 'default' is reserved for the default configuration. Please choose another name.")
            continue
        config_path = os.path.join(get_config_path(), config_name + ".yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        print(f"Saved configuration to {config_path}")
        return config

if __name__ == "__main__":
    config = choose_config()
    print(config)