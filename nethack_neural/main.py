import os
import zipfile
import datetime
from time import sleep
import yaml
import re
import subprocess
import tempfile
import sys
import curses

from nethack_neural.agents.ppo_agent import GlyphBlstatsPPOAgent, GlyphPPOAgent, BlstatPPOAgent
from nethack_neural.runners.ppo_full_runner import PPOFullRunner
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
    'buffer_size',
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
    'buffer_size': int,
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
    'buffer_size': "The size of the replay buffer.",
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

def echo(msg):
    """Print a message to the terminal. And sleep for 1 second."""
    click.echo(msg)
    sleep(1)

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
    env_name = env_name.split(" ")[0]
    env = gym.make(env_name)
    env.reset()
    observation = env.render(mode="ansi")
    return observation

def pretty_list_print(lst):
    """Format a list into a pretty string."""
    return "\n".join(lst)

def yes_no_menu(title: str):
    """A yes/no menu."""
    yes_no_menu = TerminalMenu(
        ["No", "Yes"],
        title=title,
        clear_screen=True,
        cycle_cursor=True,
        multi_select=False,
        show_multi_select_hint=False,
    )
    choice = yes_no_menu.show()
    if choice is None:
        return
    return bool(choice)

def file_browser(stdscr, start_directory='.', to_choose='f'):
    """Curses file browser."""
    curses.curs_set(0)
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_WHITE)

    current_directory = start_directory
    selected_idx = 0

    while True:
        stdscr.clear()
        
        entries = ['..'] + os.listdir(current_directory)
        h, w = stdscr.getmaxyx()

        stdscr.addstr(0, 0, f"Current directory: {current_directory}".ljust(w), curses.A_BOLD)
        
        for i, entry in enumerate(entries):
            full_path = os.path.join(current_directory, entry)
            color = curses.color_pair(3) if i == selected_idx else curses.color_pair(1) if os.path.isdir(full_path) else curses.color_pair(2)
            bold_attr = curses.A_BOLD if i == selected_idx else 0
            stdscr.addstr(i + 2, 0, entry.ljust(w), color | bold_attr)
        
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
    """Choose an environment."""
    current = config.get('env_name', 'MiniHack-Room-5x5-v0')
    message = f"Do you want to choose an environment? Currently the environment is {current}."
    choose_env = yes_no_menu(message)
    if choose_env is None or not choose_env:
        return

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

        envs_list = environments[env_type_name]
        specific_menu = TerminalMenu(
            envs_list,
            title=f"Choose a {env_type_name} environment",
            preview_command=preview_environment,
            preview_size=0.5,
        )
        
        chosen_env = specific_menu.show()
        if chosen_env is not None:
            config['env_name'] = environments[env_type_name][chosen_env].split(" ")[0]
            return

def choose_observation_keys(config: dict):
    """Choose the observation keys for the environment."""
    current = config.get('observation_keys', ['glyphs', 'blstats'])
    message = f"Do you want to choose observation keys? Currently the observation keys are {current}."
    choose_keys = yes_no_menu(message)
    if choose_keys is None or not choose_keys:
        return
    observation_menu = TerminalMenu(
        available_keys,
        title="Choose observation keys",
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
    save = yes_no_menu("Do you want to save the model?")
    if save is None:
        return
    if not save:
        config['save_model'] = None
        return
    model_save_path = curses.wrapper(file_browser, to_choose='d')
    if model_save_path is None:
        echo("No model save path chosen. The model will not be saved.")
        config['save_model'] = None
    config['save_model'] = model_save_path

def choose_load_model(config: dict):
    """Choose whether to load a model."""
    load = yes_no_menu("Do you want to load a model?")
    if load is None:
        return
    if not load:
        config['load_model'] = None
        return
    model_load_path = curses.wrapper(file_browser, to_choose='f')
    if model_load_path is None:
        echo("No model load path chosen. The model will not be loaded.")
        config['load_model'] = None
    config['load_model'] = model_load_path

def choose_training_parameters(config):
    """Choose the training parameters."""
    current = [f"{parameter}: {config[parameter]}" for parameter in training_parameters]
    message = f"Do you want to choose training parameters? Currently the training parameters are:\n{pretty_list_print(current)}"
    choose = yes_no_menu(message)
    if choose is None or not choose:
        return
    for parameter in training_parameters:
        parameter_type = training_parameters_to_types[parameter]
        parameter_explanation = training_parameters_to_explanations[parameter]
        parameter_value = click.prompt(parameter_explanation, type=parameter_type, default=config[parameter])
        config[parameter] = parameter_value

def choose_agent_parameters(config):
    """Choose the agent parameters."""
    current = [f"{parameter}: {config[parameter]}" for parameter in agent_parameters]
    message = f"Do you want to choose agent parameters? Currently the agent parameters are:\n{pretty_list_print(current)}"
    choose = yes_no_menu(message)
    if choose is None or not choose:
        return
    for parameter in agent_parameters:
        parameter_type = agent_parameters_to_types[parameter]
        parameter_explanation = agent_parameters_to_explanations[parameter]
        parameter_value = click.prompt(parameter_explanation, type=parameter_type, default=config[parameter])
        config[parameter] = parameter_value

def choose_loggers(config: dict):
    """Choose the loggers for the application."""
    loggers = config.get("loggers", [])
    logger_paths = set([logger.get("message_path", "") for logger in loggers] + 
                     [logger.get("stats_path", "") for logger in loggers])
    
    stdout_chosen = any([logger.get("type") == "StdoutLogger" for logger in loggers])
    
    while True:
        current_loggers = "\n".join([f"{logger['type']} ({logger.get('message_path', '')} {logger.get('stats_path', '')})" for logger in loggers])
        message = f"Do you want to choose loggers? Currently the loggers are:\n{current_loggers}"
        add = yes_no_menu(message)
        if add is None or not add:
            break

        logger_type_menu = TerminalMenu(
            ["File Logger", "Stdout Logger"],
            title="Choose a logger type",
            clear_screen=True,
            cycle_cursor=True,
            multi_select=False,
        )
        logger_type = logger_type_menu.show()
        if logger_type is None:
            break

        if logger_type == 0:  # FileLogger
            while True:
                message_path = click.prompt("Enter path to log messages")
                stats_path = click.prompt("Enter path to log stats")
                if message_path not in logger_paths and stats_path not in logger_paths:
                    logger_paths.update([message_path, stats_path])
                    break
                echo("Error: Paths must be unique. Please enter different paths.")
            
            plot_option_menu = TerminalMenu(
                ["None", "Save", "Show"],
                title="Plotting options for File Logger",
                clear_screen=True,
                cycle_cursor=True,
                multi_select=True,
                show_multi_select_hint=True,
            )
            plot_option = plot_option_menu.show()
            save_plot = 1 in plot_option
            show_plot = 2 in plot_option
            loggers.append({"type": "file", "message_path": message_path, "stats_path": stats_path, "save_plot": save_plot, "show_plot": show_plot})

        elif logger_type == 1 and not stdout_chosen:  # StdoutLogger
            loggers.append({"type": "stdout"})
            stdout_chosen = True
        elif stdout_chosen:
            echo("You can have only one stdout logger.")

    config["loggers"] = loggers

def choose_training_device(config):
    """Choose the device to run on."""
    current = config.get('training_device', 'auto')
    message = f"Do you want to choose the device to run on? Currently the device is {current}."
    choose = yes_no_menu(message)
    if choose is None or not choose:
        return
    gpu_available = torch.cuda.is_available()
    options = ["auto"] + (["cpu", "gpu"] if gpu_available else ["cpu"])
    device_menu = TerminalMenu(
        options,
        title="Choose a device",
        clear_screen=True,
        cycle_cursor=True,
        multi_select=False,
        show_multi_select_hint=False,
    ) 
    device = device_menu.show()
    if device is None:
        return
    config['training_device'] = options[device]

def choose_storage_device(config):
    """Choose the device to store the model on."""
    current = config.get('storage_device', 'auto')
    message = f"Do you want to choose the device to store the data on? Currently the device is {current}."
    choose = yes_no_menu(message)
    if choose is None or not choose:
        return
    gpu_available = torch.cuda.is_available()
    options = ["auto"] + (["cpu", "gpu"] if gpu_available else ["cpu"])
    device_menu = TerminalMenu(
        options,
        title="Choose a device",
        clear_screen=True,
        cycle_cursor=True,
        multi_select=False,
        show_multi_select_hint=False,
    ) 
    device = device_menu.show()
    if device is None:
        return
    config['storage_device'] = options[device]

def choose_visualization(config):
    """Choose the visualization type."""
    current = config.get('visualization', 'full')
    message = f"Do you want to choose the visualization type? Currently the visualization type is {current}."
    choose = yes_no_menu(message)
    if choose is None or not choose:
        return
    visualization_menu = TerminalMenu(
        available_vis,
        title="Choose a visualization type",
        clear_screen=True,
        cycle_cursor=True,
        multi_select=False,
        show_multi_select_hint=False,
        preview_command=lambda vis: vis_expansions[vis],
    )
    visualization = visualization_menu.show()
    if visualization is None:
        return
    config['visualization'] = available_vis[visualization]

def choose_config():
    """Choose a configuration file to run."""
    while True:
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
            echo("No configuration chosen. Exiting.")
            sys.exit(0)
        custom = bool(config_type)
        if custom:
            message = "Choose a base configuration to modify"
        else:
            message = "Choose a configuration to run"
        config_files = os.listdir(get_config_path())
        config_files = [config_file for config_file in config_files if config_file.endswith(".yaml")]
        config_preview = lambda config_file: pretty_list_print(map(lambda x: str(x).strip(), open(os.path.join(get_config_path(), config_file), 'r').readlines()))
        config_menu = TerminalMenu(
            config_files,
            title=message,
            clear_screen=True,
            cycle_cursor=True,
            multi_select=False,
            show_multi_select_hint=False,
            preview_command=config_preview,
            preview_size=1,
        )
        config_file = config_menu.show()
        if config_file is None:
            continue
        else:
            config_file = config_files[config_file]
            with open(os.path.join(get_config_path(), config_file), 'r') as f:
                config = yaml.safe_load(f)
            break
    choose_environment(config)
    choose_visualization(config)
    if not custom:
        return config
    choose_save_model(config)
    choose_load_model(config)
    choose_loggers(config)
    choose_training_device(config)
    choose_storage_device(config)
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
            echo("The name 'default' is reserved for the default configuration. Please choose another name.")
            continue
        config_path = os.path.join(get_config_path(), config_name + ".yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        echo(f"Saved configuration to {config_path}")
        return config

class Run:
    def __init__(self, config: dict):
        self.config = config
        self.env_name = config['env_name'] # done
        self.observation_keys = config['observation_keys'] # done
        self.save_model = config['save_model']
        self.load_model = config['load_model']
        self.loggers = config['loggers'] # done
        self.training_device = config['training_device'] # done
        self.storage_device = config['storage_device'] # done
        self.num_envs = config['num_envs'] 
        self.total_steps = config['total_steps']
        self.worker_steps = config['worker_steps']
        self.evaluation_period = config['evaluation_period']
        self.evaluation_length = config['evaluation_length']
        self.critic_lr = config['critic_lr'] # done
        self.actor_lr = config['actor_lr'] # done
        self.gamma = config['gamma'] # done
        self.gae_lambda = config['gae_lambda'] # done
        self.eps_clip = config['eps_clip'] # done
        self.hidden_layer_size = config['hidden_layer_size'] # done
        self.batch_size = config['batch_size'] # done
        self.epochs = config['epochs'] # done
        self.visualization = config['visualization'] # done
        self.buffer_size = config['buffer_size'] # done

    def init_loggers(self):
        """Initialize the loggers."""
        loggers = []
        for logger in self.loggers:
            logger_type = logger['type']
            logger_class = loggers_to_classes[logger_type]
            if logger_type == 'file':
                _logger = logger_class(
                    message_path=logger['message_path'],
                    stats_path=logger['stats_path'],
                    save_plot=logger['save_plot'],
                    show_plot=logger['show_plot'],
                )
            elif logger_type == 'stdout':
                _logger = logger_class()
            loggers.append({'logger': _logger, 'type': logger_type})
        self.loggers = loggers

    def init_devices(self):
        """Initialize the training device."""
        if self.training_device == 'auto':
            self.training_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            if self.training_device == 'gpu' and not torch.cuda.is_available():
                echo("Warning: GPU not available. Using CPU instead.")
                self.training_device = 'cpu'
        if self.storage_device == 'auto':
            self.storage_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            if self.storage_device == 'gpu' and not torch.cuda.is_available():
                echo("Warning: GPU not available. Using CPU instead.")
                self.storage_device = 'cpu'
        if self.training_device != self.storage_device:
           echo("Warning: Training device and storage device are different. This may cause performance issues.")

    def init_env(self):
        self.env_specs = EnvSpecs()
        self.eval_env = gym.make(self.env_name, observation_keys=self.observation_keys)
        self.env_specs.init_with_gym_env(self.eval_env, num_envs=self.num_envs)
        self.env = gym.vector.make(self.env_name, num_envs=self.num_envs, observation_keys=self.observation_keys)

    def init_agent(self):
        agent_class = keys_to_heads['+'.join(self.observation_keys)]
        self.agent = agent_class(
            env_specs=self.env_specs,
            training_device=self.training_device,
            storage_device=self.storage_device,
            critic_lr=self.critic_lr,
            actor_lr=self.actor_lr,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            eps_clip=self.eps_clip,
            hidden_layer=self.hidden_layer_size,
            batch_size=self.batch_size,
            epochs=self.epochs,
            buffer_size=self.buffer_size,
        )

    def init_runner(self):
        use_tqdm = self.visualization != 'none'
        use_visualization = self.visualization == 'full'
        self.runner = PPOFullRunner(
            env=self.env,
            agent=self.agent,
            loggers=[logger['logger'] for logger in self.loggers],
            use_tqdm=use_tqdm,
            use_visualization=use_visualization,
        )

    def load(self):
        """Load the agent's models from a file."""
        if self.load_model is None:
            return
        try:
            self.agent.load(self.load_model, zip=True)
        except FileNotFoundError:
            echo("No model found. Continuing without loading.")

    def save(self):
        """Save the agent's models to a file."""
        file = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if self.save_model is None:
            return
        save_path = os.path.join(self.save_model, file)
        self.agent.save(save_path, zip=True)

    def finalize_loggers(self):
        for logger in self.loggers:
            logger['logger'].finalize()

    def run(self):
        self.init_loggers()
        self.init_devices()
        self.init_env()
        self.init_agent()
        self.init_runner()
        self.load()
        if self.visualization != 'none' and any([logger['type'] == 'stdout' for logger in self.loggers]):
            echo("Warning: You have a stdout logger and a visualization. This may cause issues.")
        try:
            self.runner.run(
                num_envs=self.num_envs,
                eval_env=self.eval_env,
                total_steps=self.total_steps,
                worker_steps=self.worker_steps,
                evaluation_period=self.evaluation_period,
                evaluation_steps=self.evaluation_length,
            )
        except KeyboardInterrupt:
            if not self.save_model is None:
                save = yes_no_menu("Keyboard interrupt detected. Do you want to save the model?")
                if save:
                    self.save()
            self.finalize_loggers()
            sys.exit(0)
        self.save()
        self.finalize_loggers()

if __name__ == "__main__":
    config = choose_config()
    run = Run(config)
    run.run()