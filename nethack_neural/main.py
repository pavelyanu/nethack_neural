import os

from nethack_neural.agents.ppo_agent import GlyphBlstatsPPOAgent, GlyphPPOAgent
from nethack_neural.runners.ppo_runner import PPORunner
from nethack_neural.runners.ppo_visual_runner import PPOVisualRunner
from nethack_neural.loggers.file_logger import FileLogger
from nethack_neural.loggers.stdout_logger import StdoutLogger
from nethack_neural.utils.env_specs import EnvSpecs

import click
import minihack
import gym
import torch

@click.group()
def cli():
    pass


env_cli_map = {
    'corridor_battle': 'MiniHack-CorridorBattle-v0',
    'corridor': 'MiniHack-Corridor-R2-v0',
    'room': 'MiniHack-Room-5x5-v0',
    'river': 'MiniHack-River-MonsterLava-v0',
}

env_cli_options = list(env_cli_map.keys()) + ['custom']

keys_cli_map = {
    'gb': ['glyphs', 'blstats'],
    'g': ['glyphs'],
}

keys_to_agents_map = {
    'gb': GlyphBlstatsPPOAgent,
    'g': GlyphPPOAgent,
}

def resolve_environment(name, observation_keys):
    if name == 'custom':
        name = click.prompt('Enter custom environment name')
    else:
        try:
            name = env_cli_map[name]
        except KeyError:
            raise click.BadParameter('Invalid environment')
    try:
        observation_keys = keys_cli_map[observation_keys]
    except KeyError:
        raise click.BadParameter('Invalid observation keys')
    try:
        env = gym.make(name, observation_keys=observation_keys)
    except gym.error.Error:
        raise click.BadParameter('Invalid environment')
    return env, observation_keys


@cli.command()
@click.option('--environment', type=click.Choice(env_cli_options), default=env_cli_options[0], prompt='Choose an environment')
@click.option('--observation_keys', type=click.Choice(keys_cli_map.keys()), prompt='Choose observation keys. gb for glyphs and blstats, g for glyphs', default=list(keys_cli_map.keys())[0])
@click.option('--critic_lr', type=float, default=0.0003, prompt='Enter critic learning rate', required=False)
@click.option('--actor_lr', type=float, default=0.0003, prompt='Enter actor learning rate', required=False)
@click.option('--eps_clip', type=float, default=0.2, prompt='Enter eps clip', required=False)
@click.option('--hidden_layer_size', type=int, default=64, prompt='Enter hidden layer size', required=False)
@click.option('--num_envs', type=int, default=4, prompt='Enter number of environments')
@click.option('--total_steps', type=int, default=100000, prompt='Enter total steps')
@click.option('--worker_steps', type=int, default=1000, prompt='Enter worker steps')
@click.option('--evaluation_period', default=500, type=int, prompt='Enter evaluation period')
@click.option('--evaluation_length', default=5, type=int, prompt='Enter evaluation length')
@click.option('--batch_size', type=int, default=64, prompt='Enter batch size')
@click.option('--epochs', type=int, default=10, prompt='Enter number of epochs')
@click.option('--training_device', type=click.Choice(['cpu', 'gpu', 'auto']), default='auto', prompt='Choose training device')
@click.option('--logger', type=click.Choice(['stdout', 'file', 'none']), default='none', prompt='Choose logger type')
@click.option('--visualization', type=click.Choice(['none', 'full', 'bar']), default='full', prompt='Choose visualization type')
@click.option('--save_model', is_flag=False, prompt='Save model?')
@click.option('--load_model', is_flag=False, prompt='Load model?')
@click.pass_context
def fully_specialized(ctx, environment, observation_keys, critic_lr, actor_lr, eps_clip, hidden_layer_size, num_envs, total_steps, worker_steps, evaluation_period, evaluation_length, batch_size, epochs, training_device, logger, visualization, save_model, load_model):
    env, keys = resolve_environment(environment, observation_keys)
    if save_model:
        save_model_path = click.prompt('Enter model save directory')
    if load_model:
        load_model_path = click.prompt('Enter model load directory')
    if training_device == 'auto':
        training_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env_specs = EnvSpecs()
    env_specs.init_with_gym_env(env, num_envs=num_envs)
    venv = gym.vector.make(
        env.spec.id,
        num_envs=num_envs,
        observation_keys=keys)
    agent = keys_to_agents_map[observation_keys](
        env_specs,
        critic_lr=critic_lr,
        actor_lr=actor_lr,
        eps_clip=eps_clip,
        hidden_layer=hidden_layer_size,
        batch_size=batch_size,
        epochs=epochs,
        training_device=training_device,
        storage_device=training_device,
    )
    if load_model:
        try:
            agent.load(load_model_path)
        except Exception:
            print('Could not load model. Continuing without loading model. Exception: ' + str(Exception))
    loggers = []
    if logger == 'stdout':
        loggers.append(StdoutLogger())
    elif logger == 'file':
        file = click.prompt('Enter log files directory')
        if not os.path.isdir(file):
            os.mkdir(file)
        loggers.append(FileLogger(f"{file}.msg", f"{file}.csv"))
    if visualization == 'full':
        runner = PPOVisualRunner(venv, agent, loggers)
    elif visualization == 'bar':
        runner = PPORunner(venv, agent, loggers, use_tqdm=True)
    else:
        runner = PPORunner(venv, agent, loggers, use_tqdm=False)
    if save_model:
        print('Testing save model')
        try:
            agent.save(save_model_path)
        except Exception:
            print('Could not save model. Exception: ' + str(Exception))
            response = click.prompt('Continue without saving model? (y/n)', is_flag=True)
            if not response:
                return
    runner.run(
        num_envs,
        env,
        total_steps=total_steps,
        worker_steps=worker_steps,
        evaluation_period=evaluation_period,
        evaluation_steps=evaluation_length,
    )
    if save_model:
        agent.save(save_model_path)

@cli.command()
@click.option('--environment', type=click.Choice(env_cli_options), default=env_cli_options[0], prompt='Choose an environment')
@click.option('--observation_keys', type=click.Choice(keys_cli_map.keys()), prompt='Choose observation keys. gb for glyphs and blstats, g for glyphs', default=list(keys_cli_map.keys())[0])
@click.option('--num_envs', type=int, default=4, prompt='Enter number of environments')
@click.option('--total_steps', type=int, default=100000, prompt='Enter total steps')
@click.option('--worker_steps', type=int, default=1000, prompt='Enter worker steps')
@click.option('--evaluation_period', default=500, type=int, prompt='Enter evaluation period')
@click.option('--evaluation_length', default=5, type=int, prompt='Enter evaluation length')
@click.option('--visualization', type=click.Choice(['none', 'full', 'bar']), default='full', prompt='Choose visualization type')
@click.pass_context
def specialized(ctx, environment, observation_keys, num_envs, total_steps, worker_steps, evaluation_period, evaluation_length, visualization):
    return ctx.invoke(fully_specialized, environment=environment, observation_keys=observation_keys, num_envs=num_envs, total_steps=total_steps, worker_steps=worker_steps, evaluation_period=evaluation_period, evaluation_length=evaluation_length, visualization=visualization)


@cli.command()
@click.pass_context
def default(ctx):
    return ctx.invoke(fully_specialized)

if __name__ == '__main__':
    cli()