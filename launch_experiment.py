"""
Launcher for experiments. 

Takes in a config file for arguments
"""
import os
import pathlib
import json
import click
from garage import wrap_experiment

# Importing envs
from envs.halfcheetahvel import HalfCheetahVelEnv
from envs.halfcheetahdir import HalfCheetahDirEnv
from envs.hopper_wrapper import HopperRandParamsWrappedEnv
from envs.walker_wrapper import WalkerRandParamsWrappedEnv
from envs.humanoid import HumanoidDirEnv
from envs.pointRobot import PointEnv

from configs.default import default_config
from pearl_launcher import pearl_experiment
from globex_launcher import globex_experiment

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values. 
    from https://github.com/katerakelly/oyster/blob/44e20fddf181d8ca3852bdf9b6927d6b8c6f48fc/launch_experiment.py'''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.option('--config', default=None)
@click.option('--name', default=None)
@click.option('--seed', default=1)
@click.option('--use_gpu', default=True)
@click.option('--gpu_id', default=0)
def main(config, name, seed, use_gpu, gpu_id):

    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['seed'] = seed #set seed via argument
    variant['util_params']['use_gpu'] = use_gpu
    variant['util_params']['gpu_id'] = gpu_id
    
    if name:
        experiment_name = name
    else:
        experiment_name = variant['algo']+'-'+variant['env_name']

    if variant['algo'] == 'GLOBEX':
        experiment_fn = wrap_experiment(name=experiment_name, archive_launch_repo=False)(globex_experiment)
    elif variant['algo'] == 'PEARL':
        experiment_fn = wrap_experiment(name=experiment_name, archive_launch_repo=False)(pearl_experiment)
    else:
        raise ValueError('No other algorithms available at this date')
    
    # Set env function
    if variant["env_name"] == 'cheetah-vel':
        env_class = HalfCheetahVelEnv
        is_gym = True
    elif variant["env_name"] == 'cheetah-dir':
        env_class = HalfCheetahDirEnv
        is_gym = True
    elif variant["env_name"] == 'hopper-rand-params':
        env_class = HopperRandParamsWrappedEnv
        is_gym = True
    elif variant["env_name"] == 'walker-rand-params':
        env_class = WalkerRandParamsWrappedEnv
        is_gym = True
    elif variant["env_name"] == 'humanoid-dir':
        env_class = HumanoidDirEnv
        is_gym = True
    elif variant["env_name"] == 'point-robot':
        env_class = PointEnv
        is_gym = False
    else:
        raise ValueError('Environment name not supported') 
    
    experiment_fn(env_class=env_class, is_gym=is_gym, variant=variant)

if __name__ == "__main__":
    main()