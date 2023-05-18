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
from envs import HalfCheetahVelEnv

#TODO: set up configs.default and other launchers
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
@click.argument('config', default=None)
@click.option('--use_gpu', default=True)
@click.option('--gpu_id', default=0)
def main(config, use_gpu, gpu_id):

    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['use_gpu'] = use_gpu
    variant['util_params']['gpu_id'] = gpu_id

    if variant['algo'] == 'GLOBEX':
        experiment_fn = wrap_experiment(name='hello', archive_launch_repo=False)(globex_experiment)
    elif variant['algo'] == 'PEARL':
        experiment_fn = wrap_experiment(name='hello', archive_launch_repo=False)(pearl_experiment)
    else:
        raise ValueError('No other algorithms available at this date')
    
    # Set env function
    if variant["env_name"] == 'cheetah-vel':
        env_class = 1
    elif variant["env_name"] == 'cheetah-dir':
        env_class=1
    elif variant["env_name"] == 'hopper-rand-params':
        env_class=1
    elif variant["env_name"] == 'walker-rand-params':
        env_class=1
    elif variant["env_name"] == 'humanoid-dir':
        env_class=1
    elif variant["env_name"] == 'point-robot':
        env_class=1
    else:
        raise ValueError('Environment name not supported') 
    
    experiment_fn(env_class, variant)

if __name__ == "__main__":
    main()