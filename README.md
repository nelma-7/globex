# Global-Local Embeddings for Contextual Meta-RL (GLOBEX)

This repository is the official implementation of GLOBEX, from Global-Local Decomposition of Contextual Representations in Meta-Reinforcement Learning, by Nelson Ma, Junyu Xuan, Jie Lu and Guangquan Zhang, submitted to IEEE Transactions on Cybernetics. 

## Requirements
The codebase has been tested on Ubuntu 20.04. Note that the MuJoCo libraries used are generally incompatible with Windows, so using a Linux or Mac OS (or WSL) is strongly recommended.

To install requirements:
```setup
pip install -r requirements.txt
```

In order to run experiments, you will also need to install MuJoCo: 
- [Install MuJoCo210](https://github.com/deepmind/mujoco/releases/tag/2.1.0) - this should enable the HalfCheetahVel and HalfCheetahDir environments
- Install [mjpro131](https://www.roboti.us/download.html) and download the activation key [mjkey.txt](https://www.roboti.us/file/mjkey.txt) into your ~/.mujoco/ folder. This should enable the Hopper and Walker environments    
- Set LD_LIBRARY_PATH to point to both the MuJoCo binaries (/$HOME/.mujoco/mujoco210/bin) as well as the gpu drivers (something like /usr/lib/nvidia-390, you can find your version by running nvidia-smi)

## Instructions

To run experiments:

```train
python launch_experiment.py --config ./configs/[EXPERIMENT].json
```

.json files in configs/ will set up experiments. Here, we include config files for the GLOBEX and PEARL MuJoCo experiments.  

Results for MAML were generated with the [garage MAML-PPO implementation](https://github.com/rlworkgroup/garage/tree/2d594803636e341660cab0e81343abbe9a325353/src/garage/examples/torch), while results for variBAD were generated with [original paper code](https://github.com/lmzintgraf/varibad)

By default, the code will use GPU. To use CPU instead, either use the option `--use_gpu false` or edit the config file.

If ran without any config argument, launch_experiment.py will run the GLOBEX HalfCheetahVel experiment.

## Visualising Results
This implementation is based on the [garage package](https://github.com/rlworkgroup/garage), and will automatically write to /data/local/experiment. To set a custom experiment name, you can use the option `--name [NAME]` when running the experiment.

Finally, results can be accessed using tensorboard:
```results
tensorboard --logdir data
```
