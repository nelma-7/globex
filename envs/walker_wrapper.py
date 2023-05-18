"""Walker wrapper, from 
https://github.com/katerakelly/oyster/blob/44e20fddf181d8ca3852bdf9b6927d6b8c6f48fc/rlkit/envs/hopper_rand_params_wrapper.py"""

from garage import EnvSpec
import akro
import numpy as np
from rand_param_envs.walker2d_rand_params import Walker2DRandParamsEnv

class WalkerRandParamsWrappedEnv(Walker2DRandParamsEnv):
    def __init__(self, n_tasks=2, randomize_tasks=True):
        super(WalkerRandParamsWrappedEnv, self).__init__()
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)
        self.action_space = akro.Box(low=self.action_space.low, high=self.action_space.high)
        self.observation_space = akro.Box(low=self.observation_space.low, high=self.observation_space.high)
        self.spec = EnvSpec(action_space=self.action_space,
                            observation_space=self.observation_space)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()
        
    def reset(self):
        return self._reset()
    