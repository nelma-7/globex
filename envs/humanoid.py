"""Humanoid meta-environments"""

import numpy as np
from gym.envs.mujoco import HumanoidEnv 

from garage import EnvSpec

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))

class HumanoidDirEnv(HumanoidEnv):
    """Humanoid environment with reward based on linear velocity and time spent alive. 
    
    The code is adapted from
    https://github.com/katerakelly/oyster/blob/44e20fddf181d8ca3852bdf9b6927d6b8c6f48fc/rlkit/envs/humanoid_dir.py"""

    def __init__(self, task=None, randomize_tasks=True):
        self._task = task or {'direction': np.pi}
        self.randomize_tasks = randomize_tasks
        super(HumanoidDirEnv, self).__init__()
        self.spec = EnvSpec(action_space=self.action_space,
                            observation_space=self.observation_space)

    def step(self, action):
        pos_before = np.copy(mass_center(self.model, self.sim)[:2])
        self.do_simulation(action, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)[:2]

        alive_bonus = 5.0
        data = self.sim.data
        goal_direction = (np.cos(self._task['direction']), np.sin(self._task['direction']))
        lin_vel_cost = 0.25 * np.sum(goal_direction * (pos_after - pos_before)) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

        infos = dict(reward_linvel=np.asarray([lin_vel_cost]), 
                     reward_quadctrl=np.asarray([-quad_ctrl_cost]),
                     reward_alive=np.asarray([alive_bonus]), 
                     reward_impact=np.asarray([-quad_impact_cost]),
                     task_pos=np.asarray([self._task['direction']]))

        return self._get_obs(), reward, done, infos

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def set_task(self, task):
        self._task = task

    def sample_tasks(self, num_tasks):
        directions = np.random.uniform(0., 2.0 * np.pi, size=(num_tasks,))
        tasks = [{'direction': d} for d in directions]
        return tasks