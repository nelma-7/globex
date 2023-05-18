"""Simple sparse 2D environment containing a point and a goal location on a 2d half-circle
Adapted from https://github.com/katerakelly/oyster/blob/44e20fddf181d8ca3852bdf9b6927d6b8c6f48fc/rlkit/envs/point_robot.py
"""
import math

import akro
import numpy as np

from garage import Environment, EnvSpec, EnvStep, StepType


class SparsePointEnv(Environment):
    """A simple 2D point environment with sparse rewards.
    Args:
        goal (np.ndarray): A 2D array representing the goal position
        arena_size (float): The size of arena where the point is constrained
            within (-arena_size, arena_size) in each dimension
        goal_radius (float): The radius around the goal point where the agent receives a reward
        never_done (bool): Never send a `done` signal, even if the
            agent achieves the goal
        max_episode_length (int): The maximum steps allowed for an episode.
    """

    def __init__(self,
                 goal=np.array((0., 1.), dtype=np.float32),
                 arena_size=2,
                 goal_radius = 0.1,
                 never_done=True,
                 max_episode_length=20):
        goal = np.array(goal, dtype=np.float32)
        self._goal = goal
        self._never_done = never_done
        self._arena_size = arena_size
        self._goal_radius = goal_radius

        assert ((goal >= -arena_size) & (goal <= arena_size)).all()

        self._step_cnt = None
        self._max_episode_length = max_episode_length
        self._visualize = False

        self._state = np.array([0, 0])
        self._task = {'goal': self._goal}
        self._observation_space = akro.Box(low=-np.inf,
                                           high=np.inf,
                                           shape=(2, ),
                                           dtype=np.float32)
        self._action_space = akro.Box(low=-0.1,
                                      high=0.1,
                                      shape=(2, ),
                                      dtype=np.float32)
        self._spec = EnvSpec(action_space=self.action_space,
                             observation_space=self.observation_space,
                             max_episode_length=max_episode_length)

    def _get_obs(self):
        return np.copy(self._state)
    
    @property
    def action_space(self):
        """akro.Space: The action space specification."""
        return self._action_space

    @property
    def observation_space(self):
        """akro.Space: The observation space specification."""
        return self._observation_space

    @property
    def spec(self):
        """EnvSpec: The environment specification."""
        return self._spec

    @property
    def render_modes(self):
        """list: A list of string representing the supported render modes."""
        return [
            'ascii',
        ]

    def sample_tasks(self, num_tasks):
        """Sample a list of `num_tasks` tasks.
        Args:
            num_tasks (int): Number of tasks to sample.
        Returns:
            list[dict[str, np.ndarray]]: A list of "tasks", where each task is
                a dictionary containing a single key, "goal", mapping to a
                point in 2D space.
        """
        #goals = np.random.uniform(-2, 2, size=(num_tasks, 2))
        angles = np.random.random(num_tasks) * np.pi
        #angles = np.linspace(0, np.pi, num=num_tasks)
        goals = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def set_task(self, task):
        """Reset with a task.
        Args:
            task (dict[str, np.ndarray]): A task (a dictionary containing a
                single key, "goal", which should be a point in 2D space).
        """
        self._task = task
        self._goal = task['goal']
    
    def reset(self):
        """Reset the environment.
        Returns:
            numpy.ndarray: The first observation conforming to
                `observation_space`.
            dict: The episode-level information.
                Note that this is not part of `env_info` provided in `step()`.
                It contains information of he entire episode， which could be
                needed to determine the first action (e.g. in the case of
                goal-conditioned or MTRL.)
        """
        self._state = np.array([0, 0])
        self._step_cnt = 0
        return self._get_obs(), {
                           'task': self._task,
                           'sparse_reward': 0
                       }

    def step(self, action):
        """Step the environment.
        Args:
            action (np.ndarray): An action provided by the agent.
        Returns:
            EnvStep: The environment step resulting from the action.
        Raises:
            RuntimeError: if `step()` is called after the environment
            has been
                constructed and `reset()` has not been called.
        """
        if self._step_cnt is None:
            raise RuntimeError('reset() must be called before step()!')

        self._state = self._state + action
        x, y = self._state
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = False
        ob = self._get_obs()
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        if reward >= -self._goal_radius:
            sparse_reward += 1
        
        if self._visualize:
            print(self.render('ascii'))
        
        self._step_cnt += 1
        
        step_type = StepType.get_step_type(
            step_cnt=self._step_cnt,
            max_episode_length=self._max_episode_length,
            done=done)
        
        if step_type in (StepType.TERMINAL, StepType.TIMEOUT):
            self._step_cnt = None
            
        return EnvStep(env_spec=self.spec,
                       action=action.astype('float32'),
                       reward=sparse_reward.astype('float32'),
                       observation=ob.astype('float32'),
                       env_info={
                           'task': self._task,
                           'original_reward': reward,
                           'sparse_reward': sparse_reward
                       },
                       step_type=step_type)

    def render(self, mode):
        """Renders the environment.
        Args:
            mode (str): the mode to render with. The string must be present in
                `self.render_modes`.
        Returns:
            str: the point and goal of environment.
        """
        x, y = self._state
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        return f'Point: {np.round(self._state,3)}, Goal: {np.round(self._goal,3)}, Reward (Distance): {np.round(reward,3)}'

    def visualize(self):
        """Creates a visualization of the environment."""
        self._visualize = True
        print(self.render('ascii'))

    def close(self):
        """Close the env."""

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= -self._goal_radius).astype(np.float32)
        r = r * mask
        return r
