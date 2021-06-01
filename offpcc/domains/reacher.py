"""
Adapted from the original code here
https://github.com/openai/gym/blob/master/gym/envs/mujoco/reacher.py

Now the position of the goal is only shown once at the beginning of the episode (upon reset).

This is why I called it the Reacher-YOPO-Env, where YOPO stands for You-Only-Peek-Once

Changes has been made in method the step method only, with the help of a new method called zero_out_goal_info.
"""

import numpy as np
from gym import utils
import gym.spaces as spaces
from gym.envs.mujoco import mujoco_env
from copy import deepcopy


class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, yopo):
        """Modified"""

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

        self.yopo = yopo  # modification
        self.returning_first_obs = None  # set in reset() method

        if self.yopo:  # modification

            self.last_action = None  # set in reset() method

            low = self.observation_space.low
            high = self.observation_space.high

            self.observation_space = spaces.Box(
                low=np.concatenate([low, self.action_space.low]),
                high=np.concatenate([high, self.action_space.high])
            )

    def step(self, a):
        """Modified"""

        self.last_action = a  # modification

        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        done = False

        return self._get_obs(), reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        """Modified"""

        # modification
        self.last_action = np.zeros(self.action_space.shape)
        self.returning_first_obs = True

        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)

        return self._get_obs()

    @staticmethod
    def zero_out_goal_info(obs):
        """
        Added
        indices 4, 5, 8, 9, 10 contain info about the goal
        ref: https://github.com/openai/gym/wiki/Reacher-v2
        """
        obs = deepcopy(obs)  # prevent overriding the input for sanity
        obs[4, 5, 8, 9, 10] = 0
        return obs

    def _get_obs(self):
        """Modified"""
        theta = self.sim.data.qpos.flat[:2]
        ob = np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])
        if self.yopo:
            if self.returning_first_obs:
                self.returning_first_obs = False  # only show the goal info if returning the first obs
            else:
                ob = self.zero_out_goal_info(ob)
            ob = np.concatenate([ob, self.last_action])  # a is the last action
        return ob
