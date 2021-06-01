"""
Adapted from the original code here
https://github.com/openai/gym/blob/master/gym/envs/mujoco/reacher.py

Now the position of the goal is only shown once at the beginning of the episode (upon reset).

This is why I called it the Reacher-YOPO-Env, where YOPO stands for You-Only-Peek-Once

Changes has been made in method the step method only, with the help of a new method called zero_out_goal_info.
"""


import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from copy import deepcopy


class ReacherYOPOEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    @staticmethod
    def zero_out_goal_info(obs):
        """
        indices 4, 5, 8, 9, 10 contain info about the goal
        ref: https://github.com/openai/gym/wiki/Reacher-v2
        """
        obs = deepcopy(obs)  # prevent overriding the input for sanity
        obs[4, 5, 8, 9, 10] = 0
        return obs

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return self.zero_out_goal_info(ob), reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):

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

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])
