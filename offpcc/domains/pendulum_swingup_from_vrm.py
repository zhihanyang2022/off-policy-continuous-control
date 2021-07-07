from copy import deepcopy
import gym
import gym.spaces as spaces
from domains.wrappers import ConcatObs
import numpy as np


class PendulumP(gym.Env):
    """
    Partially observed Pendulum:
    Only the current angle is observed
    """
    def __init__(self):

        self.env = gym.make('Pendulum-v0').env
        self.red_position = 0
        self.green_position = 0
        self.blue_position = 0

        self.action_space = spaces.Box(
            low=-2,
            high=2,
            shape=(1,)
        )
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,)
        )

    def reset(self):
        obs = self.env.reset()
        self.old_position = obs[0:2]
        return obs[0:2]

    def step(self, action):
        action = action
        obs, r, done, _ = self.env.step(action)
        self.new_position = obs[0:2]
        self.old_position = deepcopy(self.new_position)
        return obs[0:2], r, done, {}

    def render(self, mode='human'):
        self.env.render()


class PendulumV(gym.Env):
    """
    Partially observed Pendulum:
    Only the current angular velocity is observed
    """
    def __init__(self):

        self.env = gym.make('Pendulum-v0').env
        self.red_position = 0
        self.green_position = 0
        self.blue_position = 0

        self.action_space = spaces.Box(
            low=-2,
            high=2,
            shape=(1,)
        )
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1,)
        )

    def reset(self):
        obs = self.env.reset()
        self.old_position = obs[2:3]
        return obs[2:3]

    def step(self, action):
        action = action
        obs, r, done, _ = self.env.step(action)
        self.new_position = obs[2:3]
        self.old_position = deepcopy(self.new_position)
        return obs[2:3], r, done, {}

    def render(self, mode='human'):
        self.env.render()


class PendulumVA(gym.Env):
    """
    Partially observed Pendulum:
    Only the current angular velocity is observed
    """
    def __init__(self):

        self.env = gym.make('Pendulum-v0').env
        self.red_position = 0
        self.green_position = 0
        self.blue_position = 0

        self.action_space = spaces.Box(
            low=-2,
            high=2,
            shape=(1,)
        )
        self.observation_space = spaces.Box(
            low=np.array([-1, -2]),
            high=np.array([1, 2]),
            shape=(2,)
        )

        self.last_action = None

    def reset(self):
        obs = self.env.reset()
        self.last_action = np.array([0.])
        self.old_position = obs[2:3]
        return np.concatenate([obs[2:3], self.last_action])

    def step(self, action):
        self.last_action = deepcopy(action)
        obs, r, done, _ = self.env.step(action)
        self.new_position = obs[2:3]
        self.old_position = deepcopy(self.new_position)
        return np.concatenate([obs[2:3], self.last_action]), r, done, {}

    def render(self, mode='human'):
        self.env.render()


def mdp():
    return gym.make('Pendulum-v0').env


def p():
    return PendulumP()


def v():
    return PendulumV()


def va():
    return PendulumVA()


def p_concat5():
    return ConcatObs(p(), 5)


def v_concat10():
    return ConcatObs(v(), 10)


def va_concat10():
    return ConcatObs(va(), 10)
