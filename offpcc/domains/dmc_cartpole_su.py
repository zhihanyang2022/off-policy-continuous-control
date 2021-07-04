import dmc2gym
from domains.wrappers import ConcatObs
import numpy as np


def mdp():
    return dmc2gym.make(domain_name="cartpole", task_name="swingup", keys_to_exclude=[], frame_skip=5)


def pomdp():
    return dmc2gym.make(domain_name="cartpole", task_name="swingup", keys_to_exclude=['velocity'], frame_skip=5, seed=np.random.randint(1000))


def mdp_concat5():
    return ConcatObs(pomdp(), 5)
