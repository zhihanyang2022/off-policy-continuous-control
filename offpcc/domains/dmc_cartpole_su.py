import dmc2gym
from domains.wrappers import ConcatObs


def mdp():
    return dmc2gym.make(domain_name="cartpole", task_name="swingup", keys_to_exclude=[])


def pomdp():
    return dmc2gym.make(domain_name="cartpole", task_name="swingup", keys_to_exclude=['velocity'])


def mdp_concat5():
    return ConcatObs(pomdp(), 5)
