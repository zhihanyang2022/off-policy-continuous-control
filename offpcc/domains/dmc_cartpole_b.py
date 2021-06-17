import dmc2gym
from domains.wrappers import ConcatObs


def mdp():
    return dmc2gym.make(domain_name="cartpole", task_name="balance", keys_to_exclude=[], frame_skip=5)


def pomdp():
    return dmc2gym.make(domain_name="cartpole", task_name="balance", keys_to_exclude=['velocity'], frame_skip=5)


def mdp_concat5():
    return ConcatObs(pomdp(), 5)
