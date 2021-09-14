import dmc2gym
from domains.wrappers import ConcatObs


def mdp():
    return dmc2gym.make(domain_name="cartpole", task_name="swingup", keys_to_exclude=[], frame_skip=5, track_prev_action=False)


def p():
    return dmc2gym.make(domain_name="cartpole", task_name="swingup", keys_to_exclude=['velocity'], frame_skip=5, track_prev_action=False)


def va():
    return dmc2gym.make(domain_name="cartpole", task_name="swingup", keys_to_exclude=['position'], frame_skip=5, track_prev_action=True)


def p_concat5():
    return ConcatObs(p(), 5)

def va_concat10():
    return ConcatObs(va(), 10)
