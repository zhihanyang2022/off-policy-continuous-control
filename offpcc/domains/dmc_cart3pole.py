import dmc2gym
from domains.wrappers import ConcatObs
from domains.wrappers_for_image import Normalize255Image, GrayscaleImage, ConcatImages


def mdp():
    return dmc2gym.make(domain_name="cartpole", task_name="three_poles", keys_to_exclude=[], frame_skip=5)


# def p():
#     return dmc2gym.make(domain_name="cartpole", task_name="balance", keys_to_exclude=['velocity'], frame_skip=5)
#
#
# def va():
#     return dmc2gym.make(domain_name="cartpole", task_name="balance", keys_to_exclude=['position'], frame_skip=5)
#
#
# def p_concat5():
#     return ConcatObs(p(), 5)
#
# def va_concat10():
#     return ConcatObs(va(), 10)
