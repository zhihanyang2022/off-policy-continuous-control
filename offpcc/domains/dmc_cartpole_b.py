import dmc2gym
from domains.wrappers import ConcatObs
from domains.wrappers_for_image import Normalize255Image, GrayscaleImage, ConcatImages


def mdp():
    return dmc2gym.make(domain_name="cartpole", task_name="balance", keys_to_exclude=[], frame_skip=5)


def p():
    return dmc2gym.make(domain_name="cartpole", task_name="balance", keys_to_exclude=['velocity'], frame_skip=5)


def va():
    return dmc2gym.make(domain_name="cartpole", task_name="balance", keys_to_exclude=['position'], track_prev_action=False, frame_skip=5)


def p_concat5():
    return ConcatObs(p(), 5)

def va_concat10():
    return ConcatObs(va(), 10)


def pomdp_img():
    """frame skip follows from the Dreamer benchmark"""
    raw_env = dmc2gym.make(
        domain_name="cartpole",
        task_name="balance",
        keys_to_exclude=[],
        visualize_reward=False,
        from_pixels=True,
        frame_skip=2
    )
    return GrayscaleImage(Normalize255Image(raw_env))


def mdp_img_concat3():
    return ConcatImages(pomdp_img(), 3)
