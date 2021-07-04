import dmc2gym
from domains.wrappers_for_image import Normalize255Image, GrayscaleImage, ConcatImages


def pomdp_img():
    """frame skip follows from the Dreamer benchmark"""
    raw_env = dmc2gym.make(
        domain_name="walker",
        task_name="walk",
        keys_to_exclude=[],
        visualize_reward=False,
        from_pixels=True,
        frame_skip=2
    )
    return GrayscaleImage(Normalize255Image(raw_env))


def mdp_img_concat3():
    return ConcatImages(pomdp_img(), 3)
