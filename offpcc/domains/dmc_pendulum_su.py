import dmc2gym


def mdp():
    return dmc2gym.make(domain_name="pendulum", task_name="swingup", keys_to_exclude=[], frame_skip=5, track_prev_action=False)
