from typing import Callable

import numpy as np


class ActionNoiseScheduler:

    def __init__(
        self,
        init_action_noise: float,
        schedule: Callable,
        fina_action_noise: float = 0.05
    ):

        self.init_action_noise = init_action_noise
        self.schedule = schedule  # a function of num_updates
        self.fina_action_noise = fina_action_noise

        self.for_which_update = 1

    def get_new_action_noise(self) -> float:
        new_action_noise = float(np.clip(self.init_action_noise * self.schedule(self.for_which_update),
                                         a_min=self.fina_action_noise, a_max=None))
        self.for_which_update += 1
        return new_action_noise
