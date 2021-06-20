from typing import Callable


class ActionNoiseScheduler:

    def __init__(self, schedule: Callable):
        self.schedule = schedule  # a function of num_updates
        self.for_which_update = 1

    def get_new_action_noise(self) -> float:
        new_action_noise = self.schedule(self.for_which_update)
        self.for_which_update += 1
        return new_action_noise
