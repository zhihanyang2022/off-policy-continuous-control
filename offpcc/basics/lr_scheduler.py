from typing import Callable, List

import torch.optim as optim


class LRScheduler:
    """A scheduler for one or more optimizers."""

    def __init__(
        self,
        optimizers: List[optim.Optimizer],
        init_lr: float,
        schedule: Callable
    ):

        self.optimizers = optimizers
        self.init_lr = init_lr
        self.schedule = schedule  # a function of num_updates

        self.for_which_update = 1

    def get_new_lr(self) -> None:
        for optimizer in self.optimizers:
            for g in optimizer.param_groups:
                g['lr'] = self.init_lr * self.schedule(self.for_which_update)
        self.for_which_update += 1
