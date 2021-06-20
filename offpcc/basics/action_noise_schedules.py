import gin

import numpy as np


@gin.configurable(module=__name__)
def linear_decay_with_lower_bound(for_which_update: int, num_updates: int = gin.REQUIRED,
                                  start_val: float = gin.REQUIRED, end_val: float = gin.REQUIRED):
    """
    Derivation:
    We want to fit a line to two points: (1, start_val) and (T+1, end_val), where T
    is the number of updates. Therefore, the slope of the line is
    (end_val - start_val) / ((T+1) - 1) = (end_val - start_val) / T
    """
    slope = (end_val - start_val) / num_updates  # negative if end_val < start_val, which is usually the case
    action_noise = start_val + (min(for_which_update, num_updates) - 1) * slope
    return action_noise
