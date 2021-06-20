import gin


@gin.configurable(module=__name__)
def linear_decay(for_which_update: int, num_updates: int = gin.REQUIRED):
    """
    Derivation:
    We want to fit a line to two points: (1, 1) and (T+1, 0), where T
    is the number of updates. Therefore, the slope of the line is
    (0 - 1) / ((T+1) - 1) = - 1 / T

    Example:
    If num_updates = 5, slope would be - 1/5.
    lr_multiplier =
        1 + (1 - 1) * (-1/5) = 1 if for_which_update == 1
        1 + (2 - 1) * (-1/5) = 1 - 1/5 = 4/5 if for_which_update == 2
        1 + (3 - 1) * (-1/5) = 1 - 2/5 = 3/5 if for_which_update == 3
        1 + (4 - 1) * (-1/5) = 1 - 3/5 = 2/5 if for_which_update == 4
        1 + (5 - 1) * (-1/5) = 1 - 4/5 = 1/5 if for_which_update == 5
    """
    slope = - 1 / num_updates
    multiplier = 1 + (for_which_update - 1) * slope
    return multiplier
