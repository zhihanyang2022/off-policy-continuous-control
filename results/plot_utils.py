import numpy as np


def ignore_hidden_and_png(items):
    """Sometimes we get random hidden folders..."""
    return [item for item in items if item[0] != '.' and not item.endswith('png')]


def neighbor_smooth(scalars: list, num_neighbors) -> list:
    num_each_side = int(num_neighbors / 2 - 1)
    scalars = [scalars[0]] * num_each_side + scalars + [scalars[-1]] * num_each_side
    output = []
    for i in range(num_each_side, len(scalars) - num_each_side):
        output.append(np.mean(scalars[i-num_each_side: i+num_each_side]))
    return output