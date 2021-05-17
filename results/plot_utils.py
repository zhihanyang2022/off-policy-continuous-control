import numpy as np


def eleven_neighbor_smooth(scalars: list, num_neighbors) -> list:
    num_each_side = int(num_neighbors / 2 - 1)
    scalars = [scalars[0]] * num_each_side + scalars + [scalars[-1]] * num_each_side
    output = []
    for i in range(num_each_side, len(scalars) - num_each_side):
        output.append(np.mean(scalars[i-num_each_side: i+num_each_side]))
    return output