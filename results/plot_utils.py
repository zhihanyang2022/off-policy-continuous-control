import numpy as np


def eleven_neighbor_smooth(scalars: list) -> list:
    scalars = [scalars[0]] * 5 + scalars + [scalars[-1]] * 5
    output = []
    for i in range(5, len(scalars) - 5):
        output.append(np.mean(scalars[i-5: i+5]))
    return output