import numpy as np
import matplotlib.pyplot as plt

y_half_length = 0.4
bump_diameter = 0.04
y_left_limit = -0.4
y_right_limit = 0.4

min_bump_distance = 0.3 * y_half_length
max_bump_distance = 0.8 * y_half_length
min_y_g_bump_distance = bump_diameter
y_bump1_limit_min = 0.7 * y_left_limit
y_bump2_limit_min = y_bump1_limit_min + min_bump_distance
y_bump2_limit_max = 0.7 * y_right_limit
y_bump1_limit_max = y_bump2_limit_max - min_bump_distance

red_positions = []
blue_positions = []
finger_positions = []


def _uniform_ranges(ranges):
    """
    Draws a sample from a uniform distribution on different ranges.

    :param ranges: the ranges to draw a sample
    :return: the sample drawn
    """

    index = np.random.randint(len(ranges))

    return np.random.uniform(low=ranges[index][0], high=ranges[index][1])


for i in range(10000):

    determine_bump_1_first = np.random.choice([True, False])

    if determine_bump_1_first:

        # y_bump1
        ori_y_bump1 = np.random.uniform(low=y_bump1_limit_min,
                                        high=y_bump1_limit_max)

        # y_bump2
        ori_y_bump2 = np.random.uniform(low=ori_y_bump1 + min_bump_distance,
                                        high=min(ori_y_bump1 + max_bump_distance, y_bump2_limit_max))

    else:

        ori_y_bump2 = np.random.uniform(low=y_bump2_limit_min,
                                                  high=y_bump2_limit_max)

        ori_y_bump1 = np.random.uniform(low=max(ori_y_bump2 - max_bump_distance, y_bump1_limit_min),
                                        high=ori_y_bump2 - min_bump_distance)

    # y_ur5_range1 = [y_left_limit, ori_y_bump1 - min_y_g_bump_distance]
    # y_ur5_range2 = [ori_y_bump1 + min_y_g_bump_distance, ori_y_bump2 - min_y_g_bump_distance]
    # y_ur5_range3 = [ori_y_bump2 + min_y_g_bump_distance, y_right_limit]
    # y_ur5 = _uniform_ranges([y_ur5_range1, y_ur5_range2, y_ur5_range3])

    red_positions.append(ori_y_bump1)
    blue_positions.append(ori_y_bump2)
    # finger_positions.append(y_ur5)

blue_cnt = 0
for pos in blue_positions:
    if pos >= 0.045 and pos <= 0.055:
        blue_cnt += 1

red_cnt = 0
for pos in red_positions:
    if pos >= 0.045 and pos <= 0.055:
        red_cnt += 1

print(red_cnt, blue_cnt)

# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(red_positions, blue_positions, finger_positions, alpha=0.2)
# ax.set_xlabel('red')
# plt.show()
