import gym
from gym.wrappers import RescaleAction
from domains import *
import numpy as np
from algorithms_recurrent import RecurrentSAC
import matplotlib.pyplot as plt


env = RescaleAction(gym.make("water-maze-simple-pomdp-v0"), -1, 1)
algo = RecurrentSAC(input_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
algo.load_actor("../results/water-maze-simple-pomdp-v0/rsac/2/")

# env.set_platform(np.pi * 1.25)
env.set_platform(np.random.uniform(0.75 * np.pi, 1.25 * np.pi))
obs = env.reset()
trajs_toward_platform = []
traj = []
traj_index = 1
cnt = 0

while True:

    traj.append((obs[0], obs[1]))

    if obs[2] == 1:
        cnt += 1
        if cnt == 5:
            correct_traj = traj[:-1]
            correct_traj.append(info['final_pos'])
            trajs_toward_platform.append(correct_traj)
            traj_index += 1
            if traj_index > 3:
                break
            traj = [traj[-1]]
            cnt = 0

    action = algo.act(obs, deterministic=True)
    next_obs, reward, done, info = env.step(action)

    if done:
        break

    obs = next_obs

plt.figure(figsize=(4,4))

def add_arrow(line, position=None, direction='right', size=30, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="-|>", color=color),
        size=size
    )

def index_to_rank(index):
    if index == 1:
        return "st"
    elif index == 2:
        return "nd"
    elif index == 3:
        return "rd"
    elif index == 4:
        return "th"


for i, traj in enumerate(trajs_toward_platform):
    xs, ys = list(zip(*traj))
    line = plt.plot(xs, ys, label=f"{i+1}{index_to_rank(i+1)} attempt")[0]
    # plt.scatter(xs[0], ys[0], color=line.get_color(), marker='o')
    # plt.scatter(xs[-1], ys[-1], color=line.get_color(), marker='o')
    add_arrow(line)

start = world = plt.Circle((0, 0), 0.025, color='black', fill=False)
world = plt.Circle((0, 0), 1.0, color='black', fill=False)
platform = plt.Circle((env.platform_center[0], env.platform_center[1]), 0.3, color="red", fill=False)
plt.gca().add_patch(start)
plt.gca().add_patch(world)
plt.gca().add_patch(platform)

plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.axis('off')
plt.legend()

plt.show()
