import gym
from gym.wrappers import RescaleAction
from domains import *
import matplotlib.pyplot as plt


env = RescaleAction(gym.make("water-maze-simple-pomdp-v0"), -1, 1)
env.reset()

plt.figure(figsize=(4,4))

start = plt.Circle((0, 0), 0.025, color='black', fill=True)
agent = plt.Circle((0.5, 0.5), 0.05, color='green', fill=True)
world = plt.Circle((0, 0), 1.0, color='black', fill=False)
platform = plt.Circle((env.platform_center[0], env.platform_center[1]), 0.3, color="red", fill=False)
plt.gca().add_patch(start)
plt.gca().add_patch(agent)
plt.gca().add_patch(world)
plt.gca().add_patch(platform)

plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.axis('off')

plt.show()
