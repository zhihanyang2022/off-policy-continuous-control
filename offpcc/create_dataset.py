import numpy as np
import gym
from domains import *
from gym.wrappers import RescaleAction

from algorithms_recurrent import RecurrentSAC
from basics.run_fns import make_log_dir


env = RescaleAction(gym.make("pendulum-p-v0"), -1, 1)
algorithm = RecurrentSAC(
    input_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
)
algorithm.load_actor(save_dir=make_log_dir("pendulum-p-v0", 'rsac', 1))


def rollout_one_episode(env, algorithm):

    obs, done, episode_return = env.reset(), False, 0

    algorithm.reinitialize_hidden()

    list_of_obs = []
    list_of_act = []

    while not done:

        action = algorithm.act(obs, deterministic=True)

        list_of_obs.append(obs)
        list_of_act.append(action)

        obs, reward, done, info = env.step(action)

        episode_return += reward

    return list_of_obs, list_of_act, episode_return


global_list_of_obs = []
global_list_of_act = []
for i in range(1000):
    list_of_obs, list_of_act, episode_return = rollout_one_episode(env, algorithm)
    print(i, episode_return)
    global_list_of_obs.append(list_of_obs)
    global_list_of_act.append(list_of_act)

obs_data = np.array(global_list_of_obs)
act_data = np.array(global_list_of_act)

print(obs_data.shape, act_data.shape)

np.save('../results/obs.npy', obs_data)
np.save('../results/act.npy', act_data)
