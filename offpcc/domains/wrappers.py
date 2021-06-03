from collections import deque
import numpy as np
import gym
import gym.spaces as spaces

# FYI
# - By default, wrappers directly copy the observation and action space from their wrappees.
# - The observation method in ObservationWrappers will get used by both env.reset and env.step.
# - In general, observation space is used only for getting observation shape for networks.


class FilterObsByIndex(gym.ObservationWrapper):

    def _filter(self, array: np.array) -> np.array:
        return np.array(
            [x for i, x in enumerate(array) if i in self.indices_to_keep]
        )

    def __init__(self, env, indices_to_keep: list):

        super().__init__(env)
        self.indices_to_keep = indices_to_keep

        new_high, new_low = self._filter(self.env.observation_space.high), self._filter(self.env.observation_space.low)
        self.observation_space = spaces.Box(low=new_low, high=new_high)

    def observation(self, observation):
        return self._filter(observation)


class ConcatObs(gym.ObservationWrapper):

    def __init__(self, env, window_size: int):

        super().__init__(env)

        # get info on old observation space
        old_obs_space = env.observation_space
        old_obs_space_dim = old_obs_space.shape[0]
        old_obs_space_low, old_obs_space_high = old_obs_space.low, old_obs_space.high

        # change observation space
        self.observation_space = spaces.Box(
            low=np.array(list(old_obs_space_low) * window_size),
            high=np.array(list(old_obs_space_high) * window_size)
        )

        self.window = deque(maxlen=window_size)
        for i in range(window_size - 1):
            self.window.append(np.zeros((old_obs_space_dim, )))  # append some dummy observations first

        self.window_size = window_size
        self.old_obs_space_dim = old_obs_space_dim

    def observation(self, obs: np.array) -> np.array:
        self.window.append(obs)
        return np.concatenate(self.window)

    def reset(self):
        for i in range(self.window_size - 1):
            self.window.append(np.zeros((self.old_obs_space_dim, )))  # append some dummy observations first
        observation = self.env.reset()
        return self.observation(observation)
