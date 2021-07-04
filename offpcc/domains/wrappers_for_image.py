import gym
import gym.spaces as spaces
import numpy as np
from collections import deque


class GrayscaleImage(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        old_observation_space = env.observation_space
        self.observation_space = spaces.Box(
            low=np.expand_dims(old_observation_space.low[0], axis=0),
            high=np.expand_dims(old_observation_space.high[0], axis=0),
        )
        # https://www.kite.com/python/answers/how-to-convert-an-image-from-rgb-to-grayscale-in-python
        self.rgb_weights = np.array([0.2989, 0.5870, 0.1140]).reshape(3, 1)

    def observation(self, observation):
        observation = np.moveaxis(observation, 0, 2)
        observation = observation @ self.rgb_weights
        return np.moveaxis(observation, 2, 0)


class Normalize255Image(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        old_observation_space = env.observation_space
        self.observation_space = spaces.Box(
            low=old_observation_space.low / 255,
            high=old_observation_space.high / 255
        )

    def observation(self, observation):
        return observation / 255  # uint8 automatically get converts to float64


class ConcatImages(gym.ObservationWrapper):

    def __init__(self, env, window_size: int):
        super().__init__(env)

        self.window_size = window_size
        self.old_obs_space_shape = env.observation_space.shape

        old_obs_space = env.observation_space
        old_obs_low = old_obs_space.low
        old_obs_high = old_obs_space.high

        self.observation_space = spaces.Box(
            low=np.concatenate([old_obs_low, old_obs_low, old_obs_low]),
            high=np.concatenate([old_obs_high, old_obs_high, old_obs_high]),
            dtype=np.uint8  # a must for images to work with SB3
        )

        self.window = deque(maxlen=window_size)
        for i in range(self.window_size - 1):
            self.window.append(np.zeros(self.old_obs_space_shape))

    def observation(self, observation):
        self.window.append(observation)
        return np.concatenate(self.window)

    def reset(self):
        for i in range(self.window_size - 1):
            self.window.append(np.zeros(self.old_obs_space_shape))
        observation = self.env.reset()
        return self.observation(observation)
