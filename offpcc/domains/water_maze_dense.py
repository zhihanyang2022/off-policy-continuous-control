import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import copy
import socket
if socket.gethostname() not in ['theseus', 'SXC-Wichita']:
    from gym.envs.classic_control import rendering as visualize

from domains.wrappers import FilterObsByIndex


class WaterMazeMdpEnv(gym.Env):

    def __init__(self, max_action_value=0.2):

        self.max_action_value = max_action_value

        self.action_space = spaces.Box(low=-max_action_value,
                                       high=max_action_value,
                                       shape=(2,))

        self.observation_space = spaces.Box(-1., 1., shape=(5,))

        self.platform_radius = 0.2
        self.world_radius = 1.0

        self.viewer = None

        self.screen_width = 300
        self.screen_height = 300

        self.setup_view = False

        self.scale = self.screen_width / self.world_radius

        self.step_in_platform = 0

        self.inside_platform = 0.0

        self.seed()

        self.agent_pos = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # At reset: randomize the position of the agent and the platform
    # such that the agent is not within the platform
    def reset(self):
        self.inside_platform = 0.0
        self.step_in_platform = 0

        while (True):
            theta_agent = 2 * np.pi * self.np_random.rand()
            radius_agent = self.np_random.rand()
            self.agent_pos = np.array([radius_agent * np.cos(theta_agent), radius_agent * np.sin(theta_agent)])

            theta_platform = 2 * np.pi * self.np_random.rand()
            radius_platform = self.np_random.rand()
            self.platform_center = np.array(
                [radius_platform * np.cos(theta_platform), radius_platform * np.sin(theta_platform)])

            is_platform_within_world = self._is_circle_within_circle(np.array([0, 0]), self.world_radius,
                                                                     self.platform_center, self.platform_radius)
            is_agent_not_in_platform = not self._is_within_circle(self.agent_pos, self.platform_center,
                                                                  self.platform_radius)

            if is_agent_not_in_platform and is_platform_within_world:
                break

        return self._get_obs()

    def _randomize_agent(self):
        while (True):
            theta_agent = 2 * np.pi * self.np_random.rand()
            radius_agent = self.np_random.rand()
            agent_pos = np.array([radius_agent * np.cos(theta_agent), radius_agent * np.sin(theta_agent)])

            is_agent_not_in_platform = not self._is_within_circle(agent_pos, self.platform_center, self.platform_radius)

            if is_agent_not_in_platform:
                return agent_pos

    def step(self, action):

        previous_pos = copy.deepcopy(self.agent_pos)
        action = action * self.max_action_value
        self.agent_pos += np.array(action)

        # If new action move the agent out of the world, revert back
        if not self.is_agent_inside_world():
            self.agent_pos = previous_pos

        # # The agent is rewarded if it is inside the platform
        # reward = 0
        # self.inside_platform = 0.0
        # if self._is_within_circle(self.agent_pos, self.platform_center, self.platform_radius):
        #     # counting the number of timesteps inside the platform
        #     self.step_in_platform += 1
        #     self.inside_platform = 1.0
        #     reward = 1

        vec = self.agent_pos - self.platform_center
        distance_to_go = np.linalg.norm(vec)
        reward = - distance_to_go

        if self._is_within_circle(self.agent_pos, self.platform_center, self.platform_radius):
            self.step_in_platform += 1

        # Randomize the agent again when it stays within the platform for 5 consecutive timesteps
        if self.step_in_platform % 5 == 0 and self.step_in_platform > 0:
            self.agent_pos = self._randomize_agent()
            self.step_in_platform = 0

        # Only terminate due to the TimeLimit Wrapper
        return self._get_obs(), reward, False, {}

    # The agent knows its position and whether it is inside the platform or not
    def _get_obs(self):
        return np.array([self.agent_pos[0], self.agent_pos[1], self.inside_platform, *list(self.platform_center)])

    def render(self, mode='human'):
        self._setup_view()

        # Update platform
        new_transform = self.platform_center * self.scale + np.array([300, 300])
        self.platform_transform.set_translation(new_transform[0], new_transform[1])

        # Update agent
        new_transform = self.agent_pos * self.scale + np.array([300, 300])
        self.agent_transform.set_translation(new_transform[0], new_transform[1])

        self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _setup_view(self):
        screen_width = 600
        screen_height = 600
        if not self.setup_view:
            self.viewer = visualize.Viewer(screen_width, screen_height)

            world = visualize.make_circle(300, filled=False)
            world.add_attr(visualize.Transform(translation=(300, 300)))
            world.set_color(1.0, .0, .0)
            self.viewer.add_geom(world)

            # Platform
            self.platform = visualize.make_circle(self.platform_radius * self.scale, filled=False)
            self.platform_transform = visualize.Transform()
            self.platform.add_attr(self.platform_transform)
            self.platform.set_color(0.0, 1.0, .0)
            self.viewer.add_geom(self.platform)
            self.setup_view = True

            # Agent
            self.agent = visualize.make_circle(10)
            self.agent_transform = visualize.Transform()
            self.agent.add_attr(self.agent_transform)
            self.agent.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(self.agent)

    def is_agent_inside_world(self):
        return self._is_within_circle(self.agent_pos, np.array([0.0, 0.0]), self.world_radius)

    def _is_within_circle(self, pos, c, r):
        distance = np.linalg.norm(pos - c)
        return distance < r

    def _is_circle_within_circle(self, c_big, r_big, c_small, r_small):
        d = np.linalg.norm(c_big - c_small)
        return r_big > d + r_small

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def mdp():
    return WaterMazeMdpEnv()


def pomdp():
    return FilterObsByIndex(mdp(), indices_to_keep=[0, 1, 2])

