from gym import spaces
import numpy as np
import pybullet as p

from domains.robot_envs.env import EnvObject, ASSETS_PATH
from domains.robot_envs.bumps_env import BumpsEnvBase

BUMP_S_URDF_PATH = ASSETS_PATH / 'bumps' / 'bump_40_red.urdf'
BUMP_L_URDF_PATH = ASSETS_PATH / 'bumps' / 'bump_50.urdf'


class BumpsDiffEnv(BumpsEnvBase):
    """
    Description:
        The PyBullet simulation environment of a two-different-bump environment.

    Observation:
         Type: Box(4)
         Num    Observation               Min                         Max
         0      Finger Tip Position Y     self.y_left_limit           self.y_right_limit
         1      Bump S Position Y         self.y_bump_s_limit_min     self.y_bump_s_limit_max
         2      Bump L Position Y         self.y_bump_l_limit_min     self.y_bump_l_limit_max
         3      Finger Angle              - pi / 3                    pi / 3

    Actions (discrete mode):
        Type: Discrete(4)
        Num    Action
        0      Moving left with soft finger
        1      Moving left with hard finger
        2      Moving right with soft finger
        3      Moving right with hard finger

    Actions (continuous mode):
        Type: Box(2)
        Num    Action             Range
        0      step length ratio  [-1, 1]
        1      stiffness ratio    [-1, 1]

    Reward:
        Reward of 1 for successfully pushing the larger bump after visiting both the smaller and the larger bump.

    Starting State:
        The starting state of the gripper is assigned to y_g = random and theta = 0

    Episode Termination:
        Either bump is pushed.
    """

    def __init__(self, rendering=False, hz=240, seed=None, discrete=False, action_failure_prob=-1.0):
        """
        The initialization of the PyBullet simulation environment.

        :param rendering: True if rendering, False otherwise
        :param hz: Hz for p.setTimeStep
        :param seed: the random seed
        :param discrete: True if action is discrete, False otherwise
        :param action_failure_prob: determines the probability that an action fails
        """

        super().__init__(rendering, hz, seed, discrete, action_failure_prob)

        # Flag to determine if both bumps are touched
        self.feel_bump_s = False
        self.feel_bump_l = False

        # Bumps parameters
        # Smaller bump
        self.bump_s = None  # Placeholder for declaration
        self.smaller_bump_diameter = 0.04
        self.y_bump_s = 0
        self.ori_y_bump_s = -0.2
        self.bump_s_num_contacts_static = 0

        # Larger bump
        self.bump_l = None  # Placeholder for declaration
        self.larger_bump_diameter = 0.05
        self.y_bump_l = 0
        self.ori_y_bump_l = 0.2
        self.bump_l_num_contacts_static = 0

        # Bump distances
        self.min_bump_distance = 0.6 * self.y_half_length
        self.max_bump_distance = self.y_half_length
        self.min_y_g_bump_distance = self.larger_bump_diameter
        self.y_bump_limit_min = 0.6 * self.y_left_limit
        self.y_bump_limit_max = 0.6 * self.y_right_limit

        # Reward/done thresholds
        self.pushing_reward_threshold = 0.01
        self.pushing_done_threshold = 0.001

        # Obs: (y_g, y_bump_s, y_bump_l, theta)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(4,), dtype=np.float32)

        # Declarations of the gripper's state
        self.y_g = 0
        self.theta = 0

        # Declarations of the auxiliary internal state parameters.
        self.y_ur5 = 0

    def reset(self):
        """
        Performs common reset functionality for all supported robots.
        """

        # Resets the robot to the home position.
        self.robot.reset()
        self.robot.ee.set_joints(self.default_angles, velocities=[0.1, ],
                                 forces=[self.low_stiffness, ])

        # y_bump_s
        self.ori_y_bump_s = self.np_random.uniform(low=self.y_bump_limit_min, high=self.y_bump_limit_max)

        # y_bump_l:
        if self.ori_y_bump_s < 0:
            self.ori_y_bump_l = self.np_random.uniform(low=self.ori_y_bump_s + self.min_bump_distance,
                                                       high=min(self.y_bump_limit_max,
                                                                self.ori_y_bump_s + self.max_bump_distance))
            y_bump_left = self.ori_y_bump_s
            y_bump_right = self.ori_y_bump_l

        else:
            self.ori_y_bump_l = self.np_random.uniform(low=max(self.y_bump_limit_min,
                                                               self.ori_y_bump_s - self.max_bump_distance),
                                                       high=self.ori_y_bump_s - self.min_bump_distance)
            y_bump_left = self.ori_y_bump_l
            y_bump_right = self.ori_y_bump_s

        # y_ur5
        y_ur5_range1 = [self.y_g_left_limit, y_bump_left - self.min_y_g_bump_distance]
        y_ur5_range2 = [y_bump_left + self.min_y_g_bump_distance, y_bump_right - self.min_y_g_bump_distance]
        y_ur5_range3 = [y_bump_right + self.min_y_g_bump_distance, self.y_g_right_limit]
        y_ur5 = self._uniform_ranges([y_ur5_range1, y_ur5_range2, y_ur5_range3])

        # Loads the bumps.
        if self.bump_s is None:
            self.bump_s = EnvObject(str(BUMP_S_URDF_PATH), (self.working_x_g, self.ori_y_bump_s, 0))
        else:
            self.bump_s.reset((self.working_x_g, self.ori_y_bump_s, 0))

        if self.bump_l is None:
            self.bump_l = EnvObject(str(BUMP_L_URDF_PATH), (self.working_x_g, self.ori_y_bump_l, 0))
        else:
            self.bump_l.reset((self.working_x_g, self.ori_y_bump_l, 0))

        # Moves the gripper to the start position.
        # It has to pass by a safe waypoint to avoid any possible collision.
        self.robot.move_pose(target_position=[self.working_x_g, y_ur5, self.lifting_z_g])
        self.robot.move_pose(target_position=[self.working_x_g, y_ur5, self.working_z_g])

        # Resets the bumps-touched flags.
        self.feel_bump_s = False
        self.feel_bump_l = False

        # Resets the state.
        self._update_state()

        self.bump_s_num_contacts_static = len(p.getContactPoints(self.bump_s.body_id))
        self.bump_l_num_contacts_static = len(p.getContactPoints(self.bump_l.body_id))

        return self._get_obs()

    def step(self, action=None):
        """
        Execute action with specified primitive.

        :param action: the action to execute
        :return: (obs, reward, done, info) tuple containing MDP step data.
        """

        reward = 0
        done = False

        # When the action space is in the discrete mode:
        #   0: Left&Soft, 1: Left&Hard, 2: Right&Soft, 3: Right&Hard
        # When the action space is in the continuous mode:
        #   action[0]: step length ratio ([-1, 1])
        #   action[1]: stiffness ratio ([-1, 1])
        step_length_ratio, stiffness_ratio = self._convert_action(action)

        # Executes the determined action if we are lucky.
        if self.np_random.rand() > self.action_failure_prob:
            self._move_gripper(step_length_ratio, stiffness_ratio)

            # Updates the state.
            self._update_state()

            # Reward condition:
            #   the larger bump is moving with an expected distance in the right direction
            #   and the gripper is pushing the larger bump from its left (for avoiding some tricky unexpected cases)
            #   and the smaller bumps has been touched
            #   and the larger bumps has been touched
            if self.y_bump_l - self.ori_y_bump_l > self.pushing_reward_threshold \
                    and self.y_g < self.y_bump_l \
                    and self.feel_bump_s \
                    and self.feel_bump_l:
                reward = 1.0

            # Episode Termination:
            #   when the smaller bump is pushed in either direction.
            #   or when the larger bump is pushed in wrong direction beyond the done limit
            #   or when the larger bump is pushed in right direction beyond the reward limit
            done = bool(
                abs(self.y_bump_s - self.ori_y_bump_s) > self.pushing_done_threshold
                or self.y_bump_l - self.ori_y_bump_l < -self.pushing_done_threshold
                or self.y_bump_l - self.ori_y_bump_l > self.pushing_reward_threshold
            )

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        """
        Gets the observation (y_g, y_bump_s, y_bump_l, theta).
        """

        return np.array([self.y_g / self.y_half_length,
                         self.y_bump_s / self.y_half_length,
                         self.y_bump_l / self.y_half_length,
                         self.theta])

    def _update_state(self):
        """
        Gets the data from sensors and updates the state.
        """

        self.y_g = self._get_raw_y_g()
        self.y_bump_s = self.bump_s.get_base_pose()[0][1]
        self.y_bump_l = self.bump_l.get_base_pose()[0][1]
        self.theta = self._get_theta()

        self.y_ur5 = self._get_raw_y_ur5()

    def _observation(self):
        """
        Called by self._move_gripper() for observing the given states during simulation.
        """

        if not self.feel_bump_s:
            self.feel_bump_s = len(p.getContactPoints(self.bump_s.body_id)) != self.bump_s_num_contacts_static

        if not self.feel_bump_l:
            self.feel_bump_l = len(p.getContactPoints(self.bump_l.body_id)) != self.bump_l_num_contacts_static
