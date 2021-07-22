from gym import spaces
import numpy as np

from domains.robot_envs.env import EnvObject, ASSETS_PATH
from domains.robot_envs.bumps_env import BumpsEnvBase

BUMP1_URDF_PATH = ASSETS_PATH / 'bumps' / 'bump_40_red.urdf'
BUMP2_URDF_PATH = ASSETS_PATH / 'bumps' / 'bump_40_blue.urdf'


class BumpsNormEnv(BumpsEnvBase):
    """
    Description:
        The PyBullet simulation environment of a two-same-bump environment.
    Observation:
         Type: Box(4)
         Num    Observation               Min                        Max
         0      Finger Tip Position Y     self.y_left_limit          self.y_right_limit
         1      Bump #1 Position Y        self.y_bump1_limit_min     self.y_bump1_limit_max
         2      Bump #2 Position Y        self.y_bump2_limit_min     self.y_bump2_limit_max
         3      Finger Angle              - pi / 3                   pi / 3
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
        Reward of 1 for successfully push bump #2 to the right sufficiently far.
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

        # Bumps parameters
        self.bump1 = None  # Placeholder for declaration
        self.bump2 = None  # Placeholder for declaration
        self.y_bump1 = 0
        self.y_bump2 = 0
        self.ori_y_bump1 = -0.2
        self.ori_y_bump2 = 0.2
        self.bump_diameter = 0.04
        self.min_bump_distance = 0.3 * self.y_half_length
        self.max_bump_distance = 0.8 * self.y_half_length
        self.min_y_g_bump_distance = self.bump_diameter
        self.y_bump1_limit_min = 0.7 * self.y_left_limit
        self.y_bump2_limit_min = self.y_bump1_limit_min + self.min_bump_distance
        self.y_bump2_limit_max = 0.7 * self.y_right_limit
        self.y_bump1_limit_max = self.y_bump2_limit_max - self.min_bump_distance

        # Reward/done thresholds
        self.pushing_reward_threshold = 0.01
        self.pushing_done_threshold = 0.001

        # Obs: (y_g, x_bump1, x_bump2, theta)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(2,), dtype=np.float32)

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

        determine_bump_1_first = self.np_random.choice([True, False])

        if determine_bump_1_first:

            # y_bump1
            self.ori_y_bump1 = self.np_random.uniform(low=self.y_bump1_limit_min,
                                                      high=self.y_bump1_limit_max)

            # y_bump2
            self.ori_y_bump2 = self.np_random.uniform(low=self.ori_y_bump1 + self.min_bump_distance,
                                                      high=min(self.ori_y_bump1 + self.max_bump_distance,
                                                               self.y_bump2_limit_max))

        else:

            self.ori_y_bump2 = self.np_random.uniform(low=self.y_bump2_limit_min,
                                                      high=self.y_bump2_limit_max)

            self.ori_y_bump1 = self.np_random.uniform(low=max(self.ori_y_bump2 - self.max_bump_distance,
                                                              self.y_bump1_limit_min),
                                                      high=self.ori_y_bump2 - self.min_bump_distance)

        # y_ur5
        y_ur5_range1 = [self.y_g_left_limit, self.ori_y_bump1 - self.min_y_g_bump_distance]
        y_ur5_range2 = [self.ori_y_bump1 + self.min_y_g_bump_distance, self.ori_y_bump2 - self.min_y_g_bump_distance]
        y_ur5_range3 = [self.ori_y_bump2 + self.min_y_g_bump_distance, self.y_g_right_limit]
        y_ur5 = self._uniform_ranges([y_ur5_range1, y_ur5_range2, y_ur5_range3])

        # Loads the bumps.
        if self.bump1 is None:
            self.bump1 = EnvObject(str(BUMP1_URDF_PATH), (self.working_x_g, self.ori_y_bump1, 0))
        else:
            self.bump1.reset((self.working_x_g, self.ori_y_bump1, 0))

        if self.bump2 is None:
            self.bump2 = EnvObject(str(BUMP2_URDF_PATH), (self.working_x_g, self.ori_y_bump2, 0))
        else:
            self.bump2.reset((self.working_x_g, self.ori_y_bump2, 0))

        # Moves the gripper to the start position.
        # It has to pass by a safe waypoint to avoid any possible collision.
        self.robot.move_pose(target_position=[self.working_x_g, y_ur5, self.lifting_z_g])
        self.robot.move_pose(target_position=[self.working_x_g, y_ur5, self.working_z_g])

        # Resets the state.
        self._update_state()

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
            #   when bump #2 is moving with an expected distance in the right direction
            #   and the gripper is pushing bump #2 from its left (for avoiding some tricky unexpected cases).
            if self.y_bump2 - self.ori_y_bump2 > self.pushing_reward_threshold \
                    and self.y_g < self.y_bump2:
                reward = 1.0

            # Punish condition:
            #   bump #1 is pushed in either direction
            #   bump #2 is pushed in wrong direction
            if (abs(self.y_bump1 - self.ori_y_bump1) > self.pushing_done_threshold or
                    self.y_bump2 - self.ori_y_bump2 < -self.pushing_done_threshold):
                reward = -1.0

            # Episode Termination:
            #   when reward == -1
            #   or when reward == 1
            done = bool(
                reward == -1.0
                or reward == 1.0
                or self.y_g <= self.y_g_left_limit
                or self.y_g >= self.y_g_right_limit
            )

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        """
        Gets the observation (y_g, theta).
        """

        return np.array([self.y_g,
                         self.theta])

    def _update_state(self):
        """
        Gets the data from sensors and updates the state.
        """

        self.y_g = self._get_raw_y_g()
        self.y_bump1 = self.bump1.get_base_pose()[0][1]
        self.y_bump2 = self.bump2.get_base_pose()[0][1]
        self.theta = self._get_theta()

        self.y_ur5 = self._get_raw_y_ur5()
