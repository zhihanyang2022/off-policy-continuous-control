import pybullet as p
from gym import spaces
import numpy as np

from envs.env import EnvObject, ASSETS_PATH
from envs.bumps_env import BumpsEnvBase

BUMP_URDF_PATH = ASSETS_PATH / 'bumps' / 'bump_40_red.urdf'
TARGET_URDF_PATH = ASSETS_PATH / 'bumps' / 'bump_40_virtual.urdf'
RAIL_URDF_PATH = ASSETS_PATH / 'workspace' / 'rail.urdf'


class BumpTargetEnv(BumpsEnvBase):
    """
    Description:
        The PyBullet simulation environment of a single bump environment.
        The target is to move the bump to a random target position.

    Observation:
         Type: Box(3)
         Num    Observation               Min                        Max
         0      Finger Tip Position Y     self.y_left_limit          self.y_right_limit
         1      Bump Position Y           self.y_bump_limit_min      self.y_bump_limit_max
         2      Finger Angle              - pi / 3                   pi / 3
         3      Target Position Y         self.y_bump_limit_min      self.y_bump_limit_max

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
        Reward of 1 for successfully push the bump to the target position.

    Starting State:
        The starting state of the gripper is assigned to y_g = random and theta = 0

    Episode Termination:
        The bump reaches the target position.
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

        # Gripper parameters
        # Thresholds for stably pushing a 40mm bump: 0.032
        self.low_stiffness = 0.007
        self.high_stiffness = 0.057

        # Flags to determine if the target has been reached
        self.target_reached = False
        self.target_at_right = False

        # Bumps parameters
        self.bump = None  # Placeholder for declaration
        self.y_bump = 0
        self.bump_diameter = 0.04
        self.min_y_g_bump_distance = self.bump_diameter
        self.y_bump_limit_min = 0.6 * self.y_left_limit
        self.y_bump_limit_max = 0.6 * self.y_right_limit

        # Loads two hidden boundary rails to limit the x range of the bump
        bumps_radius = self.bump_diameter / 2
        EnvObject(str(RAIL_URDF_PATH), (self.working_x_g - bumps_radius - 0.001, 0, bumps_radius), fixed=True)
        EnvObject(str(RAIL_URDF_PATH), (self.working_x_g + bumps_radius + 0.001, 0, bumps_radius), fixed=True)

        # Target highlighted mark
        self.target_mark = None  # Placeholder for declaration

        # Obs: (y_g, y_bump, theta, y_target)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(4,), dtype=np.float32)

        # Declarations of the gripper's state
        self.y_g = 0
        self.theta = 0
        self.y_target = 0

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

        # y_bump
        ori_y_bump = self.np_random.uniform(low=self.y_bump_limit_min,
                                            high=self.y_bump_limit_max)
        # y_ur5
        y_ur5_range1 = [self.y_bump_limit_min, ori_y_bump - self.min_y_g_bump_distance]
        y_ur5_range2 = [ori_y_bump + self.min_y_g_bump_distance, self.y_bump_limit_max]
        y_ur5 = self._uniform_ranges([y_ur5_range1, y_ur5_range2])

        # y_target
        self.y_target = self._uniform_ranges([y_ur5_range1, y_ur5_range2])

        # Loads the bump.
        if self.bump is None:
            self.bump = EnvObject(str(BUMP_URDF_PATH), (self.working_x_g, ori_y_bump, 0))
        else:
            self.bump.reset((self.working_x_g, ori_y_bump, 0))

        # Loads the target mark
        if self.target_mark is None:
            self.target_mark = EnvObject(str(TARGET_URDF_PATH), (self.working_x_g, self.y_target, 0), fixed=True)
        else:
            self.target_mark.reset((self.working_x_g, self.y_target, 0))

        # Moves the gripper to the start position.
        # It has to pass by a safe waypoint to avoid any possible collision.
        self.robot.move_pose(target_position=[self.working_x_g, y_ur5, self.lifting_z_g])
        self.robot.move_pose(target_position=[self.working_x_g, y_ur5, self.working_z_g])

        # Resets the target-reached flag
        self.target_reached = False
        self.target_at_right = (self.y_target > ori_y_bump)

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
            if self.target_reached:
                reward = 1.0
                done = True

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        """
        Gets the observation (y_g, y_bump, y_bump2, theta).
        """

        return np.array([self.y_g / self.y_half_length,
                         self.y_bump / self.y_half_length,
                         self.theta,
                         self.y_target / self.y_half_length])

    def _update_state(self):
        """
        Gets the data from sensors and updates the state.
        """

        self.y_g = self._get_raw_y_g()
        self.y_bump = self.bump.get_base_pose()[0][1]
        self.theta = self._get_theta()

        self.y_ur5 = self._get_raw_y_ur5()

    def _observation(self):
        """
        Called by self._move_gripper() for observing the given states during simulation.
        """

        y_bump = self.bump.get_base_pose()[0][1]

        if not self.target_reached:
            self.target_reached = (y_bump >= self.y_target) if self.target_at_right else (y_bump <= self.y_target)
