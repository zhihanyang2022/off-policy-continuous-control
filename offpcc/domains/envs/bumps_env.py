from gym import spaces
import numpy as np
import pybullet as p

from envs.env import PomdpRobotEnv, ASSETS_PATH
from robots.ur5 import Ur5
from robots.shovel import Shovel

UR5_WORKSPACE_URDF_PATH = ASSETS_PATH / 'workspace' / 'workspace.urdf'
GRID_MARK_URDF_PATH = ASSETS_PATH / 'workspace' / 'grid_mark.urdf'
PLANE_URDF_PATH = ASSETS_PATH / 'plane' / 'plane.urdf'


class BumpsEnvBase(PomdpRobotEnv):
    """
    Description:
        The PyBullet simulation environment of a bump environment.

    Observation:
         Depending on the concrete bumps bump environment derived.

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
        Depending on the concrete bumps bump environment derived.

    Starting State:
        Depending on the concrete bumps bump environment derived.

    Episode Termination:
        Depending on the concrete bumps bump environment derived.
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

        # Y range
        self.y_half_length = 0.4
        self.y_left_limit = -self.y_half_length
        self.y_right_limit = self.y_half_length
        self.y_g_left_limit = self.y_left_limit + 0.02  # To avoid unexpected y_g caused by the finger tip's offset.
        self.y_g_right_limit = self.y_right_limit - 0.02  # To avoid unexpected y_g caused by the finger tip's offset.

        # UR5 parameters
        self.working_x_g = 0.6
        self.x_text = 0.7
        self.working_z_g = 0.2
        self.lifting_z_g = 0.35
        self.default_step_length = 0.05
        self.default_speed = 0.001

        # Gripper parameters
        # Thresholds for pushing a 40mm bump: 0.027
        self.low_stiffness = 0.007
        self.high_stiffness = 0.047
        self.default_angles = [0, ]

        # Temporarily disables rendering to load scene faster.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # Loads the workspace.
        p.loadURDF(str(PLANE_URDF_PATH), [0, 0, -0.001])
        p.loadURDF(str(UR5_WORKSPACE_URDF_PATH), [0.5, 0, 0])

        # Loads the grid markers.
        p.loadURDF(str(GRID_MARK_URDF_PATH), (self.working_x_g, self.y_left_limit, 0))
        p.loadURDF(str(GRID_MARK_URDF_PATH), (self.working_x_g, 0, 0))
        p.loadURDF(str(GRID_MARK_URDF_PATH), (self.working_x_g, self.y_right_limit, 0))

        # Loads the robot.
        self.robot = Ur5(ee=Shovel)

        # Re-enables rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # Action space
        # When the action space is in the discrete mode:
        #   0: Left&Soft, 1: Left&Hard, 2: Right&Soft, 3: Right&Hard
        # When the action space is in the continuous mode:
        #   First: step length ratio [-1, 1]
        #   Second: stiffness ratio [-1, 1]
        if self.discrete:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        # Obs: Depends on the different bump env.
        self.observation_space = None

    def reset(self):
        """
        Performs common reset functionality for all supported robots.
        """

        raise NotImplementedError

    def step(self, action=None):
        """
        Execute action with specified primitive.

        :param action: the action to execute
        :return: (obs, reward, done, info) tuple containing MDP step data.
        """

        raise NotImplementedError

    def _move_gripper(self, step_length_ratio, stiffness_ratio):
        """
        Moves the gripper with the given ratios of step length and stiffness in the allowed range.

        :param step_length_ratio: the ratio of step length [-1, 1]
        :param stiffness_ratio: the ratio of stiffness [-1, 1]
        """

        # Converts from stiffness_ratio [-1, 1] to stiffness [low_stiffness, high_stiffness].
        stiffness = self.low_stiffness + (stiffness_ratio + 1) / 2 * (self.high_stiffness - self.low_stiffness)

        if self.rendering:
            # This code is for visually checking the stiffness.
            p.removeAllUserDebugItems()
            p.addUserDebugText(text="stiffness: " + str(stiffness),
                               textPosition=[self.x_text, 0, 0],
                               textColorRGB=[1.0, 0, 0])

        # Sets the stiffness.
        self.robot.ee.set_joints(self.default_angles, forces=[stiffness, ])

        if step_length_ratio > 0:
            target_y_limit = self.y_right_limit

        elif step_length_ratio < 0:
            target_y_limit = self.y_left_limit
        else:
            return

        # Converts from step_length_ratio [-1, 1] to step length [-0.1, 0.1].
        step_length = step_length_ratio * self.default_step_length

        # Calculates the target y_g.
        target_y_ur5 = self._get_raw_y_ur5() + step_length

        # Checks if the gripper has moved to the boundaries.
        def stop_condition():
            y_ur5 = self._get_raw_y_ur5()

            return ((step_length_ratio < 0) and (y_ur5 < self.y_g_left_limit or y_ur5 < target_y_ur5)) \
                   or ((step_length_ratio > 0) and (y_ur5 > self.y_g_right_limit or y_ur5 > target_y_ur5))

        self.robot.move_pose(target_position=[self.working_x_g, target_y_limit, self.working_z_g],
                             rectilinear=True, speed=self.default_speed,
                             stop_condition=stop_condition, observation=self._observation)

    def _convert_action(self, action):
        """
        Converts the given gym-style action to the ratio of the step length and stiffness.

        When the action space is in the discrete mode:
          0: Left&Soft, 1: Left&Hard, 2: Right&Soft, 3: Right&Hard
        When the action space is in the continuous mode:
          action[0]: step length ratio ([-1, 1])
          action[1]: stiffness ratio ([-1, 1])
        
        :param action: the gym-style action received
        :return: the ratio of the step length and stiffness
        """

        if self.discrete:
            if action == 0:
                step_length_ratio = -1
                stiffness_ratio = -1

            elif action == 1:
                step_length_ratio = -1
                stiffness_ratio = 1

            elif action == 2:
                step_length_ratio = 1
                stiffness_ratio = -1

            elif action == 3:
                step_length_ratio = 1
                stiffness_ratio = 1

            else:
                raise ValueError("Unknown action index received.")

        else:
            step_length_ratio = action[0]
            stiffness_ratio = action[1]

        step_length_ratio = np.clip(a=step_length_ratio, a_min=-1, a_max=1)
        stiffness_ratio = np.clip(a=stiffness_ratio, a_min=-1, a_max=1)

        return step_length_ratio, stiffness_ratio

    def _observation(self):
        """
        Called by self._move_gripper() for observing the given states during simulation.
        """

        pass

    def _get_theta(self):
        """
        Gets the current angle of the probe finger.

        :return: the current angle of the probe finger
        """

        return -self.robot.ee.get_joints()[0][0]

    def _get_raw_y_g(self):
        """
        Gets the raw value of y_g in PyBullet.

        :return: the raw value of y_g
        """

        return self.robot.ee.get_tip_pose()[0][1]

    def _get_raw_y_ur5(self):
        """
        Gets the raw value of z_ur5 in PyBullet.

        :return: the raw value of z_ur5
        """

        return self.robot.get_tip_pose()[0][1]
