from gym import spaces
import numpy as np
import pybullet as p

from domains.robot_envs.env import PomdpRobotEnv, EnvObject, ASSETS_PATH
from domains.robots.ur5 import Ur5
from domains.robots.rdda import Rdda

UR5_WORKSPACE_URDF_PATH = ASSETS_PATH / 'workspace' / 'workspace.urdf'
GRID_MARK_URDF_PATH = ASSETS_PATH / 'workspace' / 'grid_mark.urdf'
PLANE_URDF_PATH = ASSETS_PATH / 'plane' / 'plane.urdf'
PLATE_HOLDER_URDF_PATH = ASSETS_PATH / 'plate' / 'plate_holder.urdf'

# PyBullet does not support concave objects, so we use a special plate model,
# where the collision is only detected for the upper half of the plate.
PLATE_HALF_URDF_PATH = ASSETS_PATH / 'plate' / 'plate_half.urdf'


class TopPlateEnv(PomdpRobotEnv):
    """
    Description:
    The PyBullet simulation environment of a top-plate environment.
    Observation:
        Type: Box(3)
        Num    Observation            Min                     Max
        0      Gripper Position       self.z_down_limit       self.z_up_limit
        1      Top plate Position     z_plate0                z_plate_max
        2      Gripper Angle          - pi / 3                pi / 3
    Actions (discrete mode):
        Type: Discrete(3)
        Num    Action
        0      moving down
        1      Moving up
        2      Grasp
    Actions (continuous mode):
        Type: Box(2)
        Num    Action             Range
        0      step length ratio  [-1, 1]
        1      grasp boolean      [-1, 1]
        When the grasp boolean < 0, the gripper is in moving mode. Otherwise, it will stop
        and grasp.
    Reward:
        Reward of 1 for successfully grasp the top plate
    Starting State:
        The starting state of the gripper is assigned to z_g = [z_g_down_limit, z_plate_base]
        and theta = 0
    Episode Termination:
        The gripper tries to grasp.
    """

    def __init__(self, rendering=True, hz=24, seed=None, discrete=False, action_failure_prob=-1.0):
        """
        The initialization of the PyBullet simulation environment.
        :param rendering: True if rendering, False otherwise
        :param hz: Hz for p.setTimeStep
        :param seed: the random seed
        :param discrete: True if action is discrete, False otherwise
        :param action_failure_prob: determines the probability that an action fails
        """

        super().__init__(rendering, hz, seed, discrete, action_failure_prob)

        # UR5 parameters
        self.working_x_g = 0.6
        self.x_unused_plates = 0.4
        self.x_text = 0.7
        self.working_y_g = 0
        self.y_plate_holder = 0.315
        self.lifting_z_g = 0.36
        self.default_step_length = 0.02
        self.default_speed = 0.004

        # Gripper parameters
        self.default_angles = [0, 1]
        self.grasp_angles = [0, 0]

        # Plate parameters
        self.dist_between_plate = 0.014
        self.max_num_plate = 10
        self.num_plate_selected = 0
        self.plates = []

        # Temporarily disables rendering to load scene faster.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # Loads the workspace.
        p.loadURDF(str(PLANE_URDF_PATH), [0, 0, -0.001])
        p.loadURDF(str(UR5_WORKSPACE_URDF_PATH), [0.5, 0, 0])

        # Loads the robot.
        self.robot = Ur5(ee=Rdda)

        # Load the plate holder.
        plate_holder = EnvObject(str(PLATE_HOLDER_URDF_PATH),
                                 (self.working_x_g, self.y_plate_holder, 0),
                                 fixed=True)
        self.z_plate_base = p.getLinkState(plate_holder.body_id, 0)[0][2]

        # Loads all plates.
        for i in range(self.max_num_plate):
            plate = EnvObject(str(PLATE_HALF_URDF_PATH),
                              (self.working_x_g, self.y_plate_holder, self.z_plate_base + i * self.dist_between_plate),
                              fixed=True)

            self.plates.append(plate)

        # Re-enables rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # Z range
        self.z_down_limit = 0
        self.z_up_limit = self.plates[-1].get_base_pose()[0][2] + 0.25
        self.z_g_down_limit = self.z_down_limit + 0.05  # To avoid unexpected z_g caused by the finger tip's offset.
        self.z_g_up_limit = self.z_up_limit - 0.02  # To avoid unexpected z_g caused by the finger tip's offset.

        # Action space
        # When the action space is in the discrete mode:
        #   1: MoveDown, 2: MoveUP, 3:Grasp
        # When the action space is in the continuous mode:
        #   First: step length ratio [-1, 1]
        #   Second: grasp boolean: [-1, 1]
        if self.discrete:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        # Obs: (z_g, y_bump1, y_bump2, theta)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(4,), dtype=np.float32)

        # Declarations of the state parameters.
        self.z_g = 0
        self.z_t = 0  # The top (highest) plate
        self.theta = 0

        # Declarations of the auxiliary internal state parameters.
        self.z_t_1 = 0  # The second highest plate
        self.z_ur5 = 0

    def reset(self):
        """
        Performs common reset functionality for all supported robots.
        """

        # Resets the robot to the home position.
        self.robot.reset()
        self.robot.ee.set_joints(self.default_angles, velocities=[0.2, 1])

        # Determines the number of plates to use.
        self.num_plate_selected = self.np_random.randint(1, self.max_num_plate + 1)

        # Resets all plates.
        for i in range(self.max_num_plate):
            if i < self.num_plate_selected:
                # Plates to use.
                z_plate = self.z_plate_base + i * self.dist_between_plate
                plate_position = (self.working_x_g, self.y_plate_holder, z_plate)
            else:
                # Unused plates
                z_plate = (i - self.num_plate_selected) * self.dist_between_plate
                plate_position = (self.x_unused_plates, self.y_plate_holder, z_plate)

            self.plates[i].reset(plate_position)

        self.z_t = p.getLinkState(self.plates[self.num_plate_selected - 1].body_id, 0)[0][2]
        self.z_t_1 = p.getLinkState(self.plates[self.num_plate_selected - 2].body_id, 0)[0][2]

        # Determines the start state of z_ur5. It should be between self.z_g_down_limit and the z_plate0.
        z_ur5 = self.np_random.uniform(low=self.z_g_down_limit, high=self.z_plate_base)

        # Moves the gripper to the start position.
        # It has to pass by a safe waypoint to avoid any possible collision.
        self.robot.set_pose(target_position=[self.working_x_g, self.working_y_g, self.lifting_z_g])
        self.robot.set_pose(target_position=[self.working_x_g, self.working_y_g, z_ur5],
                            target_orientation=p.getQuaternionFromEuler((0, np.pi / 2, 0)))

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
        #   1: MoveDown, 2: MoveUP, 3:Grasp
        # When the action space is in the continuous mode:
        #   action[0]: step length ratio ([-1, 1])
        #   action[1]: grasp boolean ([-1, 1])
        #   The gripper moves with the step length ratio action[0] if action[1] < 0,
        #   or stop and grasp if action[1] >= 0
        if self.discrete:
            if action == 0:
                step_length_ratio = -1
                grasp_boolean = -1

            elif action == 1:
                step_length_ratio = 1
                grasp_boolean = -1

            elif action == 2:
                step_length_ratio = 0
                grasp_boolean = 1

            else:
                raise ValueError("Unknown action index received.")

        else:
            step_length_ratio = action[0]
            grasp_boolean = action[1]

        # Executes the determined action if we are lucky.
        if self.np_random.rand() > self.action_failure_prob:
            if grasp_boolean < 0:
                self._move_gripper(step_length_ratio=step_length_ratio)

            else:
                done = True

                # Reward condition:
                #   when the gripper does a grasp.
                if self._grasp():
                    reward = 1.0

        # Updates the state.
        self._update_state()

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        """
        Gets the observation (z_g, y_bump1, y_bump2, theta).
        """

        return np.array([self.z_g / self.z_up_limit,
                         self.z_t / self.z_up_limit,
                         self.theta])

    def _update_state(self):
        """
        Gets the data from sensors and updates the state.
        """

        self.z_g = self._get_raw_z_g()
        self.theta = self._get_theta()
        self.z_ur5 = self._get_raw_z_ur5()

    def _move_gripper(self, step_length_ratio):
        """
        Moves the gripper with the given ratios of step length in the allowed range.
        :param step_length_ratio: the ratio of step length [-1, 1]
        """

        if step_length_ratio > 0:
            target_z_limit = self.z_up_limit

        elif step_length_ratio < 0:
            target_z_limit = self.z_down_limit

        else:
            return

        # Converts from step_length_ratio [-1, 1] to step length [-0.1, 0.1].
        step_length = step_length_ratio * self.default_step_length

        # Calculates the target z_g.
        target_z_ur5 = self._get_raw_z_ur5() + step_length

        # Checks if the gripper has moved to the boundaries.
        def stop_condition():
            z_ur5 = self._get_raw_z_ur5()

            return ((step_length_ratio < 0) and (z_ur5 < self.z_g_down_limit or z_ur5 < target_z_ur5)) \
                   or ((step_length_ratio > 0) and (z_ur5 > self.z_g_up_limit or z_ur5 > target_z_ur5))

        self.robot.move_pose(target_position=[self.working_x_g, self.working_y_g, target_z_limit],
                             rectilinear=True, speed=self.default_speed, stop_condition=stop_condition)

    def _grasp(self):
        """
        Does a grasp.
        :return: True if the grasp succeed, False otherwise
        """

        # This code is for visually checking the grasp.
        if self.rendering:
            p.addUserDebugText(text="GRASP", textPosition=[self.x_text, 0, 0], textColorRGB=[1.0, 0, 0])

            z_ur5 = self._get_raw_z_ur5()
            self.robot.move_pose(target_position=[self.working_x_g, self.working_y_g - 0.1, z_ur5])
            self.robot.move_pose(target_position=[self.working_x_g, self.working_y_g, z_ur5])
            self.robot.ee.set_joints(self.grasp_angles, velocities=[1, 1])

            for i in range(500):
                p.stepSimulation()

            p.removeAllUserDebugItems()

        z_g = self._get_raw_z_g()

        # Right timing for grasp:
        #   1. The finger is between the top plate and the second highest plate
        #   2. The finger is touching the top plate from above
        return (self.z_t_1 < z_g <= self.z_t) or (self.z_t < z_g and p.getContactPoints(self.robot.ee.body_id))

    def _get_theta(self):
        """
        Gets the current angle of the probe finger.
        :return: the current angle of the probe finger
        """

        return -self.robot.ee.get_joints()[0][0]

    def _get_raw_z_g(self):
        """
        Gets the raw value of z_g in PyBullet.
        :return: the raw value of z_g
        """

        return self.robot.ee.get_tip_pose()[0][2]

    def _get_raw_z_ur5(self):
        """
        Gets the raw value of z_ur5 in PyBullet.
        :return: the raw value of z_ur5
        """

        return self.robot.get_tip_pose()[0][2]