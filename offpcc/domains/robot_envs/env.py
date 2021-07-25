from pathlib import Path

import gym
from gym.utils import seeding
import pybullet as p
import numpy as np

ASSETS_PATH = Path(__file__).resolve().parent / 'assets'


class PomdpRobotEnv(gym.Env):
    """
    An Open AI Gym style base class for all POMDP robot environments.
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

        self.rendering = rendering
        self.discrete = discrete

        if self.rendering:
            display_option = p.GUI
        else:
            display_option = p.DIRECT

        p.connect(display_option)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setTimeStep(1.0 / hz)

        if self.rendering:
            target = p.getDebugVisualizerCamera()[11]
            p.resetDebugVisualizerCamera(
                cameraDistance=1.1,
                cameraYaw=90,
                cameraPitch=-25,
                cameraTargetPosition=target)

        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setGravity(0, 0, -9.8)

        # Stochasticity
        self.action_failure_prob = action_failure_prob

        # numpy random
        self.np_random = None
        self.seed(seed)

    def reset(self):
        """
        Resets the current PyBullet simulation environment.

        :return: the observations of the environment
        """

        return None

    def step(self, action=None):
        """

        Steps the simulation with the given action and returns the observations.

        :param action: actions
        :return: observations, reward, done, {}
        """

        raise NotImplementedError

    def render(self, mode='rgb_array'):
        """
        Renders the environment. PyBullet does not need this method, at least for now.
        """

        pass

    def close(self):
        """
        Closes the simulation environment.
        """

        p.disconnect()

    def seed(self, seed=None):
        """
        Sets the seed for this environment's random number generator(s).

        :param seed: the seed for the random number generator(s)
        :return: the seed
        """

        self.np_random, seed_ = seeding.np_random(seed)

        return seed

    def _uniform_ranges(self, ranges):
        """
        Draws a sample from a uniform distribution on different ranges.

        :param ranges: the ranges to draw a sample
        :return: the sample drawn
        """

        range_lens = []
        for r in ranges:
            range_lens.append(r[1] - r[0])
        range_lens = np.array(range_lens)
        probs = range_lens / np.sum(range_lens)   # longer ranges should be assigned higher probability

        assert np.all(probs >= 0)
        assert np.allclose([1.0], [np.sum(probs)]), "Probs don't sum up to 1."

        index = self.np_random.choice(np.arange(len(ranges)), p=probs)

        return self.np_random.uniform(low=ranges[index][0], high=ranges[index][1])


class EnvObject:
    """
    A class for the object that interacts with the robot in the environment.
    """

    def __init__(self, urdf_path, load_position=(0, 0, 0), load_orientation=(0, 0, 0, 1.0), fixed=False):
        """
        The initialization of the object.

        :param urdf_path: the path of the urdf that describes the object
        :param load_position: the position to load the object
        :param load_orientation: the orientation to load the object
        :param fixed True if this object is created as a fixed object, False otherwise
        """

        self.load_position = load_position
        self.load_orientation = load_orientation

        self.body_id = p.loadURDF(str(urdf_path), load_position, load_orientation)

        if fixed:
            self.fixed_constraint_id = p.createConstraint(
                parentBodyUniqueId=self.body_id,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=(0, 0, 0),
                parentFramePosition=(0, 0, 0),
                childFramePosition=self.load_position,
                childFrameOrientation=self.load_orientation)
        else:
            self.fixed_constraint_id = None

    def get_body_id(self):
        """
        Gets the body id of this object in PyBullet.

        :return: the body id
        """

        return self.body_id

    def reset(self, load_position=None, load_orientation=None):
        """
        Resets this object to the given pose.

        :param load_position: the position to reset the object
        :param load_orientation: the orientation to reset the object
        """

        if load_position is not None:
            self.load_position = load_position

        if load_orientation is not None:
            self.load_orientation = load_orientation

        p.resetBasePositionAndOrientation(bodyUniqueId=self.body_id,
                                          posObj=self.load_position,
                                          ornObj=self.load_orientation)

        if self.fixed_constraint_id is not None:
            p.changeConstraint(userConstraintUniqueId=self.fixed_constraint_id,
                               jointChildPivot=self.load_position,
                               jointChildFrameOrientation=self.load_orientation)

    def get_base_pose(self):
        """
        Gets the base position and orientation of this object.

        :return: the base position and orientation
        """

        return p.getBasePositionAndOrientation(self.body_id)

    def remove(self):
        """
        Removes this object from the PyBullet environment.
        """

        p.removeBody(self.body_id)

    def fix(self):
        """
        Fixes this object if it is not fixed.
        """
        if self.fixed_constraint_id is None:
            self.fixed_constraint_id = p.createConstraint(
                parentBodyUniqueId=self.body_id,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=(0, 0, 0),
                parentFramePosition=(0, 0, 0),
                childFramePosition=self.load_position,
                childFrameOrientation=self.load_orientation)

    def release(self):
        """
        Releases this object if it is fixed.
        """
        if self.fixed_constraint_id is not None:
            p.removeConstraint(self.fixed_constraint_id)
