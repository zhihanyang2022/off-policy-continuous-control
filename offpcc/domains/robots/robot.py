from pathlib import Path
import numpy as np
import pybullet as p

ASSETS_PATH = Path(__file__).resolve().parent / 'assets'


class Robot:
    """
    A base class for all robots, including grippers.
    """

    def __init__(self, urdf_path, load_position=(0, 0, 0), load_orientation=(0, 0, 0, 1.0)):
        """
        The initialization of the robot.

        :param urdf_path: the path of the urdf that describes the robot
        :param load_position: the position to load the robot
        :param load_orientation: the orientation to load the robot
        """

        # Load the URDF.
        self.load_position = load_position
        self.load_orientation = load_orientation
        self.body_id = p.loadURDF(urdf_path, load_position, load_orientation)

        # Finds all revolute joints.
        n_joints = p.getNumJoints(self.body_id)
        joints_idx = [p.getJointInfo(self.body_id, i) for i in range(n_joints)]
        self.joints_idx = [j[0] for j in joints_idx if j[2] == p.JOINT_REVOLUTE]
        self.num_joints = len(self.joints_idx)

        # Home joints
        self.home_joints = []

        # End effector tip
        self.ee_tip_idx = 0

    def get_body_id(self):
        """
        Gets the body id of this robot in PyBullet.

        :return: the body id
        """

        return self.body_id

    def reset(self):
        """
        Resets the robot.
        """

        self._reset_joints()

    def get_base_pose(self):
        """
        Gets the base position and orientation of this robot.

        :return: the base position and orientation
        """

        return p.getBasePositionAndOrientation(self.body_id)

    def get_joints(self):
        """
        Gets the current states of all joints.

        :return: the current states of all joints
        """

        return [p.getJointState(self.body_id, j) for j in self.joints_idx]

    def set_joints(self, target_joints):
        """
        Sets all joints to the given angles.

        :param target_joints: the target angles of all joints
        """

        p.setJointMotorControlArray(
            bodyIndex=self.body_id,
            jointIndices=self.joints_idx,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_joints,
            positionGains=np.ones(len(self.joints_idx)))

    def get_tip_pose(self):
        """
        Gets the current end effector pose.

        :return: the position and orientation of the end effector
        """

        state = p.getLinkState(self.body_id, self.ee_tip_idx)

        return state[4], state[5]

    def get_link_state(self, link_index):
        """
        Gets the current state of the given link.

        :param link_index: the index of the link
        :return: the current state of the given link
        """

        return p.getLinkState(self.body_id, link_index)

    def _reset_joints(self):
        """
        Resets all joints to the home angles.
        """

        for i in range(self.num_joints):
            p.resetJointState(self.body_id,
                              jointIndex=self.joints_idx[i],
                              targetValue=self.home_joints[i])
