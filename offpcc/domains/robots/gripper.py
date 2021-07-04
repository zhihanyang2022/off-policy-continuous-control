import numpy as np
import pybullet as p

from domains.robots.end_effector import EndEffector, ASSETS_PATH

RDDA_URDF_PATH = ASSETS_PATH / 'rdda' / 'rdda.urdf'


class Gripper(EndEffector):
    """
     A class for the UR5 grippers.
     """

    def __init__(self, urdf_path, load_position, load_orientation, ur5_install_joints, ee_tip_idx):
        """
        The initialization of the UR5 gripper.

        :param urdf_path: the path of the urdf that describes the gripper
        :param load_position: the position to load the gripper
        :param load_orientation: the orientation to load the gripper
        :param ur5_install_joints: the home joints of the UR5 robot when installing this gripper
        :param ee_tip_idx: the body id of this gripper
        """

        super().__init__(urdf_path, load_position, load_orientation, ur5_install_joints, ee_tip_idx)

        # Finds all revolute joints.
        n_joints = p.getNumJoints(self.body_id)
        joints_idx = [p.getJointInfo(self.body_id, i) for i in range(n_joints)]
        self.joints_idx = [j[0] for j in joints_idx if j[2] == p.JOINT_REVOLUTE]
        self.num_joints = len(self.joints_idx)

        # Resets the home joints.
        self.home_joints = np.zeros(self.num_joints)
        self._reset_joints()

        # Default joint parameters
        self.joints_default_velocities = np.ones(self.num_joints)
        self.joints_default_forces = np.ones(self.num_joints) * 0.01

        # Current joint state
        self.joints_current_velocities = self.joints_default_velocities
        self.joints_current_forces = self.joints_default_forces

    def reset(self, reset_base=False):
        """
        Resets this gripper.

        :param reset_base: True if resetting the gripper to the base pose, False otherwise
        """

        super().reset(reset_base)
        self._reset_joints()

    def get_joints(self):
        """
        Gets the current states of all joints.

        :return: the current states of all joints
        """

        return [p.getJointState(self.body_id, j) for j in self.joints_idx]

    def set_joints(self, target_joints, velocities=None, forces=None):
        """
        Sets this gripper to the given joints configuration.

        :param target_joints: the target angles
        :param velocities: the max velocities that the joints can move with
        :param forces: the max forces that the joints can move with
        """

        self.joints_current_velocities = velocities if velocities is not None else self.joints_default_velocities
        self.joints_current_forces = forces if forces is not None else self.joints_default_forces

        for i in range(self.num_joints):
            p.setJointMotorControl2(
                bodyIndex=self.body_id,
                jointIndex=self.joints_idx[i],
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_joints[i],
                maxVelocity=self.joints_current_velocities[i],
                force=self.joints_current_forces[i])

    def _reset_joints(self):
        """
        Resets all joints to the home angles.
        """

        for i in range(self.num_joints):
            p.resetJointState(self.body_id,
                              jointIndex=self.joints_idx[i],
                              targetValue=self.home_joints[i])
