import numpy as np
import pybullet as p

from robots.gripper import Gripper, ASSETS_PATH

RDDA_URDF_PATH = ASSETS_PATH / 'rdda' / 'rdda.urdf'


class Rdda(Gripper):
    """
     A class for the RDDA gripper.
     """

    def __init__(self, ur5_id, ee_id):
        """
        The initialization of the RDDA gripper.

        :param ur5_id: the body id of the UR5 robot to install this RDDA
        :param ee_id: the link index of the UR5 robot to install this RDDA
        """

        super().__init__(urdf_path=RDDA_URDF_PATH,
                         load_position=(0.4746, 0.1092, 0.4195),
                         load_orientation=p.getQuaternionFromEuler((np.pi / 2, np.pi / 2, 0)),
                         ur5_install_joints=np.float32([-1, -0.5, 0.5, 0, 0.5, -1]) * np.pi,
                         ee_tip_idx=1)  # The ee tip of the RDDA gripper is the tip of the wide finger.

        # Installs the RDDA gripper on the UR5.
        self.offset = -0.025
        constraint_id = p.createConstraint(
            parentBodyUniqueId=ur5_id,
            parentLinkIndex=ee_id,
            childBodyUniqueId=self.body_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            parentFrameOrientation=p.getQuaternionFromEuler((np.pi / 2, 0, np.pi / 2)),
            childFramePosition=(0, self.offset, 0))
        p.changeConstraint(constraint_id, maxForce=50)

    def get_position_offset(self):
        """
        Gets the position offset of this RDDA gripper.

        :return: the position offset
        """

        return self.offset, 0, 0
