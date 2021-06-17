import numpy as np
import pybullet as p

from robots.gripper import Gripper, ASSETS_PATH

Robotiq_URDF_PATH = ASSETS_PATH / 'robotiq' / 'robotiq_2f_85.urdf'


class Robotiq(Gripper):
    """
     A class for the Robotiq gripper.
     """

    def __init__(self, ur5_id, ee_id):
        """
        The initialization of the Robotiq gripper.

        :param ur5_id: the body id of the UR5 robot to install this Robotiq
        :param ee_id: the link index of the UR5 robot to install this Robotiq
        """

        super().__init__(urdf_path=Robotiq_URDF_PATH,
                         load_position=(0.4868, 0.1093, 0.431594),
                         load_orientation=p.getQuaternionFromEuler((np.pi, 0, 0)),
                         ur5_install_joints=np.float32([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi,
                         ee_tip_idx=0)  # FIXME

        # Installs the Robotiq gripper on the UR5.
        constraint_id = p.createConstraint(
            parentBodyUniqueId=ur5_id,
            parentLinkIndex=ee_id,
            childBodyUniqueId=self.body_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            parentFrameOrientation=p.getQuaternionFromEuler((0, 0, np.pi / 2)),
            childFramePosition=(0, 0, 0))
        p.changeConstraint(constraint_id, maxForce=50)
