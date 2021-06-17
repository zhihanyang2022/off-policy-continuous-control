import numpy as np
import pybullet as p

from robots.end_effector import EndEffector, ASSETS_PATH

SPATULA_URDF_PATH = ASSETS_PATH / 'spatula' / 'spatula-base.urdf'


class Spatula(EndEffector):
    """
    A class for a simple spatula for pushing.
    """

    def __init__(self, ur5_id, ee_id):
        """
        The initialization of the spatula.

        :param ur5_id: the body id of the UR5 robot to install this spatula
        :param ee_id: the link index of the UR5 robot to install this spatula
        """

        super().__init__(urdf_path=SPATULA_URDF_PATH,
                         load_position=(0.487, 0.109, 0.438),
                         load_orientation=p.getQuaternionFromEuler((np.pi, 0, 0)),
                         ur5_install_joints=np.float32([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi,
                         ee_tip_idx=0)

        # Installs the spatula on the UR5.
        constraint_id = p.createConstraint(
            parentBodyUniqueId=ur5_id,
            parentLinkIndex=ee_id,
            childBodyUniqueId=self.body_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            parentFrameOrientation=p.getQuaternionFromEuler((0, 0, np.pi / 2)),
            childFramePosition=(0, 0, 0.01))
        p.changeConstraint(constraint_id, maxForce=50)
