from pathlib import Path
import pybullet as p

ASSETS_PATH = Path(__file__).resolve().parent / 'assets'


class EndEffector:
    """
    A base class for UR5 end effectors.
    """

    def __init__(self, urdf_path, load_position, load_orientation, ur5_install_joints, ee_tip_idx):
        """
        The initialization of the UR5 end effector.

        :param urdf_path: the path of the urdf that describes the end effector
        :param load_position: the position to load the end effector
        :param load_orientation: the orientation to load the end effector
        :param ur5_install_joints: the home joints of the UR5 robot when installing this end effector
        :param ee_tip_idx: the body id of this end effector
        """

        self.urdf_path = urdf_path
        self.load_position = load_position
        self.load_orientation = load_orientation
        self.ur5_install_joints = ur5_install_joints
        self.ee_tip_idx = ee_tip_idx

        # Loads the model.
        self.body_id = p.loadURDF(str(urdf_path), load_position, load_orientation)

    def get_body_id(self):
        """
        Gets the body id of this end effector in PyBullet.

        :return: the body id
        """

        return self.body_id

    def get_position_offset(self):
        """
        Gets the position offset of this end effector.

        :return: the position offset
        """

        return 0, 0, 0

    def get_ur5_install_joints(self):
        """
        Gets the the home joints of the UR5 robot when installing this end effector.

        :return: the UR5 home joints for this end effector
        """

        return self.ur5_install_joints

    def get_base_pose(self):
        """
        Gets the base position and orientation of this end effector.

        :return: the position and orientation of the base
        """

        return p.getBasePositionAndOrientation(self.body_id)

    def get_tip_pose(self):
        """
        Gets the tip position and orientation of this end effector.

        :return: the position and orientation of the tip
        """

        state = p.getLinkState(self.body_id, self.ee_tip_idx)

        return state[4], state[5]

    def reset(self, reset_base=False):
        """
        Resets this end effector.

        :param reset_base: True if resetting the base pose, False otherwise
        """

        if reset_base:
            p.resetBasePositionAndOrientation(bodyUniqueId=self.body_id,
                                              posObj=self.load_position,
                                              ornObj=self.load_orientation)
