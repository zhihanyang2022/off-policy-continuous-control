import numpy as np
import pybullet as p
from robots.end_effector import EndEffector, ASSETS_PATH

SUCTION_BASE_URDF_PATH = ASSETS_PATH / 'suction' / 'suction-base.urdf'
SUCTION_HEAD_URDF_PATH = ASSETS_PATH / 'suction' / 'suction-head.urdf'


class Suction(EndEffector):
    """
    A class for a simple suction dynamics.
    """

    def __init__(self, ur5_id, ee_id):
        """
        The initialization of the suction.

        To get the suction gripper pose, use p.getLinkState(self.body_id, self.ee_tip_idx),
        but not p.getBasePositionAndOrientation(self.body_id) as the latter is
        about z=0.03m higher and empirically seems worse.

        :param ur5_id: the body id of the UR5 robot to install this suction
        :param ee_id: the link index of the UR5 robot to install this suction
        """

        super().__init__(urdf_path=SUCTION_HEAD_URDF_PATH,
                         load_position=(0.487, 0.109, 0.347),
                         load_orientation=p.getQuaternionFromEuler((np.pi, 0, 0)),
                         ur5_install_joints=np.float32([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi,
                         ee_tip_idx=0)

        # Installs the suction base model (visual only) on the UR5.
        pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
        base = p.loadURDF(str(SUCTION_BASE_URDF_PATH), pose[0], pose[1])
        p.createConstraint(
            parentBodyUniqueId=ur5_id,
            parentLinkIndex=ee_id,
            childBodyUniqueId=base,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0.01))

        # Installs the suction tip model (visual and collision) on the UR5.
        constraint_id = p.createConstraint(
            parentBodyUniqueId=ur5_id,
            parentLinkIndex=ee_id,
            childBodyUniqueId=self.body_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.08))
        p.changeConstraint(constraint_id, maxForce=50)

        # Indicates whether gripper is gripping anything (rigid or def).
        self.activated = False

        # For gripping and releasing rigid objects.
        self.contact_constraint = None

    def reset(self, reset_base=False):
        """
        Resets this suction.

        :param reset_base: True if resetting the gripper to the base pose, False otherwise
        """

        super().reset(reset_base)
        self.release()

    def activate(self, object_id):
        """
        Simulates the suction using a rigid fixed constraint to contacted object.

        :param object_id: the body id of an object to suction
        """

        if not self.activated:
            points = p.getContactPoints(bodyA=self.body_id, linkIndexA=self.ee_tip_idx)

            if points:
                # Handle contact between suction with a rigid object.
                for point in points:
                    contact_body_id, contact_link = point[2], point[4]

                    if contact_body_id == object_id:
                        body_pose = p.getLinkState(self.body_id, self.ee_tip_idx)
                        obj_pose = p.getBasePositionAndOrientation(contact_body_id)
                        world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                        obj_to_body = p.multiplyTransforms(world_to_body[0],
                                                           world_to_body[1],
                                                           obj_pose[0], obj_pose[1])
                        self.contact_constraint = p.createConstraint(
                            parentBodyUniqueId=self.body_id,
                            parentLinkIndex=0,
                            childBodyUniqueId=contact_body_id,
                            childLinkIndex=contact_link,
                            jointType=p.JOINT_FIXED,
                            jointAxis=(0, 0, 0),
                            parentFramePosition=obj_to_body[0],
                            parentFrameOrientation=obj_to_body[1],
                            childFramePosition=(0, 0, 0),
                            childFrameOrientation=(0, 0, 0))

                        self.activated = True

                        return

    def release(self):
        """
        Release the gripped object, only applied if gripper is 'activated'.

        If suction off, detect contact between gripper and objects.
        If suction on, detect contact between picked object and other objects.
        """

        if not self.activated:
            return

        self.activated = False

        # Release gripped rigid object (if any).
        if self.contact_constraint is not None:
            try:
                p.removeConstraint(self.contact_constraint)
                self.contact_constraint = None
            except:  # pylint: disable=bare-except
                pass

    def detect_contact(self):
        """
        Detects a contact with a rigid object.

        When self.activated == False, detects if the suction tip is contacting any object.
        When self.activated == True, detects if the object that is currently attached is contacting
        another external object.
        """

        body, link = self.body_id, self.ee_tip_idx

        # If attaching an object, then use that object as the "source" to check contact.
        if self.activated and self.contact_constraint is not None:
            try:
                info = p.getConstraintInfo(self.contact_constraint)
                body, link = info[2], info[3]  # childBodyUniqueId, childLinkIndex

            except:  # pylint: disable=bare-except
                self.contact_constraint = None
                pass

        # Gets all contact points between the suction and a rigid body.
        points = p.getContactPoints(bodyA=body, linkIndexA=link)

        # When we are using the object attached as the "source", we need to exclude the suction tip (self.body_id)
        # from the contact list.
        if self.activated:
            points = [point for point in points if point[2] != self.body_id]

        # We know if len(points) > 0, contact is made with SOME rigid item.
        if points:
            return True

        return False

    def check_grasp(self):
        """
        Check a grasp for picking success.
        """

        suctioned_object = None

        if self.contact_constraint is not None:
            suctioned_object = p.getConstraintInfo(self.contact_constraint)[2]

        return suctioned_object is not None
