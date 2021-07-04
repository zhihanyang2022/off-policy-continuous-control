import time

import numpy as np
import pybullet as p

from domains.robots.robot import Robot, ASSETS_PATH

UR5_URDF_PATH = ASSETS_PATH / 'ur5' / 'ur5.urdf'


class Ur5(Robot):
    """
    A class for the UR5 manipulator.
    """

    def __init__(self, ee):
        """
        The initialization of the UR5 robot.

        :param ee: the end effector to be mounted on the UR5 robot
        """

        super().__init__(urdf_path=str(UR5_URDF_PATH))

        # Initializes the end effector.
        self.ee_tip_idx = 9
        self.ee = ee(self.body_id, self.ee_tip_idx)

        # Resets the home joints.
        self.home_joints = self.ee.get_ur5_install_joints()
        self._reset_joints()

    def reset(self):
        """
        Resets this UR5 and its ee.
        """

        self._reset_joints()
        self.ee.reset(reset_base=True)

    def move_joints(self, target_joints, speed=0.001, timeout=10, stop_condition=None, observation=None):
        """
        Moves UR5 to target joints configuration.

        :param target_joints: the target angles
        :param speed: the motion speed
        :param timeout: the max time allowing the robot to move
        :param stop_condition: the condition to stop the motion immediately
        :param observation: the function for the observation task during the movement
        :return: True if timeout, False otherwise
        """

        t0 = time.time()

        while (time.time() - t0) < timeout:
            # Checks if needs to stop immediately.
            if stop_condition is not None and stop_condition():
                return True

            cur_joints = np.float32([p.getJointState(self.body_id, i)[0] for i in self.joints_idx])
            joint_diff_raw = target_joints - cur_joints
            joint_diff = []

            for d in joint_diff_raw:
                if d > np.pi:
                    joint_diff.append(-2 * np.pi + d)
                elif d < -np.pi:
                    joint_diff.append(2 * np.pi + d)
                else:
                    joint_diff.append(d)

            if all(np.abs(joint_diff) < 1e-2):
                return False

            # Moves with the constant velocity.
            norm = np.linalg.norm(joint_diff)
            v = joint_diff / norm if norm > 0 else 0
            step = cur_joints + v * speed
            gains = np.ones(len(self.joints_idx))
            p.setJointMotorControlArray(
                bodyIndex=self.body_id,
                jointIndices=self.joints_idx,
                controlMode=p.POSITION_CONTROL,
                targetPositions=step,
                positionGains=gains)

            p.stepSimulation()

            # Observes the outside env using the observation() function passed in
            # observation() should be declared and implemented by the caller.
            if observation is not None:
                observation()

        print(f'Warning: movej exceeded {timeout} second timeout. Skipping.')

        return True

    def set_pose(self, target_position, target_orientation=None):
        """
        Sets this UR5 to the given end effector pose.

        :param target_position: the target position to set
        :param target_orientation: the target orientation to set
        """

        if target_orientation is None:
            target_orientation = self.get_tip_pose()[1]

        target_position = np.float32(target_position) + self.ee.get_position_offset()
        target_joints = self._solve_ik(target_position, target_orientation)
        gains = np.ones(len(self.joints_idx))
        p.setJointMotorControlArray(
            bodyIndex=self.body_id,
            jointIndices=self.joints_idx,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_joints,
            positionGains=gains)

    def move_pose(self, target_position, target_orientation=None,
                  rectilinear=False, step_num=None,
                  speed=0.001, stop_condition=None, observation=None):
        """
        Moves this UR5 to the given end effector pose.

        :param target_position: the target position to move
        :param target_orientation: the target orientation to move
        :param rectilinear: True if the motion should be rectilinear, false otherwise
        :param step_num: the number of steps to finish the motion
        :param speed: the motion speed
        :param stop_condition: the condition to stop the motion immediately
        :param observation: the function for the observation task during the movement
        """

        current_position, current_orientation = self.get_tip_pose()
        current_position = np.float32(current_position)
        current_orientation = np.float32(current_orientation)

        if target_orientation is None:
            target_orientation = current_orientation

        target_position = np.float32(target_position) + self.ee.get_position_offset()
        target_orientation = np.float32(target_orientation)

        if step_num is None:
            if rectilinear:
                step_num = 10
            else:
                step_num = 1

        position_step_size = (target_position - current_position) / step_num
        orientation_step_size = (target_orientation - current_orientation) / step_num

        for i in range(step_num):
            current_position += position_step_size
            current_orientation += orientation_step_size
            target_joints = self._solve_ik(current_position, current_orientation)
            self.move_joints(target_joints, speed, stop_condition=stop_condition, observation=observation)

    def _solve_ik(self, target_position, target_orientation):
        """
        Calculates the joints configuration with inverse kinematics.

        :param target_position: the target position to set
        :param target_orientation: the target orientation to set
        """

        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.body_id,
            endEffectorLinkIndex=self.ee_tip_idx,
            targetPosition=target_position,
            targetOrientation=target_orientation,
            lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],
            restPoses=self.home_joints.tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5)

        return np.float32(joints)
