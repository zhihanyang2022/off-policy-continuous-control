import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import gym
from gym import spaces
from pathlib import Path
from gym.utils import seeding

ASSETS_PATH = Path(__file__).resolve().parent / 'assets'


def ur5_bound_angle(angle):
    bounded_angle = np.absolute(angle) % (2 * np.pi)
    if angle < 0:
        bounded_angle = -bounded_angle

    return bounded_angle


class Ur5Env(gym.Env):

    def __init__(self, seed=None, num_frames_skip=15, rendering=False):

        model_name = "ur5_reacher.xml"

        initial_joint_pos = np.array([5.96625837e-03, 3.22757851e-03, -1.27944547e-01])
        initial_joint_pos = np.reshape(initial_joint_pos, (len(initial_joint_pos), 1))
        initial_joint_ranges = np.concatenate((initial_joint_pos, initial_joint_pos), 1)
        initial_joint_ranges[0] = np.array([-np.pi / 8, np.pi / 8])
        initial_state_space = np.concatenate((initial_joint_ranges, np.zeros((len(initial_joint_ranges), 2))), 0)

        angle_threshold = np.deg2rad(10)
        self.end_goal_thresholds = np.array([angle_threshold, angle_threshold, angle_threshold])

        self.project_state_to_end_goal = lambda sim, state: np.array(
            [ur5_bound_angle(sim.data.qpos[i]) for i in range(len(sim.data.qpos))])

        switch_angle_threshold = np.deg2rad(10)
        self.switch_thresholds = np.array([switch_angle_threshold, switch_angle_threshold, switch_angle_threshold])

        self.goal_space_test = [[-np.pi, np.pi], [-np.pi / 4, 0], [-np.pi / 4, np.pi / 4]]
        self.action_scale = np.array([np.pi, np.pi / 8, np.pi / 4])
        self.action_offset = np.array([0.0, -np.pi / 8, 0.0])

        MODEL_PATH = ASSETS_PATH / model_name

        # Create Mujoco Simulation
        self.model = load_model_from_path(str(MODEL_PATH))
        self.sim = MjSim(self.model)

        # Set dimensions and ranges of states, actions, and goals in order to configure actor/critic networks
        self.obs_dim = len(
            self.sim.data.qpos) + 3  # State will include (i) joint angles and coordinate of the target position
        self.action_dim = 3  # desired angles for 3 joints

        self.switch_pos = [0.0, 0.0, -np.pi / 4]

        # Set inital state and goal state spaces
        self.initial_state_space = initial_state_space
        self.subgoal_colors = ["Magenta", "Green", "Red", "Blue", "Cyan", "Orange", "Maroon", "Gray", "White", "Black"]

        # Implement visualization if necessary
        self.visualize = rendering  # Visualization boolean
        if self.visualize:
            self.viewer = MjViewer(self.sim)
        self.num_frames_skip = num_frames_skip

        # For Gym interface
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.action_dim,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        self.near_switch_timestamp = -1
        self.steps_cnt = 0
        self.target_visible_duration = 10  # number of timestep that the target is visible to the agent (after the switch is turned on)

        self.seed(seed)

    def _scale_action(self, action):
        return action * self.action_scale + self.action_offset

    # Get state, which concatenates joint positions and velocities
    def get_state(self, target_pos):
        return np.concatenate((self.sim.data.qpos, target_pos))

    # Reset simulation to state within initial state specified by user
    def reset(self):

        self.steps_cnt = 0
        self.near_switch_timestamp = -1

        goal_possible = False
        while not goal_possible:
            end_goal = np.zeros(shape=(3,))
            end_goal[0] = self.np_random.uniform(self.goal_space_test[0][0], self.goal_space_test[0][1])
            end_goal[1] = self.np_random.uniform(self.goal_space_test[1][0], self.goal_space_test[1][1])
            end_goal[2] = self.np_random.uniform(self.goal_space_test[2][0], self.goal_space_test[2][1])

            # Next need to ensure chosen joint angles result in achievable task (i.e., desired end effector position is above ground)

            theta_1 = end_goal[0]
            theta_2 = end_goal[1]
            theta_3 = end_goal[2]

            # shoulder_pos_1 = np.array([0,0,0,1])
            upper_arm_pos_2 = np.array([0, 0.13585, 0, 1])
            forearm_pos_3 = np.array([0.425, 0, 0, 1])
            wrist_1_pos_4 = np.array([0.39225, -0.1197, 0, 1])

            # Transformation matrix from shoulder to base reference frame
            T_1_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]])

            # Transformation matrix from upper arm to shoulder reference frame
            T_2_1 = np.array(
                [[np.cos(theta_1), -np.sin(theta_1), 0, 0], [np.sin(theta_1), np.cos(theta_1), 0, 0], [0, 0, 1, 0],
                 [0, 0, 0, 1]])

            # Transformation matrix from forearm to upper arm reference frame
            T_3_2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2), 0], [0, 1, 0, 0.13585],
                              [-np.sin(theta_2), 0, np.cos(theta_2), 0], [0, 0, 0, 1]])

            # Transformation matrix from wrist 1 to forearm reference frame
            T_4_3 = np.array(
                [[np.cos(theta_3), 0, np.sin(theta_3), 0.425], [0, 1, 0, 0], [-np.sin(theta_3), 0, np.cos(theta_3), 0],
                 [0, 0, 0, 1]])

            forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
            wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3]

            # Make sure wrist 1 pos is above ground so can actually be reached
            if np.absolute(end_goal[0]) > np.pi / 4 and forearm_pos[2] > 0.05 and wrist_1_pos[2] > 0.15:
                goal_possible = True

        self.target_pos = end_goal

        # Set initial joint positions and velocities
        for i in range(len(self.sim.data.qpos)):
            self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0], self.initial_state_space[i][1])

        for i in range(len(self.sim.data.qvel)):
            self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],
                                                      self.initial_state_space[len(self.sim.data.qpos) + i][1])

        self.sim.step()

        # Not reveal the target info at reset
        return self.get_state(np.zeros_like(self.target_pos))

    # Execute low-level action for number of frames specified by num_frames_skip
    def step(self, action):

        self.steps_cnt += 1

        desired_angles = self._scale_action(action)

        if self.visualize:
            self.display_action(desired_angles)
            self.display_target_pos()
            self.display_switch()

        # Set the joint angles to the desired ones
        self.sim.data.qpos[:] = desired_angles
        self.sim.data.qvel[:] = np.zeros(shape=(3,))

        self.sim.step()
        if self.visualize:
            self.viewer.render()

        hindsight_goal = self.project_state_to_end_goal(self.sim, False)

        # Check if the gripper is within the switch area
        near_switch = True
        for i in range(len(hindsight_goal)):
            if np.absolute(self.switch_pos[i] - hindsight_goal[i]) > self.switch_thresholds[i]:
                near_switch = False
                break

        # Near the switch, record the timestamp
        if near_switch:
            self.near_switch_timestamp = self.steps_cnt

        # If the switch is turned on, let the agent see the target for a number of timesteps
        if self.near_switch_timestamp > 0 and (
                self.steps_cnt - self.near_switch_timestamp) <= self.target_visible_duration:
            target_pos = np.copy(self.target_pos)
        else:
            target_pos = np.zeros_like(self.target_pos)

        # Check if the gripper is within the goal achievement threshold
        goal_achieved = True
        for i in range(len(hindsight_goal)):
            if np.absolute(self.target_pos[i] - hindsight_goal[i]) > self.end_goal_thresholds[i]:
                goal_achieved = False
                break

        # Calculate reward
        reward = 0.0
        if goal_achieved:
            reward = 1.0

        if reward == 0:
            done = False
        else:
            done = True

        return self.get_state(target_pos), reward, done, {}

    def display_action(self, action):
        joint_pos = self._angles2jointpos(action)
        for i in range(3):
            self.sim.data.mocap_pos[i] = joint_pos[i]

    def display_switch(self):
        joint_pos = self._angles2jointpos(self.switch_pos)

        for i in range(3):
            self.sim.data.mocap_pos[3 + i] = joint_pos[i]

    def display_target_pos(self):
        joint_pos = self._angles2jointpos(self.target_pos)

        for i in range(3):
            self.sim.data.mocap_pos[6 + i] = joint_pos[i]

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def _angles2jointpos(self, angles):
        theta_1 = angles[0]
        theta_2 = angles[1]
        theta_3 = angles[2]

        upper_arm_pos_2 = np.array([0, 0.13585, 0, 1])
        forearm_pos_3 = np.array([0.425, 0, 0, 1])
        wrist_1_pos_4 = np.array([0.39225, -0.1197, 0, 1])

        # Transformation matrix from shoulder to base reference frame
        T_1_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]])

        # Transformation matrix from upper arm to shoulder reference frame
        T_2_1 = np.array(
            [[np.cos(theta_1), -np.sin(theta_1), 0, 0], [np.sin(theta_1), np.cos(theta_1), 0, 0], [0, 0, 1, 0],
             [0, 0, 0, 1]])

        # Transformation matrix from forearm to upper arm reference frame
        T_3_2 = np.array([[np.cos(theta_2), 0, np.sin(theta_2), 0],
                          [0, 1, 0, 0.13585],
                          [-np.sin(theta_2), 0, np.cos(theta_2), 0],
                          [0, 0, 0, 1]])

        # Transformation matrix from wrist 1 to forearm reference frame
        T_4_3 = np.array([[np.cos(theta_3), 0, np.sin(theta_3), 0.425],
                          [0, 1, 0, 0],
                          [-np.sin(theta_3), 0, np.cos(theta_3), 0],
                          [0, 0, 0, 1]])

        # Determine joint position relative to original reference frame
        upper_arm_pos = T_1_0.dot(T_2_1).dot(upper_arm_pos_2)[:3]
        forearm_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(forearm_pos_3)[:3]
        wrist_1_pos = T_1_0.dot(T_2_1).dot(T_3_2).dot(T_4_3).dot(wrist_1_pos_4)[:3]

        joint_pos = [upper_arm_pos, forearm_pos, wrist_1_pos]

        return joint_pos