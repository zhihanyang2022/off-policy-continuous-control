import time
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import gym
from gym import spaces
from pathlib import Path
from gym.utils import seeding

ASSETS_PATH = Path(__file__).resolve().parent / 'assets'


class AntEnv(gym.Env):

    def __init__(self, seed=None, obs_type='coodinate', num_frames_skip=15, rendering=False):

        num_frames_skip = num_frames_skip

        model_name = "ant_reacher.xml"

        initial_joint_pos = np.array([0, 0, 0.55, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        initial_joint_pos = np.reshape(initial_joint_pos, (len(initial_joint_pos), 1))
        initial_joint_ranges = np.concatenate((initial_joint_pos, initial_joint_pos), 1)
        initial_joint_ranges[0] = np.array([-6, 6])
        initial_joint_ranges[1] = np.array([-6, 6])

        initial_state_space = np.concatenate((initial_joint_ranges, np.zeros((len(initial_joint_ranges) - 1, 2))), 0)

        self.seed(seed)

        # The subgoal space in the Ant Reacher task is the desired (x,y) position
        self.cage_max_dim = 8

        self.name = model_name

        MODEL_PATH = ASSETS_PATH / self.name

        # Create Mujoco Simulation
        self.model = load_model_from_path(str(MODEL_PATH))
        self.sim = MjSim(self.model)

        # choose what type of observation that the priest will tell the agent: coordinate - the location of the heaven,
        # something else: direction (left/right) only
        self.obs_type = obs_type

        if self.obs_type in ['coodinate']:
            self.obs_dim = 4  # State will include (x, y) coordinate of the ant + (x, y) coordinate of the heaven
        else:
            self.obs_dim = 3  # State will include (x, y) coordinate of the ant + direction to the heaven (left/right)

        self.action_dim = 2  # desired (x,y) coordinate of the ant

        # Set inital state and goal state spaces
        self.initial_state_space = initial_state_space

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

        self.heaven_hell = [[-6.25, 6.75], [6.25, 6.75]]
        self.priest_pos = [5.25, -5.75]
        self.radius = 1.9

    def render(self, mode='human'):
        pass

    def _scale_action(self, action):
        return action * self.cage_max_dim

    # Get state, which concatenates joint positions and velocities
    def get_state(self, reveal_heave_pos, at_reset=False):

        if self.obs_type in ['coodinate']:
            if at_reset:
                heaven_pos = np.zeros_like(self.heaven_pos)
                return np.concatenate((self.sim.data.qpos[:2], heaven_pos))

            heaven_pos = self.heaven_pos if reveal_heave_pos else np.zeros_like(self.heaven_pos)
            return np.concatenate((self.sim.data.qpos[:2], heaven_pos))
        else:
            if at_reset:
                return np.concatenate((self.sim.data.qpos[:2], 0.0))

            heaven_direction = np.sign(self.heaven_pos[0])
            return np.concatenate((self.sim.data.qpos[:2], heaven_direction))

    # Reset simulation to state within initial state specified by user
    def reset(self):

        # Reset controls
        self.sim.data.ctrl[:] = 0

        coin_face = self.np_random.rand() >= 0.5
        self.heaven_pos = self.heaven_hell[coin_face]
        self.hell_pos = self.heaven_hell[not coin_face]

        # Changing colors of heaven/hell area
        if coin_face:
            self.sim.model.site_rgba[2] = [0, 1, 0, 0.5]
            self.sim.model.site_rgba[4] = [1, 0, 0, 0.5]
        else:
            self.sim.model.site_rgba[4] = [0, 1, 0, 0.5]
            self.sim.model.site_rgba[2] = [1, 0, 0, 0.5]

            # Set initial joint positions and velocities
        for i in range(len(self.sim.data.qpos)):
            self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0], self.initial_state_space[i][1])

        for i in range(len(self.sim.data.qvel)):
            self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],
                                                      self.initial_state_space[len(self.sim.data.qpos) + i][1])

        # Initialize ant's position
        self.sim.data.qpos[0] = 0.0
        self.sim.data.qpos[1] = 0.0

        self.sim.step()

        # Return state
        return self.get_state(False, at_reset=True)

    # Execute low-level action for number of frames specified by num_frames_skip
    def step(self, action):

        # scale action
        position = self._scale_action(action)

        # bring the ant to the desired coordinate
        self.sim.data.qpos[0] = position[0]
        self.sim.data.qpos[1] = position[1]

        if self.visualize:
            self._display_action(position)

        self.sim.step()
        if self.visualize:
            self.viewer.render()

        ant_pos = self.sim.data.qpos[:2]

        d2heaven = np.linalg.norm(ant_pos - self.heaven_pos)
        d2hell = np.linalg.norm(ant_pos - self.hell_pos)

        reward = 0.0
        if (d2heaven < self.radius):
            reward = 1.0
        elif (d2hell < self.radius):
            reward = -1.0

        done = False
        if reward != 0.0:
            done = True

        d2priest = np.linalg.norm(ant_pos - self.priest_pos)
        if (d2priest < self.radius):
            reveal_heave_pos = True
        else:
            reveal_heave_pos = False

        return self.get_state(reveal_heave_pos), reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def _display_action(self, position):
        self.sim.data.mocap_pos[0][:2] = np.copy(position[:2])
        self.sim.model.site_rgba[0][3] = 1