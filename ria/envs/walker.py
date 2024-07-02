import collections
import os
import numpy as np
import tensorflow as tf
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
import gymnasium as gym
import time
from gymnasium.spaces import Box
_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = .025
_STAND_HEIGHT = 1.2
_WALK_SPEED = 1
_RUN_SPEED = 8

class WalkerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, mode='walk', mass_scale_set=[0.75, 1.0, 1.25], damping_scale_set=[0.75, 1.0, 1.25]):
        self.mode = mode
        if mode == 'stand':
            self.move_speed = 0
        elif mode == 'walk':
            self.move_speed = _WALK_SPEED
        elif mode == 'run':
            self.move_speed = _RUN_SPEED
        else:
            raise ValueError("Unsupported mode: choose from 'stand', 'walk', 'run'")
 
        self.mass_scale_set = mass_scale_set
        self.damping_scale_set = damping_scale_set
        self.label_index = None

        # Set the correct model path using dir_path
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(dir_path, "assets", "walker.xml")
        frame_skip = 4

        # Define action space before MujocoEnv initialization
        self.action_space = Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        mujoco_env.MujocoEnv.__init__(self, model_path, frame_skip, observation_space=None)

        obs_shape = self._get_obs().shape
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        # Define observation and action spaces
        self.original_mass = np.copy(self.model.body_mass)
        self.original_damping = np.copy(self.model.dof_damping)

        utils.EzPickle.__init__(self, self.move_speed, self.mass_scale_set, self.damping_scale_set)

    def step(self, action):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        posafter = self.sim.data.qpos[0]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(action).sum()

        standing = rewards.tolerance(self.sim.data.qpos[2],  # torso height
                                     bounds=(_STAND_HEIGHT, float('inf')),
                                     margin=_STAND_HEIGHT/2)
        upright = (1 + self.sim.data.qpos[4]) / 2  # assuming qpos[4] is torso upright
        stand_reward = (3 * standing + upright) / 4
        if self.move_speed == 0:
            reward = stand_reward
        else:
            horizontal_velocity = self.sim.data.qvel[0]  # assuming qvel[0] is horizontal velocity
            move_reward = rewards.tolerance(horizontal_velocity,
                                            bounds=(self.move_speed, float('inf')),
                                            margin=self.move_speed/2,
                                            value_at_margin=0.5,
                                            sigmoid='linear')
            reward = stand_reward * (5 * move_reward + 1) / 6

        done = False
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate(
            [self.sim.data.qpos.flat[1:], np.clip(self.sim.data.qvel.flat, -10, 10)]
        )

    def obs_preproc(self, obs):
        return obs

    def obs_postproc(self, obs, pred):
        return obs + pred

    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        try:
            self.reset_num = int(str(time.time())[-2:])
        except:
            self.reset_num = 1
        # self.np_random.seed(self.reset_num)
        # random_index = self.np_random.randint(len(self.mass_scale_set))
        # self.mass_scale = self.mass_scale_set[random_index]
        # self.label_index = random_index * len(self.damping_scale_set)
        # random_index = self.np_random.randint(len(self.damping_scale_set))
        # self.damping_scale = self.damping_scale_set[random_index]
        # self.label_index = random_index + self.label_index
        # self.change_env()
        return self._get_obs()

    def change_env(self):
        # mass = np.copy(self.original_mass)
        # damping = np.copy(self.original_damping)
        # mass *= self.mass_scale
        # damping *= self.damping_scale

        # self.model.body_mass[:] = mass
        # self.model.dof_damping[:] = damping
        pass

    def change_mass(self, mass):
        self.mass_scale = mass

    def change_damping(self, damping):
        self.damping_scale = damping

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def get_sim_parameters(self):
        return np.array([self.mass_scale, self.damping_scale])

    def num_modifiable_parameters(self):
        return 2

    def log_diagnostics(self, paths, prefix):
        return

    def reward(self, obs, action, next_obs):
        velocity = obs[..., 5]
        alive_bonus = 1.0
        reward = velocity
        reward += alive_bonus
        reward -= 1e-3 * np.square(action).sum(axis=-1)
        return reward

    def tf_reward_fn(self):
        def _thunk(obs, act, next_obs):
            velocity = obs[..., 5]
            alive_bonus = 1.0
            reward = velocity
            reward += alive_bonus
            reward -= 1e-3 * tf.reduce_sum(tf.square(act), axis=-1)
            return reward

        return _thunk

    def get_labels(self):
        return self.label_index

    def seed(self, seed=None):
        if seed is None:
            self._seed = 0
        else:
            self._seed = seed
        super().seed(seed)
