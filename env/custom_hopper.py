"""Implementation of the Hopper environment supporting
domain randomization optimization.
    
    See more at: https://www.gymlibrary.dev/environments/mujoco/hopper/
"""
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from mujoco_py.generated import const


class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses
        self.domain = domain

        if 'source' in domain:  # Source environment has an imprecise torso mass (-30% shift)
            self.sim.model.body_mass[1] *= 0.7
        
        if domain == "source-normal":
            # hardcoding the normalized parameters suggested by DROPO
            self.param_means = np.array([0.62332, 0.58322, 0.94939])
            self.param_stds = np.array([2.3163, 0.56228, 2.11593])
            self.param_cov = np.diag(self.param_stds**2)


    def set_random_parameters(self):
        """Set random masses"""
        self.set_parameters(self.sample_parameters())

    def get_task_search_bounds(self):
        """Hardcode the search bounds from our experiments"""
        min_task = np.array([0.5, 0.5, 0.5])
        max_task = np.array([10, 10, 10])
        return min_task, max_task
    
    def denormalize_parameters(self, parameters):
        """Denormalize parameters back to their original space
        
            Parameters are assumed to be normalized in
            a space of [0, 4] because we ran DROPO with normalization
        """

        # hardcoding task_dim to be 3 for our purposes
        task_dim = 3

        min_task, max_task = self.get_task_search_bounds()
        parameter_bounds = np.empty((task_dim, 2), float)
        parameter_bounds[:,0] = min_task
        parameter_bounds[:,1] = max_task

        orig_parameters = (parameters * (parameter_bounds[:,1]-parameter_bounds[:,0]))/4 + parameter_bounds[:,0]

        return np.array(orig_parameters)


    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution"""
        
        # These parameters will be passed on to set_parameters
        # set_parameters expects parameters for [1:]
        parameters = np.zeros(len(self.sim.model.body_mass) - 1)

        if self.domain == "source-udr":
            # Apply UDR to thigh,leg,foot
            for i in range(len(parameters)):
                m = self.original_masses[i]
                parameters[i] = self.np_random.uniform(low=0.5*m,high=1.5*m)
        elif self.domain == "source-normal":
            sample = np.random.multivariate_normal(self.param_means, self.param_cov)
            sample = np.clip(sample, 0, 4)

            sample = self.denormalize_parameters(sample)

        # No randomization on the torso mass
        parameters[0] = self.sim.model.body_mass[1]

        return parameters


    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses


    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[1:] = task


    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}


    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])


    def reset_model(self):
        """Reset the environment to a random initial state"""
        
        if self.domain == "source-udr" or self.domain == "source-normal":
            self.set_random_parameters()

        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
        
        # some cosmetic modifications to the rendered video
        self.viewer._run_speed = 0.25
        self.viewer.cam.fixedcamid = 0
        self.viewer.cam.type = const.CAMERA_FIXED


    def set_mujoco_state(self, state):
        """Set the simulator to a specific state

        Parameters:
        ----------
        state: ndarray,
               desired state
        """
        mjstate = deepcopy(self.get_mujoco_state())

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        self.set_sim_state(mjstate)


    def set_sim_state(self, mjstate):
        """Set internal mujoco state"""
        return self.sim.set_state(mjstate)


    def get_mujoco_state(self):
        """Returns current mjstate"""
        return self.sim.get_state()



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

gym.envs.register(
        id="CustomHopper-source-UDR-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source-udr"}
)

gym.envs.register(
        id="CustomHopper-source-normal-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source-normal"}
)