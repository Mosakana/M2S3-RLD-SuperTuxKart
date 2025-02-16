import gymnasium as gym
import numpy as np
from bbrl.agents import Agent
import torch

class DictObsToBoxWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict), \
        'The observation space must be a Dict space'
        self.discrete_space = env.observation_space.spaces['discrete']
        self.continuous_space = env.observation_space.spaces['continuous']

        self.discrete_nvec = self.discrete_space.nvec
        self.discrete_dim = len(self.discrete_nvec)

        low = np.zeros(self.discrete_dim, dtype=np.float32)
        high = (self.discrete_nvec - 1).astype(np.float32)

        new_obs_spaces = {
            'discrete': gym.spaces.Box(low=low, high=high, shape=(self.discrete_dim,), dtype=np.float32),
            'continuous': self.continuous_space
        }
        self.observation_space = gym.spaces.Dict(new_obs_spaces)

    def observation(self, obs):
        discrete_int = obs['discrete']
        discrete_float = discrete_int.astype(np.float32)

        cont_part = obs['continuous']

        return {
            'discrete': discrete_float,
            'continuous': cont_part
        }

class MultiDiscreteToBoxWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete), \
        'The action space must be a MultiDiscrete space'

        self.md_action_space = env.action_space
        self.nvec = self.md_action_space.nvec
        self.k = len(self.nvec)

        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.k,),
            dtype=np.float32
        )

    def action(self, box_action):
        discrete_action = []
        for i in range(self.k):
            val = box_action[i] * self.nvec[i]
            val_int = int(np.floor(val))
            val_int = np.clip(val_int, 0, self.nvec[i] - 1)
            discrete_action.append(val_int)

        return np.array(discrete_action, dtype=np.int64)

class FixDictActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.fire_state = 0

        assert isinstance(env.action_space, gym.spaces.Dict), \
            "Expecting a Dict action space."
        assert 'continuous' in env.action_space.spaces and 'discrete' in env.action_space.spaces, \
            "Must have 'continuous' and 'discrete' keys."
        cont_space = env.action_space.spaces['continuous']
        disc_space = env.action_space.spaces['discrete']
        assert isinstance(cont_space, gym.spaces.Box)
        assert isinstance(disc_space, gym.spaces.MultiDiscrete)

        low = np.array([-1.0, 0.0], dtype=np.float32)
        high = np.array([ 1.0, 1.0], dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(2,),
            dtype=np.float32
        )

    def action(self, box_act):
        steer = box_act[0]
        drift_float = box_act[1]

        if drift_float >= 0.5 and abs(steer) >= 0.5:
            drift = 1
        else:
            drift = 0

        if self.fire_state == 0:
            self.fire_state = 1
        else:
            self.fire_state = 0

        original_cont = np.array([1.0, steer], dtype=np.float32)  # acceleration=1, steer=? 

        # discrete
        # brake=0, drift=?, fire=1->0->1, nitro=1, rescue=0
        original_disc = np.array([0, 0, self.fire_state, 1, 0], dtype=np.int64)

        original_action = {
            'continuous': original_cont,
            'discrete': original_disc
        }

        return original_action

class DriftRewardWrapper(gym.Wrapper):
    def __init__(self, env, drift_bonus=0.1, drift_threshold=0.5):
        super(DriftRewardWrapper, self).__init__(env)
        self.drift_bonus = drift_bonus
        self.drift_threshold = drift_threshold

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        drift_value = action['discrete'][1]
        steer_value = action['continuous'][1]

        if drift_value > self.drift_threshold and abs(steer_value) >= 0.5:
            reward += self.drift_bonus

        return obs, reward, terminated, truncated, info

class SB3PolicyActor(Agent):
    def __init__(self, sb3_policy, deterministic=False):
        super().__init__()
        self.sb3_policy = sb3_policy
        self.deterministic = deterministic

    def forward(self, t, **kwargs):
        obs_d = self.get(("env/env_obs/discrete", t))
        obs_c = self.get(("env/env_obs/continuous", t))

        obs = {
            'discrete': obs_d,
            'continuous': obs_c
        }

        actions, _ = self.sb3_policy.predict(obs, deterministic=self.deterministic)

        actions = torch.tensor(actions, dtype=torch.float32)

        self.set(("action", t), actions)
