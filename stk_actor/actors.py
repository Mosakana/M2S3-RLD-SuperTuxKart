import gymnasium as gym
import numpy as np
from bbrl.agents import Agent
import torch
from stable_baselines3.common.policies import ActorCriticPolicy


class MyWrapper(gym.ActionWrapper):
    def __init__(self, env, option: int):
        super().__init__(env)
        self.option = option

    def action(self, action):
        # We do nothing here
        return action

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


class Actor(Agent):
    """Computes probabilities over action"""

    def forward(self, t: int):
        # Computes probabilities over actions
        pass

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

        print(actions)

        actions = torch.tensor(actions, dtype=torch.float32)

        self.set(("action", t), actions)

class ArgmaxActor(Agent):
    """Actor that computes the action"""

    def forward(self, t: int):
        # Selects the best actions according to the policy
        pass


class SamplingActor(Agent):
    """Samples random actions"""

    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space

    def forward(self, t: int):
        sample = torch.tensor([self.action_space.sample()], dtype=torch.float32)
        self.set(("action", t), sample)
