from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym

import inspect
from pathlib import Path

# Imports our Actor class
# IMPORTANT: note the relative import
from .actors import SB3PolicyActor, DictObsToBoxWrapper, FixDictActionWrapper, DriftRewardWrapper
from stable_baselines3 import SAC
from .tqc_policy import MultiInputPolicy

#: The base environment name
env_name = "supertuxkart/flattened-v0"

#: Player name
player_name = "Winux"


def get_actor(
    state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
) -> Agent:
    if state and len(state) != 0:
        policy = MultiInputPolicy(observation_space, action_space, lr_schedule=lambda x: 0.0, net_arch=dict(
            pi=[256, 256, 128, 128],
            qf=[256, 256, 128, 128]
        ))
        
        policy.load_state_dict(state)
    else:
        mod_path = Path(inspect.getfile(get_wrappers)).parent
        model = SAC.load(mod_path / 'model')
        policy = model.policy

    actor = SB3PolicyActor(policy, deterministic=False)
    argmax_actor = SB3PolicyActor(policy, deterministic=True)
    return Agents(actor, argmax_actor)


def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return [
        # lambda env: DriftRewardWrapper(env, drift_bonus=0.05),
        lambda env: DictObsToBoxWrapper(env),
        lambda env: FixDictActionWrapper(env),
    ]
