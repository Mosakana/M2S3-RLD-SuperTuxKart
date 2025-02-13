from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym

import inspect
from pathlib import Path

# Imports our Actor class
# IMPORTANT: note the relative import
from .actors import SB3PolicyActor, DictObsToBoxWrapper, FixDictActionWrapper, DriftRewardWrapper
from stable_baselines3 import SAC
from stable_baselines3.sac import MultiInputPolicy

#: The base environment name
env_name = "supertuxkart/flattened-v0"

#: Player name
player_name = "Winux"


def get_actor(
    state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
) -> Agent:
    # if state is None:
    #     return SamplingActor(action_space)

    mod_path = Path(inspect.getfile(get_wrappers)).parent

    model = SAC.load(mod_path / 'model')

    actor = SB3PolicyActor(model.policy, deterministic=False)
    # actor.sb3_policy.load_state_dict(state)
    argmax_actor = SB3PolicyActor(model.policy, deterministic=True)
    # argmax_actor.sb3_policy.load_state_dict(state)
    return Agents(actor, argmax_actor)


def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return [
        # lambda env: DriftRewardWrapper(env, drift_bonus=0.05),
        lambda env: DictObsToBoxWrapper(env),
        lambda env: FixDictActionWrapper(env),
    ]
