from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym
from stable_baselines3 import A2C
import inspect
from pathlib import Path

# Imports our Actor class
# IMPORTANT: note the relative import
from .actors import Actor, SB3PolicyActor, ArgmaxActor, SamplingActor, DictObsToBoxWrapper, MultiDiscreteToBoxWrapper

#: The base environment name
env_name = "supertuxkart/flattened_multidiscrete-v0"

#: Player name
player_name = "Example"


def get_actor(
    state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
) -> Agent:
    if len(state) == 0:
        return SamplingActor(action_space)


    mod_path = Path(inspect.getfile(get_wrappers)).parent

    model = A2C.load(mod_path / "model.zip")

    actor = SB3PolicyActor(model.policy, deterministic=False)
    return Agents(actor, ArgmaxActor())


def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return [
        lambda env: MultiDiscreteToBoxWrapper(env),
        lambda env: DictObsToBoxWrapper(env),
    ]
