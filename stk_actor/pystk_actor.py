from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym
from stable_baselines3 import SAC, PPO
from sb3_contrib import TQC
import inspect
from pathlib import Path
from sb3_contrib.tqc.policies import MultiInputPolicy

# Imports our Actor class
# IMPORTANT: note the relative import
from .actors import Actor, SB3PolicyActor, ArgmaxActor, SamplingActor, DictObsToBoxWrapper, MultiDiscreteToBoxWrapper, FixedActionWrapper

#: The base environment name
env_name = "supertuxkart/flattened_multidiscrete-v0"

#: Player name
player_name = "Example"


def get_actor(
    state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
) -> Agent:
    if state is None:
        return SamplingActor(action_space)

    mod_path = Path(inspect.getfile(get_wrappers)).parent

    policy = MultiInputPolicy(observation_space, action_space, lr_schedule=lambda x: 0.0, net_arch=dict(
            pi=[256, 256],   
            qf=[256, 256],   
        ))
    
    policy.load_state_dict(state)

    actor = SB3PolicyActor(policy, deterministic=False)
    # actor.sb3_policy.load_state_dict(state)
    argmax_actor = SB3PolicyActor(policy, deterministic=True)
    # argmax_actor.sb3_policy.load_state_dict(state)
    return Agents(actor, argmax_actor)


def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return [
        lambda env: DictObsToBoxWrapper(env),
        lambda env: FixedActionWrapper(env),
        lambda env: MultiDiscreteToBoxWrapper(env)
    ]
