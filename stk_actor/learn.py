import gymnasium as gym
import numpy as np
from pathlib import Path
from pystk2_gymnasium import AgentSpec
from functools import partial
import torch
import inspect
from bbrl.agents.gymnasium import ParallelGymAgent, make_env

# Note the use of relative imports
from .actors import SB3PolicyActor
from .pystk_actor import env_name, get_wrappers, player_name
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from sb3_contrib import TQC, TRPO, RecurrentPPO

if __name__ == "__main__":
    # Setup the environment
    make_stkenv = partial(
        make_env,
        env_name,
        wrappers=get_wrappers(),
        render_mode=None,
        autoreset=True,
        agent=AgentSpec(use_ai=False, name=player_name),
    )

    env = make_vec_env(make_stkenv, n_envs=16)

    n_actions = env.action_space.shape[-1]

    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    checkpoint_on_event = CheckpointCallback(save_freq=1, save_path="./logs/")
    event_callback = EveryNTimesteps(n_steps=1000, callback=checkpoint_on_event)


    # (2) Learn

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 128, 128],   
            qf=[256, 256, 128, 128],   
        ),
    )

    model = TQC("MultiInputPolicy", env, device='cuda', verbose=1,
                 tensorboard_log='./tensorboard_logs/', learning_rate=3e-4, batch_size=256,
                 gamma=0.99, ent_coef='auto',
                 tau=0.005,  buffer_size=10_000, policy_kwargs=policy_kwargs
    )
    model.learn(total_timesteps=1_000_000, callback=event_callback)
    policy = model.policy

    # (3) Save the actor sate
    sb3_actor = SB3PolicyActor(model)
    mod_path = Path(inspect.getfile(get_wrappers)).parent
    torch.save(policy.state_dict(), mod_path / "pystk_actor.pth")
    model.save(mod_path / "model.zip")
