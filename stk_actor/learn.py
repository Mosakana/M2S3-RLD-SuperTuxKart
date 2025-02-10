from pathlib import Path
from pystk2_gymnasium import AgentSpec
from functools import partial
import torch
import inspect
from bbrl.agents.gymnasium import ParallelGymAgent, make_env

# Note the use of relative imports
from .actors import SB3PolicyActor
from .pystk_actor import env_name, get_wrappers, player_name
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

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

    env = make_vec_env(make_stkenv, n_envs=4)
    # (2) Learn

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],   
            qf=[256, 256],   
        ),
    )

    model = SAC("MultiInputPolicy", env, device='cuda', verbose=1,
                 tensorboard_log='./tensorboard_logs/', learning_rate=3e-4, batch_size=256, buffer_size=10_000,
                 gamma=0.99, tau=0.005, ent_coef='auto', policy_kwargs=policy_kwargs
    )
    model.learn(total_timesteps=6_000_000)
    policy = model.policy

    # (3) Save the actor sate
    sb3_actor = SB3PolicyActor(model)
    mod_path = Path(inspect.getfile(get_wrappers)).parent
    torch.save(policy.state_dict(), mod_path / "pystk_actor.pth")
    model.save(mod_path / "model.zip")
