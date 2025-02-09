from pathlib import Path
from pystk2_gymnasium import AgentSpec
from functools import partial
import torch
import inspect
from bbrl.agents.gymnasium import ParallelGymAgent, make_env

# Note the use of relative imports
from .actors import SB3PolicyActor
from .pystk_actor import env_name, get_wrappers, player_name
from stable_baselines3 import A2C

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

    env_agent = ParallelGymAgent(make_stkenv, 1)
    env = env_agent.envs[0]

    # (2) Learn

    model = A2C("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    policy = model.policy

    # (3) Save the actor sate
    sb3_actor = SB3PolicyActor(model)
    mod_path = Path(inspect.getfile(get_wrappers)).parent
    torch.save(sb3_actor.state_dict(), mod_path / "pystk_actor.pth")
    model.save(mod_path / "model.zip")
