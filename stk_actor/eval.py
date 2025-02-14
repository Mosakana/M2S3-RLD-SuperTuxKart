import gymnasium as gym
from pystk2_gymnasium import AgentSpec
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import TQC
from functools import partial
import inspect
from bbrl.agents.gymnasium import ParallelGymAgent, make_env
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path
from .pystk_actor import env_name, get_wrappers, player_name
from .actors import DictObsToBoxWrapper, FixDictActionWrapper, DriftRewardWrapper



mod_path = Path(inspect.getfile(get_wrappers)).parent

TRACK = ['abyss', 'black_forest', 'candela_city',
             'cocoa_temple', 'cornfield_crossing', 'fortmagma',
             'gran_paradiso_island', 'hacienda', 'lighthouse',
             'mines', 'minigolf','olivermath', 'ravenbridge_mansion',
             'sandtrack', 'scotland', 'snowmountain', 'snowtuxpeak',
             'stk_enterprise', 'volcano_island', 'xr591', 'zengarden']

def make_env(track):
    def init():
        return FixDictActionWrapper(DictObsToBoxWrapper(DriftRewardWrapper(gym.make(
            env_name,
            track=track,
            render_mode=None,
            autoreset=True,
            agent=AgentSpec(use_ai=False, name=player_name),))))
    return init

env_fns = [make_env(track) for track in TRACK]

env = DummyVecEnv(env_fns)

model = TQC.load(mod_path / "model")


# 评估模型表现（运行10个episode，计算平均回报）
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)

print(mean_reward, std_reward)