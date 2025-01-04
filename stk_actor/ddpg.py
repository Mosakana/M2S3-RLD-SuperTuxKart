# import os
import copy
# import math
# import bbrl_gymnasium  # noqa: F401
import torch
import torch.nn as nn
from utils import build_mlp, setup_optimizer
from torch.distributions import Normal
# from omegaconf import OmegaConf
#
# class ContinuousQAgent(Agent):
#     def __init__(self, state_dim, hidden_layers, action_dim):
#         super().__init__()
#         self.is_q_function = True
#         self.model = build_mlp(
#             [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
#         )
#
#     def forward(self, t, **kwargs):
#         # Get the current state $s_t$ and the chosen action $a_t$
#         obs = self.get(("env/env_obs", t))  # shape B x D_{obs}
#         action = self.get(("action", t))  # shape B x D_{action}
#
#         # Compute the Q-value(s_t, a_t)
#         obs_act = torch.cat((obs, action), dim=1)  # shape B x (D_{obs} + D_{action})
#         # Get the q-value (and remove the last dimension since it is a scalar)
#         q_value = self.model(obs_act).squeeze(-1)
#         self.set((f"{self.prefix}q_value", t), q_value)
#
# class ContinuousDeterministicActor(Agent):
#     def __init__(self, state_dim, hidden_layers, action_dim):
#         super().__init__()
#         layers = [state_dim] + list(hidden_layers) + [action_dim]
#         self.model = build_mlp(
#             layers, activation=nn.ReLU(), output_activation=nn.Tanh()
#         )
#
#     def forward(self, t, **kwargs):
#         obs = self.get(("env/env_obs", t))
#         action = self.model(obs)
#         self.set(("action", t), action)
#
#
# class AddGaussianNoise(Agent):
#     def __init__(self, sigma):
#         super().__init__()
#         self.sigma = sigma
#
#     def forward(self, t, **kwargs):
#         act = self.get(("action", t))
#         dist = Normal(act, self.sigma)
#         action = dist.sample()
#         self.set(("action", t), action)
#
# class AddOUNoise(Agent):
#     """
#     Ornstein-Uhlenbeck process noise for actions as suggested by DDPG paper
#     """
#     def __init__(self, std_dev, theta=0.15, dt=1e-2):
#         super().__init__()
#         self.theta = theta
#         self.std_dev = std_dev
#         self.dt = dt
#         self.x_prev = 0
#
#     def forward(self, t, **kwargs):
#         act = self.get(("action", t))
#         x = (
#             self.x_prev
#             + self.theta * (act - self.x_prev) * self.dt
#             + self.std_dev * math.sqrt(self.dt) * torch.randn(act.shape)
#         )
#         self.x_prev = x
#         self.set(("action", t), x)
#
#
#
# def compute_critic_loss(cfg, reward: torch.Tensor, must_bootstrap: torch.Tensor, q_values: torch.Tensor,
#                         target_q_values: torch.Tensor):
#     """Compute the DDPG critic loss from a sample of transitions
#
#     :param cfg: The configuration
#     :param reward: The reward (shape 2xB)
#     :param must_bootstrap: Must bootstrap flag (shape 2xB)
#     :param q_values: The computed Q-values (shape 2xB)
#     :param target_q_values: The Q-values computed by the target critic (shape 2xB)
#     :return: the loss (a scalar)
#     """
#     # Compute temporal difference
#     # To be completed...
#     mse = nn.MSELoss()
#     target = reward[1] + cfg.algorithm.discount_factor * must_bootstrap[1] * target_q_values[1]
#
#     return mse(target, q_values[0])
#
# def compute_actor_loss(q_values):
#     """Returns the actor loss
#
#     :param q_values: The q-values (shape 2xB)
#     :return: A scalar (the loss)
#     """
#     # To be completed...
#     return -q_values[1].mean()
#
# class DDPG(EpochBasedAlgo):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#
#         # we create the critic and the actor, but also an exploration agent to
#         # add noise and a target critic. The version below does not use a target
#         # actor as it proved hard to tune, but such a target actor is used in
#         # the original paper.
#
#         obs_size, act_size = self.train_env.get_obs_and_actions_sizes()
#         self.critic = ContinuousQAgent(
#             obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
#         ).with_prefix("critic/")
#         self.target_critic = copy.deepcopy(self.critic).with_prefix("target-critic/")
#
#         self.actor = ContinuousDeterministicActor(
#             obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
#         )
#
#         # As an alternative, you can use `AddOUNoise`
#         noise_agent = AddGaussianNoise(cfg.algorithm.action_noise)
#
#         self.train_policy = Agents(self.actor, noise_agent)
#         self.eval_policy = self.actor
#
#         # Define agents over time
#         self.t_actor = TemporalAgent(self.actor)
#         self.t_critic = TemporalAgent(self.critic)
#         self.t_target_critic = TemporalAgent(self.target_critic)
#
#         # Configure the optimizer
#         self.actor_optimizer = setup_optimizer(cfg.actor_optimizer, self.actor)
#         self.critic_optimizer = setup_optimizer(cfg.critic_optimizer, self.critic)
#
# def run_ddpg(ddpg: DDPG):
#     for rb in ddpg.iter_replay_buffers():
#         rb_workspace = rb.get_shuffled(ddpg.cfg.algorithm.batch_size)
#
#         # Compute the critic loss
#
#         # Critic update
#         # Compute critic loss
#         ###############################################################################
#
#         ddpg.t_critic(rb_workspace, t=0, n_steps=2)
#         with torch.no_grad():
#             ddpg.t_actor(rb_workspace, t=0, n_steps=2)
#             ddpg.t_target_critic(rb_workspace, t=0, n_steps=2)
#         critic_q_value, terminated, reward, target_q_value = rb_workspace[
#             'critic/q_value', 'env/terminated', 'env/reward', 'target-critic/q_value']
#         critic_loss = compute_critic_loss(ddpg.cfg, reward, must_bootstrap=~terminated, q_values=critic_q_value,
#                                           target_q_values=target_q_value)
#         ###############################################################################
#
#         # Gradient step (critic)
#         ddpg.logger.add_log("critic_loss", critic_loss, ddpg.nb_steps)
#         ddpg.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         torch.nn.utils.clip_grad_norm_(
#             ddpg.critic.parameters(), ddpg.cfg.algorithm.max_grad_norm
#         )
#         ddpg.critic_optimizer.step()
#
#         # Compute the actor loss
#
#         ######################################################################
#         ddpg.t_actor(rb_workspace, t=0, n_steps=2)
#         ddpg.t_critic(rb_workspace, t=0, n_steps=2, choose_action=False)
#         q_value = rb_workspace['critic/q_value']
#         actor_loss = compute_actor_loss(q_value)
#         ######################################################################
#
#         # Gradient step (actor)
#         ddpg.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         torch.nn.utils.clip_grad_norm_(
#             ddpg.actor.parameters(), ddpg.cfg.algorithm.max_grad_norm
#         )
#         ddpg.actor_optimizer.step()
#
#         # Soft update of target q function
#         soft_update_params(
#             ddpg.critic, ddpg.target_critic, ddpg.cfg.algorithm.tau_target
#         )
#
#         # Evaluate the actor if needed
#         if ddpg.evaluate():
#             if ddpg.cfg.plot_agents:
#                 plot_policy(
#                     ddpg.actor,
#                     ddpg.eval_env,
#                     ddpg.best_reward,
#                     str(ddpg.base_dir / "plots"),
#                     ddpg.cfg.gym_env.env_name,
#                     stochastic=False,
#                 )
#
class ContinuousQAgent:
    def __init__(self, state_dim, hidden_layers, action_dim):
        self.model = build_mlp([state_dim + action_dim] + hidden_layers + [1], activation=nn.ReLU())

    def forward(self, experience):
        state = experience['state']
        action = experience['action']

        state_action = torch.cat([state, action], dim=1)
        q_value = self.model(state_action).squeeze(-1)
        return q_value

class ContinuousDeterministicActor:
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        self.model = build_mlp(
            layers, activation=nn.ReLU(), output_activation=nn.Tanh()
        )

        def forward(self, experience):
            state = experience['state']
            action = self.model(state)
            return state

class AddGaussianNoise:
    def __init__(self, sigma):
        self.sigma = sigma

    def forward(self, experience, ):
        act = experience['action']
        dist = Normal(act, self.sigma)
        noised_action = dist.sample()
        return noised_action

def compute_critic_loss(cfg, reward: torch.Tensor, must_bootstrap: torch.Tensor, q_values: torch.Tensor,
                        target_q_values: torch.Tensor):
    """Compute the DDPG critic loss from a sample of transitions

    :param cfg: The configuration
    :param reward: The reward (shape 2xB)
    :param must_bootstrap: Must bootstrap flag (shape 2xB)
    :param q_values: The computed Q-values (shape 2xB)
    :param target_q_values: The Q-values computed by the target critic (shape 2xB)
    :return: the loss (a scalar)
    """
    # Compute temporal difference
    # To be completed...
    mse = nn.MSELoss()
    target = reward[1] + cfg.algorithm.discount_factor * must_bootstrap[1] * target_q_values[1]

    return mse(target, q_values[0])

def compute_actor_loss(q_values):
    """Returns the actor loss

    :param q_values: The q-values (shape 2xB)
    :return: A scalar (the loss)
    """
    # To be completed...
    return -q_values[1].mean()

class DDPG:
    def __init__(self, cfg):

        # we create the critic and the actor, but also an exploration agent to
        # add noise and a target critic. The version below does not use a target
        # actor as it proved hard to tune, but such a target actor is used in
        # the original paper.

        obs_size, act_size = self.train_env.get_obs_and_actions_sizes()
        self.critic = ContinuousQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
        )
        self.target_critic = copy.deepcopy(self.critic)

        self.actor = ContinuousDeterministicActor(
            obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
        )

        # As an alternative, you can use `AddOUNoise`
        noise_agent = AddGaussianNoise(cfg.algorithm.action_noise)

        self.train_policy = self.actor, noise_agent
        self.eval_policy = self.actor

        # Configure the optimizer
        self.actor_optimizer = setup_optimizer(cfg.actor_optimizer, self.actor)
        self.critic_optimizer = setup_optimizer(cfg.critic_optimizer, self.critic)

def run_ddpg(ddpg: DDPG):
    for rb in ddpg.iter_replay_buffers():
        rb_workspace = rb.get_shuffled(ddpg.cfg.algorithm.batch_size)

        # Compute the critic loss

        # Critic update
        # Compute critic loss
        ###############################################################################

        ddpg.t_critic(rb_workspace, t=0, n_steps=2)
        with torch.no_grad():
            ddpg.t_actor(rb_workspace, t=0, n_steps=2)
            ddpg.t_target_critic(rb_workspace, t=0, n_steps=2)
        critic_q_value, terminated, reward, target_q_value = rb_workspace[
            'critic/q_value', 'env/terminated', 'env/reward', 'target-critic/q_value']
        critic_loss = compute_critic_loss(ddpg.cfg, reward, must_bootstrap=~terminated, q_values=critic_q_value,
                                          target_q_values=target_q_value)
        ###############################################################################

        # Gradient step (critic)
        ddpg.logger.add_log("critic_loss", critic_loss, ddpg.nb_steps)
        ddpg.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            ddpg.critic.parameters(), ddpg.cfg.algorithm.max_grad_norm
        )
        ddpg.critic_optimizer.step()

        # Compute the actor loss

        ######################################################################
        ddpg.t_actor(rb_workspace, t=0, n_steps=2)
        ddpg.t_critic(rb_workspace, t=0, n_steps=2, choose_action=False)
        q_value = rb_workspace['critic/q_value']
        actor_loss = compute_actor_loss(q_value)
        ######################################################################

        # Gradient step (actor)
        ddpg.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            ddpg.actor.parameters(), ddpg.cfg.algorithm.max_grad_norm
        )
        ddpg.actor_optimizer.step()

        # Soft update of target q function
        soft_update_params(
            ddpg.critic, ddpg.target_critic, ddpg.cfg.algorithm.tau_target
        )

        # Evaluate the actor if needed
        if ddpg.evaluate():
            if ddpg.cfg.plot_agents:
                plot_policy(
                    ddpg.actor,
                    ddpg.eval_env,
                    ddpg.best_reward,
                    str(ddpg.base_dir / "plots"),
                    ddpg.cfg.gym_env.env_name,
                    stochastic=False,
                )