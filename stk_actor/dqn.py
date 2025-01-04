import os
import copy
import bbrl_gymnasium  # noqa: F401
import torch
import torch.nn as nn
from bbrl.agents import Agent, Agents, TemporalAgent
from bbrl_utils.algorithms import EpochBasedAlgo
from bbrl_utils.nn import build_mlp, setup_optimizer, copy_parameters
from bbrl_utils.notebook import setup_tensorboard
from omegaconf import OmegaConf

class DiscreteQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.model = build_mlp(
            [state_dim] + list(hidden_layers) + [action_dim], activation=nn.ReLU()
        )

    def forward(self, t, **kwargs):
        obs = self.get(("env/env_obs", t))
        q_values = self.model(obs)
        self.set((f"{self.prefix}q_values", t), q_values)

class ArgmaxActionSelector(Agent):
    """BBRL agent that selects the best action based on Q(s,a)"""

    def forward(self, t: int, **kwargs):
        q_values = self.get(("q_values", t))
        action = q_values.argmax(-1)
        self.set(("action", t), action)


class EGreedyActionSelector(Agent):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, t: int, **kwargs):
        # Retrieves the q values
        # (matrix nb. of episodes x nb. of actions)
        q_values: torch.Tensor = self.get(("q_values", t))
        size, nb_actions = q_values.shape

        # Flag
        is_random = torch.rand(size) > self.epsilon

        # Actions (random / argmax)
        random_action = torch.randint(nb_actions, size=(size,))
        max_action = q_values.argmax(-1)

        # Choose the action based on the is_random flag
        action = torch.where(is_random, random_action, max_action)

        # Sets the action at time t
        self.set(("action", t), action)

class DQN(EpochBasedAlgo):
    def __init__(self, cfg):
        super().__init__(cfg)

        obs_size, act_size = self.train_env.get_obs_and_actions_sizes()

        # Get the two agents (critic and target critic)
        critic = DiscreteQAgent(
            obs_size, cfg.algorithm.architecture.hidden_size, act_size
        )
        target_critic = copy.deepcopy(critic).with_prefix("target/")

        # Builds the train agent that will produce transitions
        explorer = EGreedyActionSelector(cfg.algorithm.epsilon)
        self.train_policy = Agents(critic, explorer)

        self.eval_policy = Agents(critic, ArgmaxActionSelector())

        # Creates two temporal agents just for "replaying" some parts
        # of the transition buffer
        self.t_q_agent = TemporalAgent(critic)
        self.t_target_q_agent = TemporalAgent(target_critic)

        # Get an agent that is executed on a complete workspace
        self.optimizer = setup_optimizer(cfg.optimizer, self.t_q_agent)

        self.last_critic_update_step = 0

def dqn_compute_critic_loss(
    cfg, reward, must_bootstrap, q_values, target_q_values, action
):
    """Compute the critic loss

    :param reward: The reward $r_t$ (shape 2 x B)
    :param must_bootstrap: The must bootstrap flag at $t+1$ (shape 2 x B)
    :param q_values: The Q-values (shape 2 x B x A)
    :param target_q_values: The target Q-values (shape 2 x B x A)
    :param action: The chosen actions (shape 2 x B)
    :return: _description_
    """

    # Implement the DQN loss

    # Adapt from the previous notebook and adapt to our case (target Q network)
    # Don't forget that we deal with transitions (and not episodes)

    # Compute the target
    max_q = target_q_values[1].max(-1).values.detach()
    target = reward[1] + cfg.algorithm.discount_factor * max_q * must_bootstrap[1].int()

    # Compute the Q-values for the chosen actions
    act = action[0].unsqueeze(-1)
    qvals = torch.gather(q_values[0], dim=-1, index=act).squeeze(-1)


    # Compute critic loss (no need to use must_bootstrap here since we are dealing with "full" transitions)
    mse = nn.MSELoss()
    critic_loss = mse(target, qvals)


    return critic_loss

def run(dqn: DQN, compute_critic_loss):
    for rb in dqn.iter_replay_buffers():
        for _ in range(dqn.cfg.algorithm.n_updates):
            rb_workspace = rb.get_shuffled(dqn.cfg.algorithm.batch_size)

            # The q agent needs to be executed on the rb_workspace workspace
            dqn.t_q_agent(rb_workspace, t=0, n_steps=2, choose_action=False)
            with torch.no_grad():
                dqn.t_target_q_agent(rb_workspace, t=0, n_steps=2, stochastic=True)

            q_values, terminated, reward, action, target_q_values = rb_workspace[
                "q_values", "env/terminated", "env/reward", "action", "target/q_values"
            ]

            # Determines whether values of the critic should be propagated
            must_bootstrap = ~terminated

            # Compute critic loss
            critic_loss = compute_critic_loss(
                dqn.cfg, reward, must_bootstrap, q_values, target_q_values, action
            )
            # Store the loss for tensorboard display
            dqn.logger.add_log("critic_loss", critic_loss, dqn.nb_steps)
            dqn.logger.add_log("q_values/min", q_values.max(-1).values.min(), dqn.nb_steps)
            dqn.logger.add_log("q_values/max", q_values.max(-1).values.max(), dqn.nb_steps)
            dqn.logger.add_log("q_values/mean", q_values.max(-1).values.mean(), dqn.nb_steps)

            dqn.optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                dqn.t_q_agent.parameters(), dqn.cfg.algorithm.max_grad_norm
            )
            dqn.optimizer.step()

            # Update target
            if (
                dqn.nb_steps - dqn.last_critic_update_step
                > dqn.cfg.algorithm.target_critic_update
            ):
                dqn.last_critic_update_step = dqn.nb_steps
                copy_parameters(dqn.t_q_agent, dqn.t_target_q_agent)

            dqn.evaluate()