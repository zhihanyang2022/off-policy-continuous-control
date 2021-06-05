import gin

from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from basics.abstract_algorithm import RecurrentOffPolicyRLAlgorithm
from basics.actors_and_critics import MLPTanhActor, MLPCritic, set_requires_grad_flag
from basics.replay_buffer_recurrent import RecurrentBatch
from basics.cuda_utils import get_device


@gin.configurable(module=__name__)
class DDPG_LSTM(RecurrentOffPolicyRLAlgorithm):

    """Deep deterministic policy gradient with recurrent networks"""

    def __init__(
            self,
            input_dim,
            action_dim,
            hidden_size=gin.REQUIRED,
            num_lstm_layers=gin.REQUIRED,  # should ideally be configured somewhere else; TODO(zhihan)
            use_target_for_lstm=gin.REQUIRED,
            action_noise_type=gin.REQUIRED,
            action_noise=gin.REQUIRED,
            gamma=gin.REQUIRED,
            lr=gin.REQUIRED,
            polyak=gin.REQUIRED,
    ):

        # hyperparameters

        super().__init__(
            input_dim=input_dim,
            action_dim=action_dim,
            gamma=gamma,
            lr=lr,
            polyak=polyak
        )

        if action_noise_type == "ou":
            self.noise_callable = OrnsteinUhlenbeckActionNoise(mean=np.array([0]), sigma=np.array([1]))
        elif action_noise_type == "uniform":
            self.noise_callable = lambda: np.random.randn(len(action_dim))
        else:
            raise NotImplementedError(f"Noise type {action_noise_type} is not implemented.")

        self.action_noise = action_noise

        # networks

        self.actor_lstm = nn.LSTM(input_dim, hidden_size, batch_first=True, num_layers=num_lstm_layers).to(
            get_device())
        self.critic_lstm = nn.LSTM(input_dim, hidden_size, batch_first=True, num_layers=num_lstm_layers).to(
            get_device())

        self.networks_to_save_dict.update({'actor_lstm': self.actor_lstm})

        self.use_target_for_lstm = use_target_for_lstm

        if use_target_for_lstm:

            self.actor_lstm_targ = deepcopy(self.actor_lstm)
            set_requires_grad_flag(self.actor_lstm_targ, False)

            self.critic_lstm_targ = deepcopy(self.critic_lstm)
            set_requires_grad_flag(self.critic_lstm_targ, False)

        self.actor = MLPTanhActor(hidden_size, action_dim).to(get_device())
        self.actor_targ = deepcopy(self.actor)
        set_requires_grad_flag(self.actor_targ, False)

        self.networks_to_save_dict.update({'actor': self.actor})

        self.Q = MLPCritic(hidden_size, action_dim).to(get_device())
        self.Q_targ = deepcopy(self.Q)
        set_requires_grad_flag(self.Q_targ, False)

        # optimizers

        self.actor_lstm_optimizer = optim.Adam(self.actor_lstm.parameters(), lr=lr)
        self.critic_lstm_optimizer = optim.Adam(self.critic_lstm.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=lr)

    def maybe_reset_noise(self):
        """Noise types of OU-noise need to be reset upon episode termination."""
        if isinstance(self.noise_gen, OrnsteinUhlenbeckActionNoise):
            self.noise_gen.reset()

    def act(self, observation: np.array, deterministic: bool) -> np.array:
        with torch.no_grad():
            observation = torch.tensor(observation).unsqueeze(0).unsqueeze(0).float().to(get_device())
            self.actor_lstm.flatten_parameters()
            h, self.h_and_c = self.actor_lstm(observation, self.h_and_c)
            greedy_action = self.actor(h).view(-1).cpu().numpy()  # view as 1d -> to cpu -> to numpy
            if not deterministic:
                return np.clip(greedy_action + self.action_noise * self.noise_callable(), -1.0, 1.0)
            else:
                return greedy_action

    def update_networks(self, b: RecurrentBatch):

        bs, num_bptt = b.r.shape[0], b.r.shape[1]

        # compute hidden

        actor_h = self.feed_lstm(self.actor_lstm, b.o)
        critic_h = self.feed_lstm(self.critic_lstm, b.o)

        if self.use_target_for_lstm:

            actor_h_targ = self.feed_lstm(self.actor_lstm_targ, b.o)
            critic_h_targ = self.feed_lstm(self.critic_lstm_targ, b.o)

            actor_h_1_T, actor_h_2_Tplus1 = actor_h[:, :-1, :], actor_h_targ[:, 1:, :]
            critic_h_1_T, critic_h_2_Tplus1 = critic_h[:, :-1, :], critic_h_targ[:, 1:, :]

        else:

            actor_h_1_T, actor_h_2_Tplus1 = actor_h[:, :-1, :], actor_h[:, 1:, :]
            critic_h_1_T, critic_h_2_Tplus1 = critic_h[:, :-1, :], critic_h[:, 1:, :]

        # compute predictions

        predictions = self.Q(critic_h_1_T, b.a)

        assert predictions.shape == (bs, num_bptt, 1)

        # compute targets

        with torch.no_grad():

            na = self.actor_targ(actor_h_2_Tplus1)
            targets = b.r + self.gamma * (1 - b.d) * self.Q_targ(critic_h_2_Tplus1, na)

            assert na.shape == (bs, num_bptt, self.action_dim)
            assert targets.shape == (bs, num_bptt, 1)

        # compute td error

        Q_loss_elementwise = (predictions - targets) ** 2
        Q_loss = self.rescale_loss(torch.mean(b.m * Q_loss_elementwise), b.m)

        assert Q_loss.shape == ()

        # reduce td error

        self.critic_lstm_optimizer.zero_grad()
        self.Q_optimizer.zero_grad()

        Q_loss.backward()

        self.critic_lstm_optimizer.step()
        self.Q_optimizer.step()

        # compute policy loss

        a = self.actor(actor_h_1_T)
        Q_values = self.Q(critic_h_1_T.detach(), a)
        policy_loss_elementwise = - Q_values
        policy_loss = self.rescale_loss(torch.mean(b.m * policy_loss_elementwise), b.m)

        assert a.shape == (bs, num_bptt, self.action_dim)
        assert policy_loss.shape == ()

        # reduce policy loss

        self.actor_lstm_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        policy_loss.backward()

        self.actor_lstm_optimizer.step()
        self.actor_optimizer.step()

        # update target networks

        self.polyak_update(targ_net=self.actor_targ, pred_net=self.actor)
        self.polyak_update(targ_net=self.Q_targ, pred_net=self.Q)

        if self.use_target_for_lstm:

            self.polyak_update(targ_net=self.actor_lstm_targ, pred_net=self.actor_lstm)
            self.polyak_update(targ_net=self.critic_lstm_targ, pred_net=self.critic_lstm)

        return {
            # for learning the q functions
            '(qfunc) Q pred': float(predictions.mean()),
            '(qfunc) Q loss': float(Q_loss),
            # for learning the actor
            '(actor) policy loss': float(policy_loss),
        }
