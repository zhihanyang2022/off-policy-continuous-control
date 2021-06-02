import os
import gin
import numpy as np

from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim

from basics.abstract_algorithm import RecurrentOffPolicyRLAlgorithm
from basics.actors_and_critics import MLPTanhActor, MLPCritic, set_requires_grad_flag
from basics.replay_buffer_recurrent import RecurrentBatch
from basics.cuda_utils import get_device

"""
Design choices (as close as possible to a concatenation approach):
- For critic, actions are concatenated to the output of previous recurrent layers (is this really required? but should
help with reducing the complexity required by the model; so let's do that; two small 64 layers for action processing)
- Compute hidden states using T+1 length string or not (hyperparameter)
- there has to be a linear layer after an lstm, because the default inner activation of lstm is sigmoid

basically, if concat is solved with mlp of 2 layers, then we want to have 2 layers of lstm to simulate that kind of
information availability, and add actions after that

past actions are included via env, this helps lstm critic design to be easier; otherwise need to feed in action

so for pendulum var len, we can try a setting with 2 lstm layers at first and 2 linear layers up next
"""


@gin.configurable(module=__name__)
class DDPG_LSTM(RecurrentOffPolicyRLAlgorithm):

    """Deep deterministic policy gradient"""

    def __init__(
            self,
            input_dim,
            action_dim,
            hidden_size=gin.REQUIRED,
            num_lstm_layers=gin.REQUIRED,  # should ideally be configured somewhere else; TODO(zhihan)
            use_target_for_lstm=gin.REQUIRED,
            action_noise=gin.REQUIRED,
            gamma=gin.REQUIRED,
            lr=gin.REQUIRED,
            polyak=gin.REQUIRED,
    ):

        # networks

        self.actor_lstm = nn.LSTM(input_dim, hidden_size, batch_first=True, num_layers=num_lstm_layers).to(
            get_device())
        self.critic_lstm = nn.LSTM(input_dim, hidden_size, batch_first=True, num_layers=num_lstm_layers).to(
            get_device())

        self.use_target_for_lstm = use_target_for_lstm

        if use_target_for_lstm:

            self.actor_lstm_targ = deepcopy(self.actor_lstm)
            set_requires_grad_flag(self.actor_lstm_targ, False)

            self.critic_lstm_targ = deepcopy(self.critic_lstm)
            set_requires_grad_flag(self.critic_lstm_targ, False)

        self.actor = MLPTanhActor(hidden_size, action_dim).to(get_device())
        self.actor_targ = deepcopy(self.actor)
        set_requires_grad_flag(self.actor_targ, False)

        self.Q = MLPCritic(hidden_size, action_dim).to(get_device())
        self.Q_targ = deepcopy(self.Q)
        set_requires_grad_flag(self.Q_targ, False)

        # optimizers

        self.actor_lstm_optimizer = optim.Adam(self.actor_lstm.parameters(), lr=lr)
        self.critic_lstm_optimizer = optim.Adam(self.critic_lstm.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        # hyper-parameters

        self.gamma = gamma
        self.action_noise = action_noise
        self.polyak = polyak

        # miscellaneous

        self.input_dim = input_dim
        self.action_dim = action_dim

        self.h_and_c = None

    def reinitialize_hidden(self) -> None:
        self.h_and_c = None

    def act(self, observation: np.array, deterministic: bool) -> np.array:
        with torch.no_grad():
            observation = torch.tensor(observation).unsqueeze(0).unsqueeze(0).float().to(get_device())
            self.actor_lstm.flatten_parameters()
            h, self.h_and_c = self.actor_lstm(observation, self.h_and_c)
            greedy_action = self.actor(h).view(-1).cpu().numpy()  # view as 1d -> to cpu -> to numpy
            if not deterministic:
                return np.clip(greedy_action + self.action_noise * np.random.randn(len(greedy_action)), -1.0, 1.0)
            else:
                return greedy_action

    @staticmethod
    def rescale_loss(loss: torch.tensor, mask: torch.tensor) -> torch.tensor:
        return loss / mask.sum() * np.prod(mask.shape)

    @staticmethod
    def feed_lstm(lstm, o):
        """Nothing special; just making code more readbale in update_networks"""
        lstm.flatten_parameters()  # prevent some arbitrary error that I don't understand
        h, h_and_c = lstm(o)
        return h

    def polyak_update(self, targ_net, pred_net) -> None:
        for old_param, new_param in zip(targ_net.parameters(), pred_net.parameters()):
            old_param.data.copy_(old_param.data * self.polyak + new_param.data * (1 - self.polyak))

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

        set_requires_grad_flag(self.Q, False)

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

        set_requires_grad_flag(self.Q, True)

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

    def save_networks(self, save_dir: str) -> None:

        torch.save(self.actor_lstm.state_dict(), os.path.join(save_dir, 'actor_lstm.pth'))
        torch.save(self.critic_lstm.state_dict(), os.path.join(save_dir, 'critic_lstm.pth'))

        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'actor.pth'))
        torch.save(self.Q.state_dict(), os.path.join(save_dir, 'Q.pth'))
        torch.save(self.Q_targ.state_dict(), os.path.join(save_dir, 'Q_targ.pth'))

    def load_actor(self, save_dir: str) -> None:

        self.actor_lstm.load_state_dict(
            torch.load(os.path.join(save_dir, 'actor_lstm.pth'), map_location=torch.device(get_device())))

        self.actor.load_state_dict(
            torch.load(os.path.join(save_dir, 'actor.pth'), map_location=torch.device(get_device())))

    def load_networks(self, save_dir: str) -> None:

        self.actor_lstm.load_state_dict(
            torch.load(os.path.join(save_dir, 'actor_lstm.pth'), map_location=torch.device(get_device())))
        self.critic_lstm.load_state_dict(
            torch.load(os.path.join(save_dir, 'critic_lstm.pth'), map_location=torch.device(get_device())))

        self.actor.load_state_dict(
            torch.load(os.path.join(save_dir, 'actor.pth'), map_location=torch.device(get_device())))
        self.Q1.load_state_dict(
            torch.load(os.path.join(save_dir, 'Q1.pth'), map_location=torch.device(get_device())))
        self.Q1_target.load_state_dict(
            torch.load(os.path.join(save_dir, 'Q1_targ.pth'), map_location=torch.device(get_device())))
