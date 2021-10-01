from typing import Union
import gin

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Independent

from basics.abstract_algorithms import RecurrentOffPolicyRLAlgorithm
from basics.actors_and_critics import MLPGaussianActor, MLPCritic
from basics.summarizer import Summarizer
from basics.replay_buffer_recurrent import RecurrentBatch
from basics.utils import get_device, create_target, mean_of_unmasked_elements, polyak_update, save_net, load_net


@gin.configurable(module=__name__)
class RecurrentSAC(RecurrentOffPolicyRLAlgorithm):

    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_dim=256,
        gamma=0.99,
        lr=3e-4,
        polyak=0.995,
        alpha=1.0,  # if autotune_alpha, this becomes the initial alpha value
        autotune_alpha: bool = True,
    ):

        # hyperparameters

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lr = lr
        self.polyak = polyak

        self.autotune_alpha = autotune_alpha

        if autotune_alpha:
            self.log_alpha = torch.log(torch.ones(1) * alpha).to(get_device()).requires_grad_(True)
            self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

        self.target_entropy = - self.action_dim  # int, but it will get broadcasted over a FloatTensor as a float

        # trackers

        self.hidden = None

        # networks

        self.actor_summarizer = Summarizer(input_dim, hidden_dim).to(get_device())

        self.Q1_summarizer = Summarizer(input_dim, hidden_dim).to(get_device())
        self.Q1_summarizer_targ = create_target(self.Q1_summarizer)

        self.Q2_summarizer = Summarizer(input_dim, hidden_dim).to(get_device())
        self.Q2_summarizer_targ = create_target(self.Q2_summarizer)

        self.actor = MLPGaussianActor(input_dim=hidden_dim, action_dim=action_dim).to(get_device())

        self.Q1 = MLPCritic(input_dim=hidden_dim, action_dim=action_dim).to(get_device())
        self.Q1_targ = create_target(self.Q1)

        self.Q2 = MLPCritic(input_dim=hidden_dim, action_dim=action_dim).to(get_device())
        self.Q2_targ = create_target(self.Q2)

        # optimizers

        self.actor_summarizer_optimizer = optim.Adam(self.actor_summarizer.parameters(), lr=lr)
        self.Q1_summarizer_optimizer = optim.Adam(self.Q1_summarizer.parameters(), lr=lr)
        self.Q2_summarizer_optimizer = optim.Adam(self.Q2_summarizer.parameters(), lr=lr)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.Q1_optimizer = optim.Adam(self.Q1.parameters(), lr=lr)
        self.Q2_optimizer = optim.Adam(self.Q2.parameters(), lr=lr)

    def reinitialize_hidden(self) -> None:
        self.hidden = None

    def sample_action_from_distribution(
            self,
            summary: torch.tensor,
            deterministic: bool,
            return_log_prob: bool,
    ) -> Union[torch.tensor, tuple]:  # tuple of 2 tensors if return_log_prob is True; else torch.tensor

        bs, seq_len = summary.shape[0], summary.shape[1]  # seq_len can be 1 (inference) or num_bptt (training)

        means, stds = self.actor(summary)

        means, stds = means.view(bs * seq_len, self.action_dim), stds.view(bs * seq_len, self.action_dim)

        if deterministic:
            u = means
        else:
            mu_given_s = Independent(Normal(loc=means, scale=stds), reinterpreted_batch_ndims=1)  # normal distribution
            u = mu_given_s.rsample()

        a = torch.tanh(u).view(bs, seq_len, self.action_dim)  # shape checking

        if return_log_prob:
            log_pi_a_given_s = mu_given_s.log_prob(u) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(dim=1)
            return a, log_pi_a_given_s.view(bs, seq_len, 1)  # add another dim to match Q values
        else:
            return a

    def act(self, observation: np.array, deterministic: bool) -> np.array:
        with torch.no_grad():
            observation = torch.tensor(observation).unsqueeze(0).unsqueeze(0).float().to(get_device())
            summary, self.hidden = self.actor_summarizer(observation, self.hidden, return_hidden=True)
            action = self.sample_action_from_distribution(summary, deterministic=deterministic, return_log_prob=False)
            return action.view(-1).cpu().numpy()  # view as 1d -> to cpu -> to numpy

    def get_current_alpha(self):
        if self.autotune_alpha:
            return np.exp(float(self.log_alpha))
        else:
            return self.alpha

    def update_networks(self, b: RecurrentBatch) -> dict:

        bs, num_bptt = b.r.shape[0], b.r.shape[1]

        # compute summary

        actor_summary = self.actor_summarizer(b.o)
        Q1_summary = self.Q1_summarizer(b.o)
        Q2_summary = self.Q2_summarizer(b.o)

        Q1_summary_targ = self.Q1_summarizer_targ(b.o)
        Q2_summary_targ = self.Q2_summarizer_targ(b.o)

        actor_summary_1_T, actor_summary_2_Tplus1 = actor_summary[:, :-1, :], actor_summary[:, 1:, :]
        Q1_summary_1_T, Q1_summary_2_Tplus1 = Q1_summary[:, :-1, :], Q1_summary_targ[:, 1:, :]
        Q2_summary_1_T, Q2_summary_2_Tplus1 = Q2_summary[:, :-1, :], Q2_summary_targ[:, 1:, :]

        assert actor_summary.shape == (bs, num_bptt+1, self.hidden_dim)

        # compute predictions

        Q1_predictions = self.Q1(Q1_summary_1_T, b.a)
        Q2_predictions = self.Q2(Q2_summary_1_T, b.a)

        assert Q1_predictions.shape == (bs, num_bptt, 1)
        assert Q2_predictions.shape == (bs, num_bptt, 1)

        # compute targets

        with torch.no_grad():

            na, log_pi_na_given_ns = self.sample_action_from_distribution(actor_summary_2_Tplus1, deterministic=False,
                                                                          return_log_prob=True)

            n_min_Q_targ = torch.min(self.Q1_targ(Q1_summary_2_Tplus1, na),
                                     self.Q2_targ(Q2_summary_2_Tplus1, na))
            n_sample_entropy = - log_pi_na_given_ns

            targets = b.r + self.gamma * (1 - b.d) * (n_min_Q_targ + self.get_current_alpha() * n_sample_entropy)

            assert na.shape == (bs, num_bptt, self.action_dim)
            assert log_pi_na_given_ns.shape == (bs, num_bptt, 1)
            assert n_min_Q_targ.shape == (bs, num_bptt, 1)
            assert targets.shape == (bs, num_bptt, 1)

        # compute td error

        Q1_loss_elementwise = (Q1_predictions - targets) ** 2
        Q1_loss = mean_of_unmasked_elements(Q1_loss_elementwise, b.m)

        Q2_loss_elementwise = (Q2_predictions - targets) ** 2
        Q2_loss = mean_of_unmasked_elements(Q2_loss_elementwise, b.m)

        assert Q1_loss.shape == ()
        assert Q2_loss.shape == ()

        # reduce td error

        self.Q1_summarizer_optimizer.zero_grad()
        self.Q1_optimizer.zero_grad()
        Q1_loss.backward()
        self.Q1_summarizer_optimizer.step()
        self.Q1_optimizer.step()

        self.Q2_summarizer_optimizer.zero_grad()
        self.Q2_optimizer.zero_grad()
        Q2_loss.backward()
        self.Q2_summarizer_optimizer.step()
        self.Q2_optimizer.step()

        # compute policy loss

        a, log_pi_a_given_s = self.sample_action_from_distribution(actor_summary_1_T, deterministic=False,
                                                                   return_log_prob=True)

        min_Q = torch.min(self.Q1(Q1_summary_1_T.detach(), a),
                          self.Q2(Q2_summary_1_T.detach(), a))
        sample_entropy = - log_pi_a_given_s

        policy_loss_elementwise = - (min_Q + self.get_current_alpha() * sample_entropy)
        policy_loss = mean_of_unmasked_elements(policy_loss_elementwise, b.m)

        assert a.shape == (bs, num_bptt, self.action_dim)
        assert log_pi_a_given_s.shape == (bs, num_bptt, 1)
        assert min_Q.shape == (bs, num_bptt, 1)
        assert policy_loss.shape == ()

        # reduce policy loss

        self.actor_summarizer_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_summarizer_optimizer.step()
        self.actor_optimizer.step()

        if self.autotune_alpha:

            # compute log alpha loss

            excess_entropy = sample_entropy.detach() - self.target_entropy
            log_alpha_loss = self.log_alpha * torch.mean(excess_entropy)

            assert log_alpha_loss.shape == (1,)

            # reduce log alpha loss

            self.log_alpha_optimizer.zero_grad()
            log_alpha_loss.backward()
            self.log_alpha_optimizer.step()

        else:

            log_alpha_loss = 0  # needed only for logging purposes

        # update target networks

        polyak_update(targ_net=self.Q1_summarizer_targ, pred_net=self.Q1_summarizer, polyak=self.polyak)
        polyak_update(targ_net=self.Q2_summarizer_targ, pred_net=self.Q2_summarizer, polyak=self.polyak)

        polyak_update(targ_net=self.Q1_targ, pred_net=self.Q1, polyak=self.polyak)
        polyak_update(targ_net=self.Q2_targ, pred_net=self.Q2, polyak=self.polyak)

        return {
            # for learning the q functions
            '(qfunc) Q1 pred': float(mean_of_unmasked_elements(Q1_predictions, b.m)),
            '(qfunc) Q2 pred': float(mean_of_unmasked_elements(Q2_predictions, b.m)),
            '(qfunc) Q1 Q2 pairwise abs diff': float(
                mean_of_unmasked_elements(torch.abs(Q1_predictions - Q2_predictions), b.m)),
            '(qfunc) Q1 loss': float(Q1_loss),
            '(qfunc) Q2 loss': float(Q2_loss),
            # for learning the actor
            '(actor) min Q value': float(mean_of_unmasked_elements(min_Q, b.m)),
            '(actor) entropy (sample)': float(mean_of_unmasked_elements(sample_entropy, b.m)),
            # for learning the entropy coefficient (alpha)
            '(alpha) alpha': self.get_current_alpha(),
            '(alpha) log alpha loss': float(log_alpha_loss)
        }

    def save_actor(self, save_dir: str) -> None:
        save_net(net=self.actor_summarizer, save_dir=save_dir, save_name="actor_summarizer.pth")
        save_net(net=self.actor, save_dir=save_dir, save_name="actor.pth")

    def load_actor(self, save_dir: str) -> None:
        load_net(net=self.actor_summarizer, save_dir=save_dir, save_name="actor_summarizer.pth")
        load_net(net=self.actor, save_dir=save_dir, save_name="actor.pth")

    def copy_networks_from(self, algorithm) -> None:

        self.actor_summarizer.load_state_dict(algorithm.actor_summarizer.state_dict())

        self.Q1_summarizer.load_state_dict(algorithm.Q1_summarizer.state_dict())
        self.Q1_summarizer_targ.load_state_dict(algorithm.Q1_summarizer_targ.state_dict())

        self.Q2_summarizer.load_state_dict(algorithm.Q2_summarizer.state_dict())
        self.Q2_summarizer_targ.load_state_dict(algorithm.Q2_summarizer_targ.state_dict())

        self.actor.load_state_dict(algorithm.actor.state_dict())

        self.Q1.load_state_dict(algorithm.Q1.state_dict())
        self.Q1_targ.load_state_dict(algorithm.Q1_targ.state_dict())
        self.Q2.load_state_dict(algorithm.Q2.state_dict())
        self.Q2_targ.load_state_dict(algorithm.Q2_targ.state_dict())
