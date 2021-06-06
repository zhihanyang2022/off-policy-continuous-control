import gin

import numpy as np
import torch
import torch.optim as optim

from basics.abstract_algorithms import RecurrentOffPolicyRLAlgorithm
from basics.summarizer import Summarizer
from basics.actors_and_critics import MLPTanhActor, MLPCritic
from basics.replay_buffer_recurrent import RecurrentBatch
from basics.utils import get_device, create_target, polyak_update, save_net, load_net


@gin.configurable(module=__name__)
class RecurrentTD3(RecurrentOffPolicyRLAlgorithm):

    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_dim=256,
        gamma=0.99,
        lr=3e-4,
        polyak=0.995,
        action_noise=0.1,  # standard deviation of action noise
        target_noise=0.2,  # standard deviation of target smoothing noise
        noise_clip=0.5,  # max abs value of target smoothing noise
        policy_delay=2
    ):

        # hyper-parameters

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lr = lr
        self.polyak = polyak

        self.action_noise = action_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip

        self.policy_delay = policy_delay

        # trackers

        self.hidden = None
        self.num_Q_updates = 0
        self.mean_Q1_val = 0

        # networks

        self.actor_summarizer = Summarizer(input_dim, hidden_dim).to(get_device())
        self.actor_summarizer_targ = create_target(self.actor_summarizer)

        self.Q1_summarizer = Summarizer(input_dim, hidden_dim).to(get_device())
        self.Q1_summarizer_targ = create_target(self.Q1_summarizer)

        self.Q2_summarizer = Summarizer(input_dim, hidden_dim).to(get_device())
        self.Q2_summarizer_targ = create_target(self.Q2_summarizer)

        self.actor = MLPTanhActor(hidden_dim, action_dim).to(get_device())
        self.actor_targ = create_target(self.actor)

        self.Q1 = MLPCritic(hidden_dim, action_dim).to(get_device())
        self.Q1_targ = create_target(self.Q1)

        self.Q2 = MLPCritic(hidden_dim, action_dim).to(get_device())
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

    def act(self, observation: np.array, deterministic: bool) -> np.array:
        with torch.no_grad():
            observation = torch.tensor(observation).unsqueeze(0).unsqueeze(0).float().to(get_device())
            summary, self.hidden = self.actor_summarizer(observation, self.hidden, return_hidden=True)
            greedy_action = self.actor(summary).view(-1).cpu().numpy()  # view as 1d -> to cpu -> to numpy
            if deterministic:
                return greedy_action
            else:
                return np.clip(greedy_action + self.action_noise * np.random.randn(self.action_dim), -1.0, 1.0)

    def update_networks(self, b: RecurrentBatch):

        bs, num_bptt = b.r.shape[0], b.r.shape[1]

        # compute summary

        actor_summary = self.actor_summarizer(b.o)
        Q1_summary = self.Q1_summarizer(b.o)
        Q2_summary = self.Q2_summarizer(b.o)

        actor_summary_targ = self.actor_summarizer_targ(b.o)
        Q1_summary_targ = self.Q1_summarizer_targ(b.o)
        Q2_summary_targ = self.Q2_summarizer_targ(b.o)

        actor_summary_1_T, actor_summary_2_Tplus1 = actor_summary[:, :-1, :], actor_summary_targ[:, 1:, :]
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

            na = self.actor_targ(actor_summary_2_Tplus1)
            noise = torch.clamp(
                torch.randn(na.size()) * self.target_noise, -self.noise_clip, self.noise_clip
            ).to(get_device())
            smoothed_na = torch.clamp(na + noise, -1, 1)

            n_min_Q_targ = torch.min(self.Q1_targ(Q1_summary_2_Tplus1, smoothed_na),
                                     self.Q2_targ(Q2_summary_2_Tplus1, smoothed_na))

            targets = b.r + self.gamma * (1 - b.d) * n_min_Q_targ

            assert na.shape == (bs, num_bptt, self.action_dim)
            assert n_min_Q_targ.shape == (bs, num_bptt, 1)
            assert targets.shape == (bs, num_bptt, 1)

        # compute td error

        Q1_loss = torch.mean((Q1_predictions - targets) ** 2)
        Q2_loss = torch.mean((Q2_predictions - targets) ** 2)

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

        self.num_Q_updates += 1

        if self.num_Q_updates % self.policy_delay == 0:  # delayed policy update; special in TD3

            # compute policy loss

            a = self.actor(actor_summary_1_T)
            Q1_val = self.Q1(Q1_summary_1_T.detach(), a)  # val stands for values
            policy_loss = - torch.mean(Q1_val)

            self.mean_Q1_val = float(Q1_val.mean())
            assert a.shape == (bs, num_bptt, self.action_dim)
            assert Q1_val.shape == (bs, num_bptt, 1)
            assert policy_loss.shape == ()

            # reduce policy loss

            self.actor_summarizer_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_summarizer_optimizer.step()
            self.actor_optimizer.step()

            # update target networks

            polyak_update(targ_net=self.actor_summarizer_targ, pred_net=self.actor_summarizer, polyak=self.polyak)
            polyak_update(targ_net=self.Q1_summarizer_targ, pred_net=self.Q1_summarizer, polyak=self.polyak)
            polyak_update(targ_net=self.Q2_summarizer_targ, pred_net=self.Q2_summarizer, polyak=self.polyak)

            polyak_update(targ_net=self.actor_targ, pred_net=self.actor, polyak=self.polyak)
            polyak_update(targ_net=self.Q1_targ, pred_net=self.Q1, polyak=self.polyak)
            polyak_update(targ_net=self.Q2_targ, pred_net=self.Q2, polyak=self.polyak)

        return {
            # for learning the q functions
            '(qfunc) Q1 pred': float(Q1_predictions.mean()),
            '(qfunc) Q2 pred': float(Q2_predictions.mean()),
            '(qfunc) Q1 loss': float(Q1_loss),
            '(qfunc) Q2 loss': float(Q2_loss),
            # for learning the actor
            '(actor) Q1 val': self.mean_Q1_val  # no need to track policy loss; just its negation
        }

    def save_actor(self, save_dir: str) -> None:
        save_net(net=self.actor_summarizer, save_dir=save_dir, save_name="actor_summarizer.pth")
        save_net(net=self.actor, save_dir=save_dir, save_name="actor.pth")

    def load_actor(self, save_dir: str) -> None:
        load_net(net=self.actor_summarizer, save_dir=save_dir, save_name="actor_summarizer.pth")
        load_net(net=self.actor, save_dir=save_dir, save_name="actor.pth")
