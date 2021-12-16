import gin

import numpy as np
import torch
import torch.optim as optim

from basics.abstract_algorithms import TransformerOffPolicyRLAlgorithm
from basics.summarizer import TransformerSummarizer
from basics.actors_and_critics import MLPTanhActor, MLPCritic
from basics.replay_buffer_recurrent import RecurrentBatch
from basics.utils import get_device, create_target, mean_of_unmasked_elements, polyak_update, save_net, load_net


@gin.configurable(module=__name__)
class TransformerDDPG(TransformerOffPolicyRLAlgorithm):

    """Deep deterministic policy gradient with recurrent networks"""

    def __init__(
        self,
        input_dim,
        action_dim,
        max_len,
        hidden_dim=256,
        gamma=0.99,
        lr=3e-4,
        polyak=0.995,
        action_noise=0.1
    ):

        # hyperparameters

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lr = lr
        self.polyak = polyak

        self.action_noise = action_noise

        # trackers

        self.prev_observations = None

        # networks

        self.actor_summarizer = TransformerSummarizer(input_dim, hidden_dim, max_len=max_len).to(get_device())
        self.actor_summarizer_targ = create_target(self.actor_summarizer)

        self.critic_summarizer = TransformerSummarizer(input_dim, hidden_dim, max_len=max_len).to(get_device())
        self.critic_summarizer_targ = create_target(self.critic_summarizer)

        self.actor = MLPTanhActor(hidden_dim, action_dim).to(get_device())
        self.actor_targ = create_target(self.actor)

        self.Q = MLPCritic(hidden_dim, action_dim).to(get_device())
        self.Q_targ = create_target(self.Q)

        # optimizers

        self.actor_summarizer_optimizer = optim.Adam(self.actor_summarizer.parameters(), lr=lr)
        self.critic_summarizer_optimizer = optim.Adam(self.critic_summarizer.parameters(), lr=lr)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=lr)

    def reinitialize_prev_observations(self) -> None:
        self.prev_observations = None

    def act(self, observation: np.array, deterministic: bool) -> np.array:

        with torch.no_grad():
            observation = torch.tensor(observation).unsqueeze(0).unsqueeze(0).float().to(get_device())
            summary = self.actor_summarizer(observation, self.prev_observations)  # (1, 1, hidden_size)
            print('Summary:', summary.shape)
            if self.prev_observations is None:
                self.prev_observations = observation
            else:
                self.prev_observations = torch.cat([self.prev_observations, observation], dim=1)  # along seq_len dim
            greedy_action = self.actor(summary).view(-1).cpu().numpy()  # view as 1d -> to cpu -> to numpy
            print('Action:', greedy_action.shape)
            if deterministic:
                return greedy_action
            else:
                return np.clip(greedy_action + self.action_noise * np.random.randn(self.action_dim), -1.0, 1.0)

    def update_networks(self, b: RecurrentBatch):

        bs, num_bptt = b.r.shape[0], b.r.shape[1]

        # compute summary

        actor_summary = self.actor_summarizer(b.o)  # (bs, max_len+1, obs_dim) => (bs, max_len+1, hidden_dim)
        critic_summary = self.critic_summarizer(b.o)

        actor_summary_targ = self.actor_summarizer_targ(b.o)
        critic_summary_targ = self.critic_summarizer_targ(b.o)

        actor_summary_1_T, actor_summary_2_Tplus1 = actor_summary[:, :-1, :], actor_summary_targ[:, 1:, :]
        critic_summary_1_T, critic_summary_2_Tplus1 = critic_summary[:, :-1, :], critic_summary_targ[:, 1:, :]

        assert actor_summary.shape == (bs, num_bptt+1, self.hidden_dim)

        # compute predictions

        predictions = self.Q(critic_summary_1_T, b.a)

        assert predictions.shape == (bs, num_bptt, 1)

        # compute targets

        with torch.no_grad():

            na = self.actor_targ(actor_summary_2_Tplus1)
            targets = b.r + self.gamma * (1 - b.d) * self.Q_targ(critic_summary_2_Tplus1, na)

            assert na.shape == (bs, num_bptt, self.action_dim)
            assert targets.shape == (bs, num_bptt, 1)

        # compute td error

        Q_loss_elementwise = (predictions - targets) ** 2
        Q_loss = mean_of_unmasked_elements(Q_loss_elementwise, b.m)

        assert Q_loss.shape == ()

        # reduce td error

        self.critic_summarizer_optimizer.zero_grad()
        self.Q_optimizer.zero_grad()

        Q_loss.backward()

        self.critic_summarizer_optimizer.step()
        self.Q_optimizer.step()

        # compute policy loss

        a = self.actor(actor_summary_1_T)
        Q_values = self.Q(critic_summary_1_T.detach(), a)
        policy_loss_elementwise = - Q_values
        policy_loss = mean_of_unmasked_elements(policy_loss_elementwise, b.m)

        assert a.shape == (bs, num_bptt, self.action_dim)
        assert Q_values.shape == (bs, num_bptt, 1)
        assert policy_loss.shape == ()

        # reduce policy loss

        self.actor_summarizer_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        policy_loss.backward()

        self.actor_summarizer_optimizer.step()
        self.actor_optimizer.step()

        # update target networks

        polyak_update(targ_net=self.actor_targ, pred_net=self.actor, polyak=self.polyak)
        polyak_update(targ_net=self.Q_targ, pred_net=self.Q, polyak=self.polyak)

        polyak_update(targ_net=self.actor_summarizer_targ, pred_net=self.actor_summarizer, polyak=self.polyak)
        polyak_update(targ_net=self.critic_summarizer_targ, pred_net=self.critic_summarizer, polyak=self.polyak)

        return {
            # for learning the q functions
            '(qfunc) Q pred': float(mean_of_unmasked_elements(predictions, b.m)),
            '(qfunc) Q loss': float(Q_loss),
            # for learning the actor
            '(actor) Q value': float(mean_of_unmasked_elements(Q_values, b.m)),
        }

    def save_actor(self, save_dir: str) -> None:
        save_net(net=self.actor_summarizer, save_dir=save_dir, save_name="actor_summarizer.pth")
        save_net(net=self.actor, save_dir=save_dir, save_name="actor.pth")

    def load_actor(self, save_dir: str) -> None:
        load_net(net=self.actor_summarizer, save_dir=save_dir, save_name="actor_summarizer.pth")
        load_net(net=self.actor, save_dir=save_dir, save_name="actor.pth")

    def copy_networks_from(self, another_recurrent_ddpg):

        self.actor_summarizer.load_state_dict(another_recurrent_ddpg.actor_summarizer.state_dict())
        self.actor_summarizer_targ.load_state_dict(another_recurrent_ddpg.actor_summarizer_targ.state_dict())

        self.critic_summarizer.load_state_dict(another_recurrent_ddpg.critic_summarizer.state_dict())
        self.critic_summarizer_targ.load_state_dict(another_recurrent_ddpg.critic_summarizer_targ.state_dict())

        self.actor.load_state_dict(another_recurrent_ddpg.actor.state_dict())
        self.actor_targ.load_state_dict(another_recurrent_ddpg.actor_targ.state_dict())

        self.Q.load_state_dict(another_recurrent_ddpg.Q.state_dict())
        self.Q_targ.load_state_dict(another_recurrent_ddpg.Q_targ.state_dict())
