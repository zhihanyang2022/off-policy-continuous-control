import gin

import numpy as np
import torch
import torch.optim as optim

from basics.abstract_algorithms import RecurrentOffPolicyRLAlgorithm
from basics.summarizer import Summarizer
from basics.actors_and_critics import MLPTanhActor, MLPCritic
from basics.replay_buffer_recurrent import RecurrentBatch
from basics.utils import get_device, create_target, mean_of_unmasked_elements, polyak_update, save_net, load_net
from basics.action_noise_scheduler import ActionNoiseScheduler


@gin.configurable(module=__name__)
class RecurrentDDPG(RecurrentOffPolicyRLAlgorithm):

    """Deep deterministic policy gradient with recurrent networks"""

    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_dim=256,
        gamma=0.99,
        lr=3e-4,
        polyak=0.995,
        action_noise=1.0,
        action_noise_schedule=None,
        exploration_mode="dqn_style",  # or "standard"
    ):

        # hyperparameters

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lr = lr
        self.polyak = polyak

        self.action_noise = action_noise
        self.action_noise_schedule = action_noise_schedule
        self.exploration_mode = exploration_mode

        assert self.exploration_mode in ["dqn_style", "standard"], f"{exploration_mode} is not a valid exploration mode"

        if self.action_noise_schedule is not None:
            self.action_noise_scheduler = ActionNoiseScheduler(init_action_noise=action_noise,
                                                               schedule=action_noise_schedule)

        # trackers

        self.hidden = None

        # networks

        self.actor_summarizer = Summarizer(input_dim, hidden_dim).to(get_device())
        self.actor_summarizer_targ = create_target(self.actor_summarizer)

        self.critic_summarizer = Summarizer(input_dim, hidden_dim).to(get_device())
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
                if self.exploration_mode == "standard":
                    return np.clip(greedy_action + self.action_noise * np.random.randn(self.action_dim), -1.0, 1.0)
                elif self.exploration_mode == "dqn_style":
                    if np.random.uniform() > self.action_noise:
                        return greedy_action
                    else:
                        return np.random.uniform(-1.0, 1.0)

    def update_networks(self, b: RecurrentBatch):

        bs, num_bptt = b.r.shape[0], b.r.shape[1]

        # compute summary

        actor_summary = self.actor_summarizer(b.o)
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

        # update action noise

        if self.action_noise_scheduler is not None:
            self.action_noise = self.action_noise_scheduler.get_new_action_noise()

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
