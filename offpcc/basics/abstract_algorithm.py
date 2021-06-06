from abc import ABC, abstractmethod
import os

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from basics.replay_buffer import Batch
from basics.replay_buffer_recurrent import RecurrentBatch
from basics.cuda_utils import get_device


class OffPolicyRLAlgorithm(ABC):

    def __init__(self, input_dim, action_dim, gamma, lr, polyak):

        """Simplify saves input arguments to instance attributes"""

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.polyak = polyak

        self.actor = None

    # methods for collecting rollouts

    @abstractmethod
    def act(self, state: np.array, deterministic: bool) -> np.array:
        """Only called during online rollout"""
        pass

    # methods for updating networks using batches

    def polyak_update(self, targ_net: nn.Module, pred_net: nn.Module) -> None:
        with torch.no_grad():  # no grad is not actually required here; only for sanity check
            for targ_p, p in zip(targ_net.parameters(), pred_net.parameters()):
                targ_p.data.copy_(targ_p.data * self.polyak + p.data * (1 - self.polyak))

    @abstractmethod
    def update_networks(self, b: Batch) -> dict:  # return a dictonary of stats that you want to track; could be empty
        """Only called during learning"""
        pass

    # methods for saving networks

    def save_actor(self, save_dir: str) -> None:
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'actor.pth'))

    def load_actor(self, save_dir: str) -> None:
        """Load the actor network only"""
        self.actor.load_state_dict(
            torch.load(os.path.join(save_dir, 'actor.pth'), map_location=torch.device(get_device()))
        )


class RecurrentOffPolicyRLAlgorithm(OffPolicyRLAlgorithm):

    def __init__(self, input_dim, action_dim, gamma, lr, polyak):
        super().__init__(input_dim, action_dim, gamma, lr, polyak)
        self.h_and_c = None
        self.actor_lstm = None

    # methods for collecting rollouts

    def reinitialize_hidden(self) -> None:
        """For recurrent agents only; called at the beginning of each episode to reset hidden states"""
        self.h_and_c = None

    # methods for updating networks using batches

    @staticmethod
    def rescale_loss(loss: torch.tensor, mask: torch.tensor) -> torch.tensor:
        return loss / mask.sum() * np.prod(mask.shape)

    @staticmethod
    def feed_lstm(lstm, o):
        """Nothing special; just making code more readbale in update_networks"""
        lstm.flatten_parameters()  # prevent some arbitrary error that I don't understand
        h, h_and_c = lstm(o)
        return h

    def save_actor(self, save_dir: str) -> None:
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'actor.pth'))
        torch.save(self.actor_lstm.state_dict(), os.path.join(save_dir, 'actor_lstm.pth'))

    def load_actor(self, save_dir: str) -> None:
        self.actor.load_state_dict(
            torch.load(os.path.join(save_dir, 'actor.pth'), map_location=torch.device(get_device()))
        )
        self.actor_lstm.load_state_dict(
            torch.load(os.path.join(save_dir, 'actor_lstm.pth'), map_location=torch.device(get_device()))
        )
