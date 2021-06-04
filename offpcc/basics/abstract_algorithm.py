from abc import ABC, abstractmethod
import os

import numpy as np
import torch
import torch.nn as nn

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
        self.networks_dict = {}

    @abstractmethod
    def act(self, state: np.array, deterministic: bool) -> np.array:
        """Only called during online rollout"""
        pass

    def polyak_update(self, targ_net: nn.Module, pred_net: nn.Module) -> None:
        with torch.no_grad():  # no grad is not actually required here; only for sanity check
            for targ_p, p in zip(targ_net.parameters(), pred_net.parameters()):
                targ_p.data.copy_(targ_p.data * self.polyak + p.data * (1 - self.polyak))

    @abstractmethod
    def update_networks(self, b: Batch) -> dict:  # return a dictonary of stats that you want to track; could be empty
        """Only called during learning"""
        pass

    def save_networks(self, save_dir: str) -> None:
        """Save all the networks"""
        for network_name, network in self.networks_dict.items():
            torch.save(network.state_dict(), os.path.join(save_dir, f'{network_name}.pth'))

    def load_actor(self, save_dir: str) -> None:
        """Load the actor network only"""
        self.actor.load_state_dict(
            torch.load(
                os.path.join(save_dir, 'actor.pth'),
                map_location=torch.device(get_device())
            )
        )

    def load_networks(self, save_dir: str) -> None:
        """Load all the networks"""
        for network_name, network in self.networks_dict.items():
            network.load_state_dict(
                torch.load(
                    os.path.join(save_dir, f'{network_name}.pth'),
                    map_location=torch.device(get_device())
                )
            )


class RecurrentOffPolicyRLAlgorithm(ABC, OffPolicyRLAlgorithm):

    def __init__(self, input_dim, action_dim, gamma, lr, polyak):
        super().__init__(input_dim, action_dim, gamma, lr, polyak)
        self.h_and_c = None
        self.actor_lstm = None

    def reinitialize_hidden(self) -> None:
        """For recurrent agents only; called at the beginning of each episode to reset hidden states"""
        self.h_and_c = None

    def load_actor(self, save_dir: str) -> None:
        """Load the actor network only"""
        self.actor_lstm.load_state_dict(
            torch.load(
                os.path.join(save_dir, 'actor_lstm.pth'),
                map_location=torch.device(get_device())
            )
        )
        self.actor.load_state_dict(
            torch.load(
                os.path.join(save_dir, 'actor.pth'),
                map_location=torch.device(get_device())
            )
        )
