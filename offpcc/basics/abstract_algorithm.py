from typing import Union
from abc import ABC, abstractmethod
import numpy as np
from basics.replay_buffer import Batch


class OffPolicyRLAlgorithm(ABC):

    @abstractmethod
    def act(self, state: np.array, deterministic: bool) -> np.array:
        """Only called during online rollout"""
        pass

    @abstractmethod
    def update_networks(self, b: Batch) -> dict:  # return a dictonary of stats that you want to track; could be empty
        """Only called during learning"""
        pass

    @abstractmethod
    def save_networks(self, save_dir: str) -> None:
        """Save all the networks"""
        pass

    @abstractmethod
    def load_actor(self, save_dir: str) -> None:
        """Load the actor network only"""
        pass

    @abstractmethod
    def load_networks(self, save_dir: str) -> None:
        """Load all the networks"""
        pass

    def restart(self) -> None:
        """For recurrent agents only; called at the beginning of each episode to reset hidden states"""
        pass
