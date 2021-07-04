from abc import ABC, abstractmethod

import numpy as np

from basics.replay_buffer import Batch
from basics.replay_buffer_recurrent import RecurrentBatch


class OffPolicyRLAlgorithm(ABC):

    @abstractmethod
    def act(self, state: np.array, deterministic: bool) -> np.array:
        pass

    @abstractmethod
    def update_networks(self, b: Batch) -> dict:
        pass

    @abstractmethod
    def save_actor(self, save_dir: str) -> None:
        pass

    @abstractmethod
    def load_actor(self, save_dir: str) -> None:
        pass


class RecurrentOffPolicyRLAlgorithm(ABC):

    """
    Based on Liskov Substitution Principle, this class should not inherit from OffPolicyRLAlgorithm
    """

    @abstractmethod
    def reinitialize_hidden(self) -> None:
        pass

    @abstractmethod
    def act(self, state: np.array, deterministic: bool) -> np.array:
        pass

    @abstractmethod
    def update_networks(self, b: RecurrentBatch) -> dict:
        pass

    @abstractmethod
    def save_actor(self, save_dir: str) -> None:
        pass

    @abstractmethod
    def load_actor(self, save_dir: str) -> None:
        pass

    @abstractmethod
    def copy_networks_from(self, algorithm) -> None:
        pass
