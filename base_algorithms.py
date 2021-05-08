from abc import ABC, abstractmethod
import numpy as np
import torch.nn as nn
from buffer import Batch

class OffPolicyRLAlgorithmDiscreteAction(ABC):

    @abstractmethod
    def __init__(self, actor: nn.Module, gamma: float, lr: float, polyak: float):
        self.actor = actor

    @abstractmethod
    def act(self, state: np.array) -> int:
        pass

    @abstractmethod
    def update_networks(self, batch: Batch) -> None:
        pass

    @abstractmethod
    def save_actor(self, save_dir: str, save_filename: str) -> None:
        pass

    @abstractmethod
    def load_actor(self, save_dir: str, save_filename: str) -> None:
        pass

class OffPolicyRLAlgorithmDiscreteAction(ABC):

    @abstractmethod
    def __init__(self, actor: nn.Module, critic: nn.Module, gamma: float, lr: float, polyak: float):
        pass

    @abstractmethod
    def act(self, state: np.array) -> int:
        pass

    @abstractmethod
    def update_networks(self, batch: Batch) -> None:
        pass

    @abstractmethod
    def save_actor(self, save_dir: str, save_filename: str) -> None:
        pass

    @abstractmethod
    def load_actor(self, save_dir: str, save_filename: str) -> None:
        pass
