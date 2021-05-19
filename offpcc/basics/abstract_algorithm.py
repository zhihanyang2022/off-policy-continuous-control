from typing import Union
from abc import ABC, abstractmethod
import numpy as np
from basics.replay_buffer import Batch


class OffPolicyRLAlgorithm(ABC):

    @abstractmethod
    def act(self, state: np.array, deterministic: bool) -> Union[int, np.array]:
        """int is for discrete action space; np.array is for continuous action space"""
        pass

    @abstractmethod
    def update_networks(self, b: Batch) -> dict:   # return a dictonary of stats that you want to track; could be empty
        pass

    @abstractmethod
    def save_networks(self, save_dir: str) -> None:
        pass

    @abstractmethod
    def load_networks(self, save_dir: str) -> None:
        pass
