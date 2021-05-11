from typing import Union
from abc import ABC, abstractmethod
import numpy as np
from basics.replay_buffer import Batch


class OffPolicyRLAlgorithm(ABC):

    """
    Only these methods should be called in basics.trainer.Trainer.
    Also, they have to be called with the correct signature as described below.
    """

    @abstractmethod
    def act(self, state: np.array, deterministic: bool) -> Union[int, np.array]:
        """int is for discrete action space; np.array is for continuous action space"""
        pass

    @abstractmethod
    def update_networks(self, b: Batch) -> None:
        pass

    @abstractmethod
    def save_actor(self, save_dir: str, save_filename: str) -> None:
        pass

    @abstractmethod
    def load_actor(self, save_dir: str, save_filename: str) -> None:
        pass
