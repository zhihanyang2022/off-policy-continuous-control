from typing import Union
from abc import ABC, abstractmethod
import numpy as np
from basics.buffer import Batch

class OffPolicyRLAlgorithm(ABC):

    @abstractmethod
    def act(self, state: np.array) -> Union[int, np.array]:
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
