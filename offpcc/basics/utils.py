import os

import random
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# def set_random_seed(seed: int, device: str) -> None:
#     """
#     Adapted from stable-baselines3
#     https://stable-baselines3.readthedocs.io/en/latest/_modules/stable_baselines3/common/utils.html
#
#     Seed the different random generators
#     :param seed: (int)
#     :param device: (str)
#     """
#     # Seed python RNG
#     random.seed(seed)
#     # Seed numpy RNG
#     np.random.seed(seed)
#     # seed the RNG for all devices (both CPU and CUDA)
#     torch.manual_seed(seed)
#
#     if device == 'cuda':
#         # Deterministic operations for CuDNN, it may impact performances
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False


def polyak_update(targ_net: nn.Module, pred_net: nn.Module, polyak: float) -> None:
    with torch.no_grad():  # no grad is not actually required here; only for sanity check
        for targ_p, p in zip(targ_net.parameters(), pred_net.parameters()):
            targ_p.data.copy_(targ_p.data * polyak + p.data * (1 - polyak))


def mean_of_unmasked_elements(tensor: torch.tensor, mask: torch.tensor) -> torch.tensor:
    return torch.mean(tensor * mask) / mask.sum() * np.prod(mask.shape)


def save_net(net: nn.Module, save_dir: str, save_name: str) -> None:
    torch.save(net.state_dict(), os.path.join(save_dir, save_name))


def load_net(net: nn.Module, save_dir: str, save_name: str) -> None:
    net.load_state_dict(
        torch.load(os.path.join(save_dir, save_name), map_location=torch.device(get_device()))
    )


def set_requires_grad_flag(net: nn.Module, requires_grad: bool) -> None:
    for p in net.parameters():
        p.requires_grad = requires_grad


def create_target(net: nn.Module) -> nn.Module:
    target = deepcopy(net)
    set_requires_grad_flag(target, False)
    return target
