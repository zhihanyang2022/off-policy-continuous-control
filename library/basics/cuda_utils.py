import torch


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'