import torch.nn as nn

def clip_gradient(net: nn.Module) -> None:
    for param in net.parameters():
        param.grad.data.clamp_(-1, 1)