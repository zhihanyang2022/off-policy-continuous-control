import torch
from copy import deepcopy

net = torch.nn.Linear(20, 20)
net2 = deepcopy(net)

print(net.weight)