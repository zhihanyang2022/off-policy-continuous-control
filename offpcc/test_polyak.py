import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

net = torch.nn.Linear(20, 20)
net2 = deepcopy(net)

net.weight = nn.Parameter(torch.ones_like(net.weight))
net2.weight = nn.Parameter(torch.zeros_like(net.weight))

net_opt = optim.Adam(net.parameters(), lr=1e-3)

net2.requires_grad_(False)

print(net.weight.requires_grad)
print(net2.weight.requires_grad)

for p2, p in zip(net2.parameters(), net.parameters()):
    p2.data.add_(p2.data)

print(net.weight.requires_grad)
print(net2.weight.requires_grad)
