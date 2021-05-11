import gin
import random
import torch
from collections import namedtuple, deque
from basics.cuda_utils import get_device

Transition = namedtuple('Transition', 's a r ns d')
Batch = namedtuple('Batch', 's a r ns d')


@gin.configurable(module=__name__)
class ReplayBuffer:
    """Just a standard FIFO replay buffer."""

    def __init__(self, capacity=gin.REQUIRED, batch_size=gin.REQUIRED):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size

    def push(self, transition: Transition) -> None:
        self.memory.appendleft(transition)

    def ready(self) -> bool:
        # TODO: change to use numpy to avoid this issue
        # TODO: then everything rely on update_after
        """Otherwise random.sample raises error."""
        return len(self.memory) >= self.batch_size

    def sample(self) -> Batch:
        batch = random.sample(self.memory, self.batch_size)
        batch = Batch(*zip(*batch))
        s = torch.tensor(batch.s, dtype=torch.float).view(self.batch_size, -1).to(get_device())
        a = torch.tensor(batch.a, dtype=torch.float).view(self.batch_size, -1).to(get_device())
        r = torch.tensor(batch.r, dtype=torch.float).view(self.batch_size, 1).to(get_device())
        ns = torch.tensor(batch.ns, dtype=torch.float).view(self.batch_size, -1).to(get_device())
        d = torch.tensor(batch.d, dtype=torch.float).view(self.batch_size, 1).to(get_device())
        return Batch(s, a, r, ns, d)
