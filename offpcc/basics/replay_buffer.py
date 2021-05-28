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

    def push(self, s, a, r, ns, d) -> None:
        self.memory.appendleft(Transition(s, a, r, ns, d))

    def sample(self) -> Batch:
        assert len(self.memory) >= self.batch_size, "Please increase update_after to be >= batch_size"
        transitions = random.sample(self.memory, self.batch_size)
        batch_raw = Batch(*zip(*transitions))
        s = torch.tensor(batch_raw.s, dtype=torch.float).view(self.batch_size, -1)
        a = torch.tensor(batch_raw.a, dtype=torch.float).view(self.batch_size, -1)
        r = torch.tensor(batch_raw.r, dtype=torch.float).view(self.batch_size, 1)
        ns = torch.tensor(batch_raw.ns, dtype=torch.float).view(self.batch_size, -1)
        d = torch.tensor(batch_raw.d, dtype=torch.float).view(self.batch_size, 1)
        return Batch(*list(map(lambda x: x.to(get_device()), [s, a, r, ns, d])))
