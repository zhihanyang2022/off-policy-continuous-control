import gin
import random
import torch
from collections import namedtuple, deque
from basics.utils import get_device

Transition = namedtuple('Transition', 's a r ns d')
Batch = namedtuple('Batch', 's a r ns d')


@gin.configurable(module=__name__)
class ReplayBuffer:
    """Just a standard FIFO replay buffer."""

    def __init__(self, capacity=int(1e6), batch_size=100):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size

    def push(self, s, a, r, ns, d) -> None:
        self.memory.appendleft(Transition(s, a, r, ns, d))

    def can_sample(self):
        return len(self.memory) >= self.batch_size

    def sample(self) -> Batch:
        assert len(self.memory) >= self.batch_size, "Please increase update_after to be >= batch_size"
        transitions = random.choices(self.memory, k=self.batch_size)  # sampling WITH replacement
        batch_raw = Batch(*zip(*transitions))
        s = torch.tensor(batch_raw.s, dtype=torch.float).view(self.batch_size, -1)
        a = torch.tensor(batch_raw.a, dtype=torch.float).view(self.batch_size, -1)
        r = torch.tensor(batch_raw.r, dtype=torch.float).view(self.batch_size, 1)
        ns = torch.tensor(batch_raw.ns, dtype=torch.float).view(self.batch_size, -1)
        d = torch.tensor(batch_raw.d, dtype=torch.float).view(self.batch_size, 1)
        return Batch(*list(map(lambda x: x.to(get_device()), [s, a, r, ns, d])))
