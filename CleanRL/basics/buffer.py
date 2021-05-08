import gin
from collections import namedtuple

Batch = namedtuple('Batch', 's a r ns d')

@gin.configurable(module=__name__)
class Buffer:

    def __init__(self, capacity, batch_size):
        pass

    def ready(self):
        pass

    def sample(self):
        pass