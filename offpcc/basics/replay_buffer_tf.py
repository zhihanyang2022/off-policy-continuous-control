import gin
import random
from collections import deque, namedtuple
import tensorflow as tf

Transition = namedtuple('Transition', 's a r ns d')
Batch = namedtuple('Batch', 's a r ns d')


@gin.configurable(module=__name__)
class ReplayBuffer_tf:
    """Just a standard FIFO replay buffer."""

    def __init__(self, capacity=int(1e6), batch_size=100):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size

    def push(self, s, a, r, ns, d) -> None:
        self.memory.appendleft(Transition(s, a, r, ns, d))

    def sample(self) -> Batch:
        assert len(self.memory) >= self.batch_size, "Please increase update_after to be >= batch_size"
        transitions = random.choices(self.memory, k=self.batch_size)  # sampling WITH replacement
        batch_raw = Batch(*zip(*transitions))
        s = tf.reshape(tf.convert_to_tensor(batch_raw.s, dtype=tf.float32), (self.batch_size, -1))
        a = tf.reshape(tf.convert_to_tensor(batch_raw.a, dtype=tf.float32), (self.batch_size, -1))
        r = tf.reshape(tf.convert_to_tensor(batch_raw.r, dtype=tf.float32), (self.batch_size, 1))
        ns = tf.reshape(tf.convert_to_tensor(batch_raw.ns, dtype=tf.float32), (self.batch_size, -1))
        d = tf.reshape(tf.convert_to_tensor(batch_raw.d, dtype=tf.float32), (self.batch_size, 1))
        return Batch(s, a, r, ns, d)
