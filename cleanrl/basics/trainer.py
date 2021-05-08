from typing import Union
import gin

from cleanrl.basics.buffer import Buffer
from cleanrl.basics.abstract_algorithms import OffPolicyRLAlgorithmDiscreteAction, OffPolicyRLAlgorithmContinuousAction

class Trainer:

    def __init__(
        self,
        env,
        algorithm: Union[OffPolicyRLAlgorithmDiscreteAction, OffPolicyRLAlgorithmContinuousAction],
        buffer: Buffer,
        log_dir: str
    ):
        pass

    @gin.configurable(module=__name__)
    def run(
        self,
        num_epochs: int,
        num_exploration_steps: int,
        num_env_steps_per_epoch: int,
        num_grad_steps_per_epoch: int,
        num_eval_episodes_per_epoch: int
    ):
        pass

