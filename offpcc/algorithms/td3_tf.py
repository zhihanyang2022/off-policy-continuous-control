import gin

import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from basics.abstract_algorithms import OffPolicyRLAlgorithm
from basics.actors_and_critics_tf import MLPTanhActor, MLPCritic
from basics.replay_buffer import Batch
from basics.utils_tf import polyak_update, save_net, load_net


@gin.configurable(module=__name__)
class TD3_tf(OffPolicyRLAlgorithm):

    def __init__(
        self,
        input_dim,
        action_dim,
        gamma=0.99,
        lr=3e-4,
        lr_schedule=None,
        polyak=0.995,
        action_noise=0.1,  # standard deviation of action noise
        target_noise=0.2,  # standard deviation of target smoothing noise
        noise_clip=0.5,  # max abs value of target smoothing noise
        policy_delay=2
    ):

        # hyper-parameters

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.polyak = polyak

        self.action_noise = action_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip

        self.policy_delay = policy_delay

        # trackers

        self.num_Q_updates = 0  # for delaying updates
        self.mean_Q1_value = 0  # for logging; the actor does not get updated every iteration,
        # so this statistic is not available every iteration

        # networks
        # (keras.models.clone_model cannot be used for subclassed models)
        # (weirdly, the weights must be converted to numpy for set_weights to work)

        self.actor = MLPTanhActor(input_dim, action_dim)
        self.actor_targ = MLPTanhActor(input_dim, action_dim)
        self.actor_targ.set_weights([w.numpy() for w in self.actor.weights])

        self.Q1 = MLPCritic(input_dim, action_dim)
        self.Q1_targ = MLPCritic(input_dim, action_dim)
        self.Q1_targ.set_weights([w.numpy() for w in self.Q1.weights])

        self.Q2 = MLPCritic(input_dim, action_dim)
        self.Q2_targ = MLPCritic(input_dim, action_dim)
        self.Q2_targ.set_weights([w.numpy() for w in self.Q2.weights])

        # optimizers

        self.actor_optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.Q1_optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.Q2_optimizer = keras.optimizers.Adam(learning_rate=lr)

    def act(self, state: np.array, deterministic: bool) -> np.array:
        state = tf.convert_to_tensor(state.reshape(1, -1))
        greedy_action = self.actor(state).numpy().reshape(-1)  # to numpy -> view as 1d
        if deterministic:
            return greedy_action
        else:
            return np.clip(greedy_action + self.action_noise * np.random.randn(self.action_dim), -1.0, 1.0)

    def update_networks(self, b: Batch):

        bs = len(b.ns)  # for shape checking

        # compute targets

        na = self.actor_targ(b.ns)
        noise = tf.clip_by_value(
            tf.random.normal(na.shape) * self.target_noise, -self.noise_clip, self.noise_clip
        )
        smoothed_na = tf.clip_by_value(na + noise, -1, 1)

        n_min_Q_targ = tf.math.minimum(self.Q1_targ((b.ns, smoothed_na)), self.Q2_targ((b.ns, smoothed_na)))

        targets = b.r + self.gamma * (1 - b.d) * n_min_Q_targ

        assert na.shape == (bs, self.action_dim)
        assert n_min_Q_targ.shape == (bs, 1)
        assert targets.shape == (bs, 1)

        with tf.GradientTape(persistent=True) as tape:

            # compute predictions

            Q1_predictions = self.Q1((b.s, b.a))
            Q2_predictions = self.Q2((b.s, b.a))

            # compute td error

            Q1_loss = tf.reduce_mean((Q1_predictions - targets) ** 2)
            Q2_loss = tf.reduce_mean((Q2_predictions - targets) ** 2)

        assert Q1_loss.shape == ()
        assert Q2_loss.shape == ()

        # reduce td error

        Q1_gradients = tape.gradient(Q1_loss, self.Q1.trainable_weights)
        self.Q1_optimizer.apply_gradients(zip(Q1_gradients, self.Q1.trainable_weights))

        Q2_gradients = tape.gradient(Q2_loss, self.Q2.trainable_weights)
        self.Q2_optimizer.apply_gradients(zip(Q2_gradients, self.Q2.trainable_weights))

        self.num_Q_updates += 1

        if self.num_Q_updates % self.policy_delay == 0:  # delayed policy update; special in TD3

            # compute policy loss



            with tf.GradientTape() as tape:

                a = self.actor(b.s)
                Q1_values = self.Q1((b.s, a))
                policy_loss = - tf.reduce_mean(Q1_values)

            self.mean_Q1_value = float(-policy_loss)
            assert a.shape == (bs, self.action_dim)
            assert Q1_values.shape == (bs, 1)
            assert policy_loss.shape == ()

            # reduce policy loss

            policy_gradients = tape.gradient(policy_loss, self.actor.trainable_weights)
            self.actor_optimizer.apply_gradients(zip(policy_gradients, self.actor.trainable_weights))

            # update target networks

            polyak_update(targ_net=self.actor_targ, pred_net=self.actor, polyak=self.polyak)
            polyak_update(targ_net=self.Q1_targ, pred_net=self.Q1, polyak=self.polyak)
            polyak_update(targ_net=self.Q2_targ, pred_net=self.Q2, polyak=self.polyak)

        return {
            # for learning the q functions
            '(qfunc) Q1 pred': float(tf.reduce_mean(Q1_predictions)),
            '(qfunc) Q2 pred': float(tf.reduce_mean(Q2_predictions)),
            '(qfunc) Q1 loss': float(Q1_loss),
            '(qfunc) Q2 loss': float(Q2_loss),
            # for learning the actor
            '(actor) Q1 value': self.mean_Q1_value
        }

    def save_actor(self, save_dir: str) -> None:
        save_net(net=self.actor, save_dir=save_dir, save_name="actor.h5")

    def load_actor(self, save_dir: str) -> None:
        load_net(net=self.actor, save_dir=save_dir, save_name="actor.h5")
