import os

import numpy as np
import tensorflow as tf
from tensorflow import keras


def polyak_update(targ_net: keras.Model, pred_net: keras.Model, polyak: float) -> None:

    mixed_weights = []

    for (pred_net_w, targ_net_w) in zip(pred_net.weights, targ_net.weights):
        mixed_weights.append((pred_net_w * (1 - polyak) + targ_net_w * polyak).numpy())

    targ_net.set_weights(mixed_weights)


def mean_of_unmasked_elements(tensor: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tensor * mask) / tf.reduce_sum(mask) * np.prod(mask.shape)


def save_net(net: nn.Module, save_dir: str, save_name: str) -> None:
    torch.save(net.state_dict(), os.path.join(save_dir, save_name))


def load_net(net: nn.Module, save_dir: str, save_name: str) -> None:
    net.load_state_dict(
        torch.load(os.path.join(save_dir, save_name), map_location=torch.device(get_device()))
    )
