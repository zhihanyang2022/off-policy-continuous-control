import os
import tensorflow.keras as keras


def polyak_update(targ_net: keras.Model, pred_net: keras.Model, polyak: float) -> None:

    mixed_weights = []

    for (pred_net_w, targ_net_w) in zip(pred_net.weights, targ_net.weights):
        mixed_weights.append(pred_net_w * (1 - polyak) + targ_net_w * polyak)

    targ_net.set_weights(mixed_weights)


def save_net(net: keras.Model, save_dir: str, save_name: str) -> None:
    net.save_weights(os.path.join(save_dir, save_name))


def load_net(net: keras.Model, save_dir: str, save_name: str) -> None:
    net.load_weights(os.path.join(save_dir, save_name))
