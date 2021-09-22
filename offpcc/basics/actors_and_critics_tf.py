import gin
import tensorflow as tf
from tensorflow import keras


@gin.configurable(module=__name__)
def make_MLP(num_in, num_out, final_activation, hidden_dimensions=(256, 256)):

    tensor_dimensions = [num_in]
    if hidden_dimensions is not None:
        tensor_dimensions.extend(hidden_dimensions)
    if num_out is not None:
        tensor_dimensions.append(num_out)

    num_layers = len(tensor_dimensions)  # now including the input layer
    list_of_layers = []

    # tf uses lazy instantiation, so input dimension is inferred during forward pass

    for i, output_dimension in enumerate(tensor_dimensions):
        if i == 0:
            list_of_layers.append(tf.keras.Input(output_dimension))
        elif i == num_layers - 1:
            if final_activation is None:
                list_of_layers.append(tf.keras.layers.Dense(output_dimension))
            else:
                list_of_layers.append(tf.keras.layers.Dense(output_dimension, activation=final_activation))
        else:
            list_of_layers.append(tf.keras.layers.Dense(output_dimension, activation='relu'))
    net = keras.Sequential(list_of_layers)

    return net  # actual_num_out is not required


class MLPTanhActor(keras.Model):
    """Output actions from [-1, 1]."""
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = make_MLP(num_in=input_dim, num_out=action_dim, final_activation='tanh')
        self.build(input_shape=(None, input_dim))  # create the parameters within init based on call; crucial

    def call(self, states: tf.Tensor):
        return self.net(states)


class MLPGaussianActor(keras.Model):
    """Output parameters for some multi-dimensional zero-covariance Gaussian distribution."""

    def __init__(self, input_dim, action_dim):
        super().__init__()

        self.shared_net = make_MLP(num_in=input_dim, num_out=None, final_activation='relu')
        self.means_layer = tf.keras.layers.Dense(action_dim)
        self.log_stds_layer = tf.keras.layers.Dense(action_dim)

        self.build(input_shape=(None, input_dim))

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def call(self, states: tf.Tensor) -> tuple:
        out = self.shared_net(states)
        means, log_stds = self.means_layer(out), self.log_stds_layer(out)
        stds = tf.exp(tf.clip_by_value(log_stds, self.LOG_STD_MIN, self.LOG_STD_MAX))
        return means, stds


class MLPCritic(keras.Model):

    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = make_MLP(num_in=input_dim + action_dim, num_out=1, final_activation=None)
        self.build(input_shape=(None, input_dim + action_dim))

    def call(self, states_and_actions: tuple):
        return self.net(tf.concat(states_and_actions, axis=-1))
