import tensorflow as tf
from tensorflow import keras
import numpy as np
import gin


@gin.configurable(module=__name__)
class CustomModel(keras.Model):

    def __init__(self):
        super().__init__()
        self.layers_ = keras.layers.Dense(10)

    def do(self, inputs):
        return self.layers_(inputs)


model = CustomModel()
optimizer = keras.optimizers.Adam(learning_rate=1e-3)


@tf.function
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        prediction = model.do(tf.clip_by_value(x, -1, 1)) + tf.random.normal(x.shape)
        prediction2 = model.do(x)
        final_prediction = tf.math.minimum(prediction, prediction2)
        loss = tf.reduce_sum((final_prediction - y) ** 2)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))


for i in range(10):
    train_one_step(model, optimizer, np.random.randn(64, 10).astype('float32'), np.random.randn(64, 10).astype('float32'))
