import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import torch
from torch.utils.data import TensorDataset, DataLoader
import time


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0


class CustomModel(keras.Model):

    def __init__(self):
        super().__init__()
        self.l1 = layers.Dense(256, activation='relu')
        self.l2 = layers.Dense(256, activation='relu')
        self.l3 = layers.Dense(10)
        self.build(input_shape=(None, 784))

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        return x


model = CustomModel()
optimizer = keras.optimizers.Adam(lr=1e-3)

train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

num_epochs = 5
for epoch in range(num_epochs):
    start = time.perf_counter()
    for xb, yb in train_dl:

        xb, yb = xb.numpy(), yb.numpy()

        with tf.GradientTape() as tape:
            yb_pred = model(xb)
            loss = keras.metrics.sparse_categorical_crossentropy(yb, yb_pred, from_logits=True)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        with tf.GradientTape() as tape:
            yb_pred = model(xb)
            loss = keras.metrics.sparse_categorical_crossentropy(yb, yb_pred, from_logits=True)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        with tf.GradientTape() as tape:
            yb_pred = model(xb)
            loss = keras.metrics.sparse_categorical_crossentropy(yb, yb_pred, from_logits=True)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    end = time.perf_counter()
    print(epoch + 1, end - start)


