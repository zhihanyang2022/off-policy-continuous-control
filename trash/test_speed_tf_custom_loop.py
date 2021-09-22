from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0


class CustomModel(keras.Model):

    def __init__(self):
        super().__init__()
        self.l1 = layers.Dense(256, activation='relu')
        self.l2 = layers.Dense(256, activation='relu')
        self.l3 = layers.Dense(10)

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.l2(inputs)
        x = self.l3(inputs)
        return x


model = CustomModel()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=2)
