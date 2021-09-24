import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tensorflow.keras.datasets import mnist
import time


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

num_epochs = 5
for epoch in range(num_epochs):
    start = time.perf_counter()
    for xb, yb in train_dl:

        yb_pred = model(xb)
        loss = loss_fn(yb_pred, yb.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        yb_pred = model(xb)
        loss = loss_fn(yb_pred, yb.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        yb_pred = model(xb)
        loss = loss_fn(yb_pred, yb.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end = time.perf_counter()
    print(epoch + 1, end - start)
