import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from tqdm import tqdm

from utils import predict
from model import Model


# agregrar una seed


def calculate_mse(y, y_pred):
    # hacer mascara con los largos de secuencia
    mask = y != 0
    return (((y - y_pred) ** 2) * mask).sum(dim=-1)


def define_input_and_target(x):
    new_x = list()
    y = list()
    for xi in x:
        new_x.append(xi[:, :-1])
        y.append(xi[1, 1:])
    return new_x, y


def list2torch(x):
    x_torch = [torch.tensor(xi, dtype=torch.float) for xi in x]
    return x_torch


def estimate_deltas(data_list):
    deltas = list()
    for dat in tqdm(data_list):
        dt = dat[0, 1:] - dat[0, :-1]
        dx = dat[1, 1:] - dat[1, :-1]
        # se agrega un delta ficticio = 0
        dt = np.insert(dt, 0, 0)
        dx = np.insert(dx, 0, 0)
        deltas.append(np.vstack((dt, dx)))
    return deltas


def sample_data(n):
    metadata = list()
    data = list()
    for _ in tqdm(range(n)):
        t, signal, freq, epsilon = generate_signal()
        signal = np.vstack((t, signal))
        metadata.append(dict(frequency=freq, epsilon=epsilon))
        data.append(signal)
    metadata = pd.DataFrame(metadata)
    return data, metadata


def generate_signal():
    f = np.random.uniform(30, 40)
    epsilon = 10 ** np.random.uniform(0, 1)
    max_steps = np.random.randint(16, 32)
    dt = np.random.uniform(0, 0.01, size=max_steps)
    t = np.cumsum(dt)
    signal = np.exp(- t * epsilon) * np.sin(2 * np.pi * f * t)
    return t, signal, f, epsilon


# actualizar
device = "cpu"

# genera datos de juguete
data, metadata = sample_data(1000)

# anclar tus datos
# dim0: tiempo
# dim1: señal
# shape: ndim, nsamples

import matplotlib.pyplot as plt
# plt.figure()
# for i in range(len(data))[:10]:
#     plt.plot(data[i][0], data[i][1])
# plt.show()

# procesamiento de las señales
x = estimate_deltas(data)
x = list2torch(x)
new_x, y = define_input_and_target(x)
new_x = [xi.T for xi in new_x]

# vamos a hacer una normlizacion segun el maximo por componente
x_features = metadata.values
cte = x_features.max(axis=0, keepdims=True)
x_features = x_features / cte
x_features = torch.tensor(x_features, dtype=torch.float)

n_rep = 3
_, ndim = x_features.shape
nhidden = int(n_rep * ndim)
nlayers = 2
dropout = 0.
bias = True
batch_first = True
bidirectional = False
output_size = 1
input_size = 2

model = Model(input_size, nhidden, nlayers, bias, batch_first, dropout, bidirectional, output_size, n_rep)
model.to(device)

# falta parte de mini-batches

x_train = torch.nn.utils.rnn.pack_sequence(new_x, enforce_sorted=False)
y_train = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)
seq_len = torch.tensor([len(yi) for yi in y])

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

niter = 2
loss_values = list()
for idx in tqdm(range(niter)):
    y_pred = model(x_train, x_features)
    loss = calculate_mse(y_train, y_pred).mean()
    loss_values.append(loss.item())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# guardar el modelo

plt.figure()
plt.plot(loss_values)
plt.show()

# evaluacion
data, metadata = sample_data(10)

x = estimate_deltas(data)
x = list2torch(x)

# vamos a hacer una normlizacion segun el maximo por componente
x_features = metadata.values
x_features = x_features / cte
x_features = torch.tensor(x_features, dtype=torch.float)

dt_mean = np.mean([xi[0].mean().item() for xi in x])

n_samples = 24
delta_t = dt_mean
predicted_ts = list()
for feat in x_features:
    predicted_ts.append(predict(feat, n_samples, delta_t, model))

dt = np.ones(n_samples) * delta_t
t = np.cumsum(dt)

for idx in range(len(data)):
    plt.figure()
    plt.plot(data[idx][0], data[idx][1], label="real")
    plt.plot(t, np.cumsum(predicted_ts[idx]), label="pred")
    plt.legend()
    plt.savefig(f"results/fig_{idx}.png", dpi=200)
