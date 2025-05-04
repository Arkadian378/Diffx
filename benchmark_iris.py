import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from core.tensor import Tensor
from core.nn import MLP, cross_entropy, mse_loss
from core.optim import SGD

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

data = load_iris()
X = data["data"]
y = data["target"].reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)

def to_tensor(x):
    return Tensor([[float(val)] for val in x])

def to_tensor_output(y):
    return Tensor([[float(v)] for v in y])

X_train_diffx = [to_tensor(x) for x in X_train]
y_train_diffx = [to_tensor_output(y) for y in y_train]
X_test_diffx  = [to_tensor(x) for x in X_test]
y_test_diffx  = [np.argmax(y) for y in y_test]

model_diffx = MLP(in_features=4, hidden_layers=[10], out_features=3)
opt = SGD(model_diffx.parameters(), lr=0.1)

losses_diffx = []
start = time.time()

def softmax(tensor):
    values = [v[0] for v in tensor.data]  
    max_val = max(v.data for v in values) 

    exps = [(v - max_val).exp() for v in values]  
    total = sum(exps)
    normed = [[e / total] for e in exps]
    return Tensor(normed)


for epoch in range(100):
    total_loss = 0
    correct = 0

    for x,y in zip(X_train_diffx, y_train_diffx):
        pred_raw = model_diffx(x)
        pred = softmax(pred_raw)
        loss = cross_entropy(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.data
        pred_class = np.argmax([v[0].data for v in pred.data])
        target_class = np.argmax([v[0].data for v in y.data])
        correct += (pred_class == target_class)
    
    acc = correct / len(X_train_diffx)
    losses_diffx.append(total_loss / len(X_train_diffx))
    if epoch % 10 == 0:
        print(f"[DiffX] Epoch {epoch:03d} | Loss: {losses_diffx[-1]:.4f} | Acc: {acc:.2f}")

time_diffx = time.time() - start

X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)
X_test_torch  = torch.tensor(X_test, dtype=torch.float32)
y_test_torch  = torch.tensor(y_test_diffx, dtype=torch.long)

torch_model = nn.Sequential(
    nn.Linear(4, 10),
    nn.Tanh(),
    nn.Linear(10, 3)
)

optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

losses_torch = []
start = time.time()

for epoch in range(100):
    optimizer.zero_grad()
    output = torch_model(X_train_torch)
    loss = criterion(output, y_train_torch)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        pred = output.argmax(dim=1)
        acc = (pred == y_train_torch).float().mean().item()
    losses_torch.append(loss.item())

    if epoch % 10 == 0:
        print(f"[PyTorch] Epoch {epoch:03d} | Loss: {loss.item():.4f} | Acc: {acc:.2f}")

time_torch = time.time() - start

plt.plot(losses_diffx, label="DiffX")
plt.plot(losses_torch, label="PyTorch")
plt.title("Loss su IRIS dataset")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

correct_diffx = 0
for x, y_true in zip(X_test_diffx, y_test_diffx):
    y_pred = model_diffx(x)
    y_pred_class = np.argmax([v[0].data for v in y_pred.data])
    correct_diffx += (y_pred_class == y_true)
acc_diffx = correct_diffx / len(y_test_diffx)

with torch.no_grad():
    y_pred_torch = torch_model(X_test_torch).argmax(dim=1)
    acc_torch = (y_pred_torch == y_test_torch).float().mean().item()

print("\n== BENCHMARK FINALE SU IRIS ==")
print(f"DiffX Accuracy:   {acc_diffx:.2f} | Tempo: {time_diffx:.2f}s")
print(f"PyTorch Accuracy: {acc_torch:.2f} | Tempo: {time_torch:.2f}s")
    
