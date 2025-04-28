from core.tensor import Tensor
from core.nn import MLP, mse_loss, cross_entropy
from core.optim import SGD
import matplotlib.pyplot as plt
import json

# Linear
"""# Training data: y = 2x + 3
X = [Tensor([[x]]) for x in [-2.0, -1.0, 0.0, 1.0, 2.0]]
Y = [Tensor([[2 * x + 3]]) for x in [-2.0, -1.0, 0.0, 1.0, 2.0]]

# Model
model = MLP(in_features=1, hidden_layers=[4])
opt = SGD(model.parameters(), lr=0.05)

# Training loop
for epoch in range(100):
    total_loss = 0
    for x,y in zip(X, Y):
        pred = model(x)
        loss_tensor = mse_loss(pred, y)
        loss = loss_tensor

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.data
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss = {total_loss / len(X):.6f}")

# Test finale
for x in X:
    pred = model(x)
    print(f"x = {x.data[0][0].data:.1f}, predicted = {pred.data[0][0].data:.4f}")
"""
# XOR
X = [
    Tensor([[0.0], [0.0]]),
    Tensor([[0.0], [1.0]]),
    Tensor([[1.0], [0.0]]),
    Tensor([[1.0], [1.0]])
]

Y = [
    Tensor([[0.0]]),
    Tensor([[1.0]]),
    Tensor([[1.0]]),
    Tensor([[0.0]]),
]

# MLP con 2 input, 1 hidden con 4 neuroni, 1 output
model = MLP(in_features=2, hidden_layers=[4], out_features=1)
opt = SGD(model.parameters(), lr=0.1)

losses = []

# Training loop
for epoch in range(200):
    total_loss = 0
    for x,y in zip(X, Y):
        pred = model(x)
        loss_tensor = mse_loss(pred, y)
        loss = loss_tensor

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.data

    avg_loss = total_loss / len(X)
    losses.append(avg_loss)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss = {avg_loss:.6f}")

# Test finale
print("\n== XOR Predictions ==")
for x, y in zip(X,Y):
    pred = model(x)
    output = pred.data[0][0].data
    print(f"Input: {[v[0].data for v in x.data]} -> Predicted: {output:.4f} | Target: {y.data[0][0].data:.1f}")

"""
def save_model(model, filename):
    params = [p.data for p in model.parameters()]
    with open(filename, "w") as f:
        json.dump(params, f)

def load_model(model, filename):
    with open(filename, "r") as f:
        values = json.load(f)
    for p, val in zip(model.parameters(), values):
        p.data = val
"""

plt.plot(losses)
plt.title("Loss durante il training")
plt.xlabel("Epoca")
plt.ylabel("Loss")
plt.grid(True)
plt.show()