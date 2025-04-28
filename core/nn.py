import random
import math
from .tensor import Tensor
from .value import Value

# Linear Layer
class Linear:
    def __init__(self, in_features, out_features, label="Linear"):
        self.W = Tensor([
            [random.uniform(-1, 1) for _ in range(in_features)]
            for _ in range(out_features)
        ], label=f"{label}.W")

        self.b = Tensor([
            [0.0] for _ in range(out_features)
        ], label=f"{label}.b")
    
    def __call__(self, x): # forward
        return self.W @ x + self.b
    
    def parameters(self):
        return self.W.parameters() + self.b.parameters()

# MLP ---> seq di Linear + Activation Function
class MLP:
    def __init__(self, in_features, hidden_layers, out_features=1):
           sizes = [in_features] + hidden_layers + [out_features]
           self.layers = [Linear(sizes[i], sizes[i + 1], label=f"L{i}") for i in range(len(sizes) - 1)]

    def __call__(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = out.tanh() # attivazione solo nei hidden
        return out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]  
    
# MSE Loss
def mse_loss(pred, target):
    diff = pred + (-target)
    squared = diff * diff

    values = [v for row in squared.data for v in row]
    total = Value(0.0)
    for v in values:
        total += v

    return total / len(values)  # restituisce un singolo Value

# CrossEntropy Loss
def safe_log(x, eps=1e-12):
    return (x + eps).log()

def cross_entropy(pred, target):
    terms = []
    for i in range(len(pred.data)):
        p = pred.data[i][0]           # Value
        y = target.data[i][0]         # Value
        term = y * safe_log(p) + (Value(1) - y) * safe_log(Value(1) - p)
        terms.append(term)
    return -sum(terms)  # ✅ meno finale = loss positiva
