# DiffX: Autodiff Engine & MLP Framework 🚀

Welcome to **DiffX**, a fully **pure Python** automatic differentiation (autodiff) engine  
DiffX includes:
- A dynamic autodiff engine with computational graphs
- A module to build and train neural networks (MLP)
- Full benchmarks on real dataset (Iris)

This project demonstrates how to build a **machine learning system from scratch**, without external libraries like PyTorch or TensorFlow (except optionally for benchmarking).

---

# 📚 Table of Contents
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Benchmarks](#benchmarks)
- [Usage](#usage)
- [Examples](#examples)
- [Future Roadmap](#future-roadmap)

---

# ✨ Key Features

- **Dynamic autodiff** based on Directed Acyclic Graphs (DAG)
- **Tensor** support for 2D matrix operations
- **Modular MLP**: easily define multilayer models
- **Minimalist SGD Optimizer**
- **Implemented loss functions**:
  - MSELoss
  - CrossEntropyLoss (numerically stabilized)
- **Loss curves visualization** with matplotlib
- **Full benchmark** against PyTorch

---

# 🛠️ Project Architecture

```
DiffX/
├── core/
│   ├── value.py         # Core autodiff node
│   ├── tensor.py        # 2D Tensor built on Value
│   ├── nn.py            # Linear Layer, MLP, Losses
│   ├── optim.py         # Optimizers
├── benchmark_iris.py    # Iris dataset benchmark
├── utils.py  # Function to see in real time building of graph
├── README.md            # This file
```

---

# 📊 Benchmarks

| Dataset | Framework | Final Accuracy | Training Time |
|---------|-----------|----------------|---------------|
| Iris    | DiffX      | 97%             | 15s           |
| Iris    | PyTorch    | 93%             | 0.04s         |

✅ DiffX reaches **the same accuracy** as PyTorch on small datasets!  
🐢 Slower execution due to pure Python dynamic graph building.
*for accuracy I have taken the mean of 5 training

---

# 🚀 Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-user/diffx.git
cd diffx
```

### 2. Install optional dependencies
```bash
pip install matplotlib scikit-learn torch
```

### 3. Run a benchmark
```bash
python benchmark_iris.py
python benchmark_digits.py
```

You will see the training logs and loss curves plotted.


---

# 🧪 Examples

### Create and train an MLP

```python
from core.tensor import Tensor
from core.nn import MLP, mse_loss
from core.optim import SGD

x = Tensor([[1.0], [0.5]])  # 2 inputs
model = MLP(in_features=2, hidden_layers=[4, 4], out_features=1)
opt = SGD(model.parameters(), lr=0.1)

pred = model(x)
loss = mse_loss(pred, Tensor([[0.0]]))

opt.zero_grad()
loss.backward()
opt.step()
```

---

# 🛤️ Future Roadmap

- [ ] Computation graph visualization with React + D3.js
- [ ] Interactive UI for step-by-step training
- [ ] Model saving and loading
- [ ] CNN (Convolutional Neural Networks) support
- [ ] Mini-batch SGD support
- [ ] Google Colab integration for public demos

---

# 📜 License

Open Source - Be inspired, contribute, and improve it with me 🚀

---

> Created with ❤️ by Gabriele

