# =============================================================================
# Chapter 4: PyTorch Fundamentals
# =============================================================================
# Code companion for Chapter 4
# Covers: tensor construction, operations, memory management, computational
#         graphs, autograd, backpropagation, simple neural networks
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L

import matplotlib.pyplot as plt

# =============================================================================
# Setup: palette, rcParams, CFG
# =============================================================================

custom_palette = ["#000000", "#0072B2", "#D55E00", "#009E73",
                  "#CC79A7", "#56B4E9", "#E69F00"]
line_styles = ['-', '--', '-.', ':']

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Source Sans Pro', 'Arial']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titlesize'] = 18

class CFG:
    data_folder = Path.cwd() / "Data"
    img_dim1 = 12
    img_dim2 = 6
    fontsize = 18

plt.rcParams.update({'figure.figsize': (CFG.img_dim1, CFG.img_dim2)})


# =============================================================================
# 4.1  Basic Tensor Construction
# =============================================================================

scalar = torch.tensor(42, dtype=torch.float32)

vector = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)

matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]], dtype=torch.float32)

tensor_3d = torch.tensor([[[1, 2], [3, 4]],
                          [[5, 6], [7, 8]],
                          [[9, 10], [11, 12]]], dtype=torch.float32)

print(scalar)
print(vector)
print(matrix)
print(tensor_3d)


# =============================================================================
# 4.2  Tensor from Random Normal Distribution
# =============================================================================

torch.manual_seed(42)

randn_tensor = torch.randn(3, 2)

print(f'data type \n', type(randn_tensor))
print(f'within tensor data type \n', randn_tensor.dtype)
print(f'tensor values \n', randn_tensor)


# =============================================================================
# 4.3  Tensor Operations
# =============================================================================

print(f'tensor x tensor \n', randn_tensor.mul(randn_tensor), f'\n')
print(f'matrix multiplication \n', randn_tensor.matmul(randn_tensor.T))
print(f'add 1 \n', randn_tensor.add_(1))

# Zero and concatenation
zero_tensor = torch.randn(3, 2).zero_()
concat_tensor = torch.cat([zero_tensor, randn_tensor], dim=1)

print(f'zero tensor \n', zero_tensor)
print(f'concatonated tensors \n', concat_tensor)


# =============================================================================
# 4.4  CSV to Tensor
# =============================================================================

# Load dataset
data = pd.read_csv(CFG.data_folder / 'passengers.csv')

# Add a count column
data['count'] = np.arange(1, len(data) + 1)
tensor_1_col = torch.tensor(data['passengers'].values, dtype=torch.float32)
tensor_2_col = torch.tensor(data[['passengers', 'count']].values, dtype=torch.float32)

# print(f'simple univariate tensor', tensor_1_col)
# print(f'simple multivariate tensor', tensor_2_col)


# =============================================================================
# 4.5  Memory Efficiency
# =============================================================================

torch.manual_seed(42)

A = torch.randn(3, 2)
B = torch.ones(3, 2)
print(f'Orginal location in memory', id(B))

B = A + B
print(f'New location in memory', id(B))

# Inplace operation preserves memory address
A = torch.randn(3, 2)
B = torch.ones(3, 2)
print('Location in before operation', id(B))

B[:] = A + B
print('Location in after operation', id(B))


# =============================================================================
# 4.6  Computational Graphs — Function and Derivative
# =============================================================================

# --- Figure 4.4: Function and derivative ---
def f(x):
    return x ** 2

def df(x):
    return 2 * x

x = np.linspace(-10, 10, 100)

plt.figure(figsize=(10, 6))
plt.plot(x, f(x), label='f(x) = x^2', color=custom_palette[1])
plt.plot(x, df(x), label="f'(x) = 2x", color=custom_palette[6], linestyle='--')

for point in [-7, 0, 7]:
    plt.scatter(point, f(point), color=custom_palette[2])
    plt.text(point, f(point) + 10, f"f'({point}) = {df(point)}", ha='center')

plt.title('Function and its Derivative')
plt.xlabel('x')
plt.ylabel('f(x) and f\'(x)')
plt.legend()
plt.grid(True)
plt.show()


# =============================================================================
# 4.7  Autograd — Gradient Calculation
# =============================================================================

# Define operations of forward pass
def forward_pass(x):
    # Operation 1 - logarithm
    log_x = torch.log(x)
    # Operation 2 - sine
    sin_log_x = torch.sin(log_x)
    return sin_log_x

# Instantiate x tensor, setting requires_grad=True to track computations
x = torch.tensor([2.0], requires_grad=True)

# Forward pass to compute output
y = forward_pass(x)

# Backward pass to compute gradient of y with respect to x
y.backward()

# Gradient stored in x.grad
print(f"Gradient of y with respect to x is: {x.grad}")


# =============================================================================
# 4.8  Single NN Layer
# =============================================================================

# Definitions - input data, weights, and 'true output' for loss calculation
input_tensor = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
weights = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True)
true_output = torch.tensor([[0.7, 1.5]])

# Forward pass - Matrix multiplication (input with weights), using ReLU activation
layer_output = torch.mm(input_tensor, weights.t()).relu()

# Calculate loss - MSE
predicted_output = layer_output
loss = (true_output - predicted_output).pow(2).mean()

# Backward pass with autograd to compute gradients
loss.backward()

# dict of tensors and gradients
nn_basic_results = {
    "Input Tensor": input_tensor,
    "Weights": weights,
    "Layer Output after ReLU": layer_output,
    "Loss (MSE)": loss,
    "Gradient with respect to Input Tensor": input_tensor.grad,
    "Gradient with respect to Weights": weights.grad
}
for k, v in nn_basic_results.items():
    print(f"{k}: {v}")


# =============================================================================
# 4.9  Pure PyTorch NN
# =============================================================================

# Define neural network
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.weights = torch.nn.Parameter(torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))

    def forward(self, x):
        # Forward pass through layer and ReLU activation
        return torch.mm(x, self.weights.t()).relu()

# Instantiate neural network
model = SimpleNN()

# Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Define tensors for input and output
input_tensor = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
true_output = torch.tensor([[0.7, 1.5]])

# Wrap tensors in a DataLoader
train_loader = DataLoader(TensorDataset(input_tensor, true_output), batch_size=1)

# Training loop with loss tracking
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        predicted_output = model(inputs)
        loss = torch.mean((targets - predicted_output).pow(2))
        loss.backward()
        optimizer.step()
    losses.append(loss.item())

print(f"Final loss after {num_epochs} epochs: {losses[-1]:.6f}")


# =============================================================================
# 4.10  Lightning NN
# =============================================================================

class SimpleNNLightning(L.LightningModule):
    def __init__(self):
        super(SimpleNNLightning, self).__init__()
        self.weights = torch.nn.Parameter(torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))

    def forward(self, x):
        return torch.mm(x, self.weights.t()).relu()

    def training_step(self, batch, batch_idx):
        x, true_output = batch
        predicted_output = self(x)
        loss = torch.mean((true_output - predicted_output).pow(2))
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

model_lightning = SimpleNNLightning()

input_tensor = torch.tensor([[1.0, 2.0, 3.0]])
true_output = torch.tensor([[0.7, 1.5]])

train_loader = DataLoader(TensorDataset(input_tensor, true_output), batch_size=1)

# Disable default logging to avoid creating lightning_logs/ folder
trainer = L.Trainer(max_epochs=200, logger=False, enable_checkpointing=False)
trainer.fit(model_lightning, train_loader)


# =============================================================================
# 4.11  Training Loss Plot
# =============================================================================

# --- Figure: Training loss over epochs (Pure PyTorch NN) ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), losses, color=custom_palette[1], linewidth=2)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('MSE Loss', fontsize=14)
plt.title('Training Loss: Pure PyTorch SimpleNN', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()