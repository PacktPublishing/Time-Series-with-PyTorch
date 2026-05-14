# Extracted from chapter4_Pytorch_fundementals.qmd
# Do not edit the source .qmd file directly.

#| label: Chapter 3 libraries
#| message: false
#| echo: false
#| eval: true
from pathlib import Path
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')


# Define palette
custom_palette = ["#000000", "#0072B2", "#D55E00","#009E73","#CC79A7", "#56B4E9","#E69F00"]


plt.rcParams['figure.figsize'] = (8, 4)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Source Sans Pro', 'Arial']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titlesize'] = 18

# general settings
class CFG:
    data_folder = Path.cwd().parent / "data"
    img_dim1 = 12
    img_dim2 = 6
    fontsize = 18
    
    
# adjust the parameters for displayed figures    
plt.rcParams.update({'figure.figsize': (CFG.img_dim1,CFG.img_dim2)})

# ----------------------------------------------------------------------

#| label: Basic tensor construction
#| message: false
#| echo: true
#| eval: true
import torch

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

# ----------------------------------------------------------------------

#| label: Tensor with data from random normal distribution
#| message: false
#| echo: true
#| eval: true

torch.manual_seed(42)

randn_tensor = torch.randn(3,2)

print(f'data type \n', type(randn_tensor))
print(f'within tensor data type \n',randn_tensor.dtype)
print(f'tensor values \n', randn_tensor)

# ----------------------------------------------------------------------

#| label: Tensor Opperations
#| message: false
#| echo: true
#| eval: true

print(f'tensor x tensor \n', randn_tensor.mul(randn_tensor), f'\n')
print(f'matrix multiplication \n', randn_tensor.matmul(randn_tensor.T))
print(f'add 1 \n', randn_tensor.add_(1))

# ----------------------------------------------------------------------

#| label: Zero and concat
#| message: false
#| echo: true
#| eval: true

zero_tensor = torch.randn(3,2).zero_()
concat_tensor = torch.cat([zero_tensor, randn_tensor], dim = 1)

print(f'zero tensor \n', zero_tensor)
print(f'concatonated tensors \n', concat_tensor)

# ----------------------------------------------------------------------

#| label: CSV to tensor
#| message: false
#| echo: true
#| eval: true

# Load dataset
data = pd.read_csv(CFG.data_folder / 'passengers.csv')

# Add a count column 
data['count'] = np.arange(1, len(data) + 1)
tensor_1_col = torch.tensor(data['passengers'].values, dtype=torch.float32)
tensor_2_col = torch.tensor(data[['passengers', 'count']].values, dtype=torch.float32)

#print(f'simple univariate tensor', tensor_1_col)
#print(f'simple multivariate tensor', tensor_2_col)

# ----------------------------------------------------------------------

#| label: Tensors in memory
#| message: false
#| echo: true
#| eval: true

torch.manual_seed(42)

A = torch.randn(3,2)
B = torch.ones(3,2)
print(f'Orginal location in memory', id(B))

B = A + B
print(f'New location in memory', id(B))

# ----------------------------------------------------------------------

#| label: Inplace tensor opperations
#| message: false
#| echo: true
#| eval: true

A = torch.randn(3,2)
B = torch.ones(3,2)
print('Location in before operation', id(B))

B[:] = A + B
print('Location in after operation', id(B))

# ----------------------------------------------------------------------

#| label: Function and it's derivative
#| message: false
#| fig-cap: "Figure 4.4: Function and derivative"
#| echo: false
#| eval: true

# Define quadratic function and it's derivative
def f(x):
    return x ** 2

def df(x):
    return 2 * x

# Generate values of x
x = np.linspace(-10, 10, 100)

# Plot derivative
plt.figure(figsize=(10, 6))
plt.plot(x, f(x), label='f(x) = x^2', color='blue')
plt.plot(x, df(x), label="f'(x) = 2x", color='orange', linestyle='--')

# Highlight gradient 
for point in [-7, 0, 7]:
    plt.scatter(point, f(point), color='red')
    plt.text(point, f(point) + 10, f"f'({point}) = {df(point)}", ha='center')

# Annotate the derivative plot
plt.title('Function and its Derivative')
plt.xlabel('x')
plt.ylabel('f(x) and f\'(x)')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------------------------------------

#| label: Simple example of gradient calculation
#| message: false
#| echo: true
#| eval: true

# Define opperations of forward pass 
def forward_pass(x):
    # Operation 1 - logarithm
    log_x = torch.log(x)
    # Operation 2 - sine
    sin_log_x = torch.sin(log_x)
    return sin_log_x

# Instantiate x tensor, setting requires_grad=True in order to track computations
x = torch.tensor([2.0], requires_grad=True)

# Forward pass to compute output
y = forward_pass(x)

# Backward pass, to compute gradient of y with respect to x
y.backward()

# Gradient stored in x.grad
print(f"Gradient of y with respect to x is: {x.grad}")

# ----------------------------------------------------------------------

#| label: Calculating a single NN layer
#| message: false
#| echo: true
#| eval: false

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
# print result
nn_basic_results

# ----------------------------------------------------------------------

#| label: A simple Pytorch model and training loop 
#| message: false
#| echo: true
#| eval: false

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

# Training loop
for epoch in range(1):  
    for batch in train_loader:
        # Separate data into inputs and outputs
        inputs, targets = batch
        # Zero gradients
        optimizer.zero_grad()
        # Forward pass
        predicted_output = model(inputs)
        # Compute loss
        loss = torch.mean((targets - predicted_output).pow(2))
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()

# ----------------------------------------------------------------------

#| label: A simple Pytorch model and training with Lightening
#| message: false
#| echo: true
#| eval: false

import lightning as L

class SimpleNN(L.LightningModule):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.weights = torch.nn.Parameter(torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))

    def forward(self, x):
        # Forward pass 
        return torch.mm(x, self.weights.t()).relu()

    def training_step(self, batch, batch_idx):
        # Forward pass with loss calculation
        x, true_output = batch
        predicted_output = self(x)
        loss = torch.mean((true_output - predicted_output).pow(2))
        return loss

    def configure_optimizers(self):
        # Optimization with SGD
        return torch.optim.SGD(self.parameters(), lr=0.01)

# Instantiate neural network
model = SimpleNN()

# Define input and output tensors 
input_tensor = torch.tensor([[1.0, 2.0, 3.0]])
true_output = torch.tensor([[0.7, 1.5]])

# Wrap data in a DataLoader
train_loader = DataLoader(TensorDataset(input_tensor, true_output), batch_size=1)

# Define trainer
trainer = L.Trainer(max_epochs=1)

# Fit model
trainer.fit(model, train_loader)