# Extracted from chapter6_Optimisation.qmd
# Do not edit the source .qmd file directly.

#| label: Chapter 6 libraries
#| message: false
#| echo: false
#| eval: true

import os
import sys
import psutil
from pathlib import Path

import pandas as pd
import seaborn as sns
import numpy as np

import itertools

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset

import optuna

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
plt.style.use('fivethirtyeight')

# Get working directory
cwd = os.getcwd()

# Get parent directory
parent_dir = os.path.dirname(cwd)

# Add parent directory to system path
sys.path.insert(0, parent_dir)

# Import plot function for activation
from src.utils import plot_activation_fn, plot_activation_fn_and_derivative, plot_grid, sample_grid, colorizer



# Define palette
custom_palette = ["#000000", "#0072B2", "#D55E00","#009E73","#CC79A7", "#56B4E9","#E69F00"]
line_styles = ['-', '--', '-.', ':']

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

# Mean Absolute Error (MAE)
def mae(actual, predicted):
    return np.mean(np.abs(predicted - actual))

# Mean Squared Error (MSE)
def mse(actual, predicted):
    return np.mean(np.square(predicted - actual))

# Root Mean Squared Error (RMSE)
def rmse(actual, predicted):
    return np.sqrt(np.mean(np.square(predicted - actual)))

# Mean Absolute Percentage Error (MAPE)
def mape(actual, predicted):
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

# symetrical MAPE (sMAPE)
def smape(actual, predicted):
    return 100/len(actual) * np.sum(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))

# denormalize 
def denormalize(tensor, min_val, max_val):
    return tensor * (max_val - min_val) + min_val

# read tsf function

# Get working directory
cwd = os.getcwd()

# Get parent directory
parent_dir = os.path.dirname(cwd)

# Build path to src directory
src_dir = os.path.join(parent_dir, 'src')

# Add src directory to system path
sys.path.insert(0, src_dir)

# Import datalaoder for tsf file
from tsf_data_loader import convert_tsf_to_dataframe

# ----------------------------------------------------------------------

#| label: Plot of Sigmoid Function and Derivative
#| message: false
#| echo: true
#| eval: true

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)

sigmoid = torch.nn.Sigmoid()

plot_activation_fn_and_derivative(sigmoid, x, xlabel='x', ylabel='Sigmoid(x)', title='Sigmoid Activation Function')

# ----------------------------------------------------------------------

#| label: Plot of Tanh Function and Derivative
#| message: false
#| echo: true
#| eval: true

y_tanh = torch.nn.Tanh()

plot_activation_fn_and_derivative(y_tanh, x, xlabel='x', ylabel='Tanh(x)', title='Tanh Activation Function')

# ----------------------------------------------------------------------

#| label: Plot of ReLU Function and Derivative
#| message: false
#| echo: true
#| eval: true

y_relu = torch.nn.ReLU()

plot_activation_fn_and_derivative(y_relu, x, xlabel='x', ylabel='ReLU(x)', title='ReLU Activation Function')

# ----------------------------------------------------------------------

#| label: Plot of Leaky ReLU Function and Derivative
#| message: false
#| echo: true
#| eval: true

plot_activation_fn_and_derivative(torch.nn.LeakyReLU(negative_slope=0.1), x, xlabel='x', ylabel='Leaky ReLU(x)', title='Leaky ReLU Activation Function', ylim=(-2, 6))

# ----------------------------------------------------------------------

#| label: Some plot of activation functions 
#| message: false
#| echo: true
#| eval: true

# Define range of input values
x = torch.linspace(-10, 10, 1000, requires_grad=True)

# Define activation functions
activation_functions = {
    'ReLU': torch.nn.ReLU(),
    'Leaky ReLU': torch.nn.LeakyReLU(negative_slope=0.1)  # specify negative slope
}

# Plot each activation function and its derivative
for name, fn in activation_functions.items():
    # Compute activation function
    y = fn(x)
    
    # Compute gradients
    grad_y = torch.ones_like(y)
    y.backward(grad_y, retain_graph=True)  # compute gradients
    dy_dx = x.grad.clone()  # clone gradients
    
    # Plot activation function
    plt.figure(figsize=(4, 3))
    plt.plot(x.detach().numpy(), y.detach().numpy(), label=f'{name} function')
    
    # Plot derivative of activation function
    plt.plot(x.detach().numpy(), dy_dx.detach().numpy(), linestyle='--', label=f'{name} derivative')
    
    # Set range of x and y axes
    plt.xlim([-10, 10])
    plt.ylim([-1.5, 1.5])
    
    # Add a horizontal line at y=0
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
    
    # Set labels
    plt.xlabel('x')
    plt.ylabel(f'{name}(x)')
    plt.title(f'{name} Activation Function and its Derivative')
    plt.legend()
    
    # Show plot
    plt.show()
    
    # Clear gradients for next computation
    x.grad.zero_()

# ----------------------------------------------------------------------

#| label: Plot of Swish Function and Derivative
#| message: false
#| echo: true
#| eval: true

class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

swish_fn = Swish(beta=1.0)

plot_activation_fn_and_derivative(swish_fn, x, xlabel='x', ylabel='Swish(x)', title='Swish Activation Function')

# ----------------------------------------------------------------------

#| label: Data preparation for activation function comparison
#| message: false
#| echo: true
#| eval: true

np.random.seed(42)

data = pd.read_csv(CFG.data_folder / 'passengers.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

train_size = int(len(data) * 0.6)
val_size = int(len(data) * 0.2)
train, val, test = data[:train_size], data[train_size:train_size + val_size], data[train_size + val_size:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
val_scaled = scaler.transform(val)
test_scaled = scaler.transform(test)

def create_time_windows(data, window_size):
    inputs = []
    targets = []
    for i in range(len(data) - window_size):
        inputs.append(data[i:i+window_size])
        targets.append(data[i+window_size])
    return np.array(inputs), np.array(targets)

window_size = 12

train_inputs, train_targets = create_time_windows(train_scaled, window_size)
val_inputs, val_targets = create_time_windows(val_scaled, window_size)
test_inputs, test_targets = create_time_windows(test_scaled, window_size)

x_train = torch.FloatTensor(train_inputs)
y_train = torch.FloatTensor(train_targets)
x_val = torch.FloatTensor(val_inputs)
y_val = torch.FloatTensor(val_targets)
x_test = torch.FloatTensor(test_inputs)
y_test = torch.FloatTensor(test_targets)

# ----------------------------------------------------------------------

#| label: FFN and data module for activation function comparison
#| message: false
#| echo: true
#| eval: true

class TimeSeriesDataModule(L.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, batch_size=64):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = TensorDataset(*self.train_data)
        self.val_dataset = TensorDataset(*self.val_data)
        self.test_dataset = TensorDataset(*self.test_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

class ffnetwork(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, activation_func, output_dim=1, learning_rate=0.0001):
        super(ffnetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = activation_func
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = x.squeeze(-1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

# ----------------------------------------------------------------------

#| label: Hyperparameters and activation function dictionary
#| message: false
#| echo: true
#| eval: true

input_dim = 12  
hidden_dim = 50 
output_dim = 1
learning_rate = 0.0001

batch_size = 64
data_module = TimeSeriesDataModule(train_data=(x_train, y_train),
                                   val_data=(x_val, y_val),
                                   test_data=(x_test, y_test),
                                   batch_size=batch_size)

activations = {
    'ReLU': nn.ReLU(),
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'LeakyReLU': nn.LeakyReLU(0.01),
    'Swish': swish()
}

# ----------------------------------------------------------------------

#| label: Model training with multiple activation functions 
#| message: false
#| echo: true
#| eval: true

results = {}
for name, activation_func in activations.items():
    print(f"Training with {name} activation...")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = ffnetwork(input_dim, hidden_dim, activation_func, output_dim, learning_rate)
    
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=False)

    trainer = L.Trainer(max_epochs=2000, callbacks=[early_stop_callback])
    trainer.fit(model, data_module)

    model.eval()
    last_window = x_val[-1].view(1, -1)
    recursive_preds = []

    for _ in range(len(x_test)):
        with torch.no_grad():
            pred = model(last_window)
        recursive_preds.append(pred.item())
        last_window = torch.cat((last_window[:, 1:], pred.view(1, 1)), dim=1)

    unscaled_predictions = scaler.inverse_transform(np.array(recursive_preds).reshape(-1, 1))
    results[name] = unscaled_predictions

    del model
    torch.cuda.empty_cache()

# ----------------------------------------------------------------------

#| label: Plot for each activation function 
#| message: false
#| echo: true
#| eval: true

plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
for idx, (name, preds) in enumerate(results.items()):
    test_index = data.index[-len(preds):]
    plt.plot(test_index, preds, label=f'{name}', color=custom_palette[idx + 1], linewidth=2.0)

plt.plot(data.index, data['passengers'], label='Actual', color=custom_palette[0], linewidth=2.0)
plt.title('Airline Passengers - Predictions with Different Activation Functions')
plt.legend()
plt.show()

# ----------------------------------------------------------------------

#| label: Accuracy table for each activation function 
#| message: false
#| echo: false
#| eval: true

unscaled_actuals = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

errors = {'Activation': [], 'MAE': [], 'MSE': [], 'RMSE': [], 'MAPE': [], 'sMAPE': []}

for name, preds in results.items():
    mae_val = mean_absolute_error(unscaled_actuals, preds)
    mse_val = mean_squared_error(unscaled_actuals, preds)
    rmse_val = rmse(unscaled_actuals, preds)
    mape_val = mape(unscaled_actuals, preds)
    smape_val = smape(unscaled_actuals, preds)

    errors['Activation'].append(name)
    errors['MAE'].append(mae_val)
    errors['MSE'].append(mse_val)
    errors['RMSE'].append(rmse_val)
    errors['MAPE'].append(mape_val)
    errors['sMAPE'].append(smape_val)

fcst_err_df = pd.DataFrame(errors)
print(fcst_err_df)

# ----------------------------------------------------------------------

#| label: FFN setup for multi-layer testing
#| message: false
#| echo: true
#| eval: true

class ffnetwork(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers=1, output_dim=1, learning_rate=0.0001, activation_func=nn.ReLU()):
        super(ffnetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.activation = activation_func
        self.learning_rate = learning_rate

    def forward(self, x):
        x = x.squeeze(-1)
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# ----------------------------------------------------------------------

#| label: Testing multiple layers with FNN
#| message: false
#| echo: true
#| eval: true

results = {}
hidden_layer_configs = range(1, 7)

for num_layers in hidden_layer_configs:
    print(f"Training model with {num_layers} hidden layers...")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = ffnetwork(input_dim, hidden_dim, num_layers=num_layers, output_dim=output_dim, learning_rate=learning_rate)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=False)
    trainer = L.Trainer(max_epochs=2000, callbacks=[early_stop_callback])
    trainer.fit(model, data_module)

    model.eval()
    last_window = x_val[-1].view(1, -1)
    recursive_preds = []

    for _ in range(len(x_test)):
        with torch.no_grad():
            pred = model(last_window)
        recursive_preds.append(pred.item())
        last_window = torch.cat((last_window[:, 1:], pred.view(1, 1)), dim=1)

    unscaled_predictions = scaler.inverse_transform(np.array(recursive_preds).reshape(-1, 1))
    results[f'{num_layers} layers'] = unscaled_predictions

    del model
    torch.cuda.empty_cache()

# ----------------------------------------------------------------------

#| label: Impact of multiple layers on modelling
#| message: false
#| echo: true
#| eval: true

plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
for idx, (name, preds) in enumerate(results.items()):
    test_index = data.index[-len(preds):]
    plt.plot(test_index, preds, label=f'{name}', color=custom_palette[idx + 1], linewidth=2.0)

plt.plot(data.index, data['passengers'], label='Actual', color=custom_palette[0], linewidth=2.0)
plt.title('Airline Passengers - Predictions with Different Numbers of Hidden Layers')
plt.legend()
plt.show()

# ----------------------------------------------------------------------

#| label: Impact of number of hidden layers on accuracy 
#| message: false
#| echo: false
#| eval: true


errors = {'Model': [], 'MAE': [], 'MSE': [], 'RMSE': [], 'MAPE': [], 'sMAPE': []}

for name, preds in results.items():
    mae_val = mean_absolute_error(unscaled_actuals, preds)
    mse_val = mean_squared_error(unscaled_actuals, preds)
    rmse_val = rmse(unscaled_actuals, preds)
    mape_val = mape(unscaled_actuals, preds)
    smape_val = smape(unscaled_actuals, preds)

    errors['Model'].append(name)
    errors['MAE'].append(mae_val)
    errors['MSE'].append(mse_val)
    errors['RMSE'].append(rmse_val)
    errors['MAPE'].append(mape_val)
    errors['sMAPE'].append(smape_val)

fcst_err_df = pd.DataFrame(errors)
print(fcst_err_df)

# ----------------------------------------------------------------------

#| label: Loss function comparison
#| message: false
#| echo: true
#| eval: false

# MSE loss (default throughout this chapter)
loss = nn.functional.mse_loss(y_hat, y)

# MAE loss
loss = nn.functional.l1_loss(y_hat, y)

# Huber loss, combines MSE and MAE, with a threshold delta
loss = nn.functional.huber_loss(y_hat, y, delta=1.0)

# ----------------------------------------------------------------------

#| label: Function and observed data
#| message: false
#| echo: true
#| eval: true

def f(x):
    return x * torch.cos(np.pi * x) + 2
def fd(x):
    return f(x) + 0.3 * torch.cos(6 * np.pi * x)

x = torch.arange(0.5, 1.5, 0.01)

plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
plt.plot(x, f(x), label='f (generalisation)', color=custom_palette[1])  
plt.plot(x, fd(x), label='fd (training)', color=custom_palette[2])  
plt.xlabel('x')
plt.ylabel('Value')
plt.legend()
plt.show()

# ----------------------------------------------------------------------

#| label:  Local minima
#| message: false
#| echo: true
#| eval: true
def fd(x):
    return f(x) + 0.3 * torch.cos(4 * np.pi * x)

plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
plt.plot(x, fd(x), label='fd (training)', color=custom_palette[2])  
plt.xlabel('x')
plt.ylabel('Value')
plt.legend()
plt.show()

# ----------------------------------------------------------------------

#| label:  Updated FFN for comparing optimisers
#| message: false
#| echo: true
#| eval: true

from lightning.pytorch.loggers import CSVLogger


class ffnetwork(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers=1, output_dim=1, learning_rate=0.0001, dropout_rate=0.00, activation_func=nn.ReLU(), optimizer_name='Adam'):
        super(ffnetwork, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        self.dropout = nn.Dropout(dropout_rate)
        
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.activation = activation_func
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name

    def forward(self, x):
        x = x.squeeze(-1)
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)  
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)  
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizers = {
            'SGD': torch.optim.SGD(self.parameters(), lr=self.learning_rate),
            'Adam': torch.optim.Adam(self.parameters(), lr=self.learning_rate),
            'Adadelta': torch.optim.Adadelta(self.parameters(), lr=self.learning_rate),
            'RMSprop': torch.optim.RMSprop(self.parameters(), lr=self.learning_rate),
            'Adagrad': torch.optim.Adagrad(self.parameters(), lr=self.learning_rate),
        }
        if self.optimizer_name in optimizers:
            return optimizers[self.optimizer_name]
        else:
            raise ValueError(f"Optimizer '{self.optimizer_name}' not recognized.")

# ----------------------------------------------------------------------

#| label:  Comparison of optimiser performance
#| message: false
#| echo: true
#| eval: true

# List of optimiser algorithms
optimizers = ['SGD', 'Adam', 'Adadelta', 'RMSprop', 'Adagrad']  
results = {}
for optimizer_name in optimizers:
    print(f"Training with {optimizer_name} optimizer...")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = ffnetwork(input_dim, hidden_dim, num_layers=2, output_dim=output_dim, learning_rate=0.001, optimizer_name=optimizer_name)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=False,
        mode='min'
    )
    # Dictionary for logs to be saved
    log_dir = f'lightning_logs/{optimizer_name}' 
    logger = CSVLogger("lightning_logs", name=optimizer_name)
    # Early stopping 
    trainer = L.Trainer(
        max_epochs=2000,
        default_root_dir=log_dir,
        callbacks=[early_stop_callback],
        logger=logger # CSV logging of training 
    )
    # Train model
    trainer.fit(model, data_module)
    # Evaluate model with test data and store predictions for plotting
    model.eval()
    last_window = x_val[-1].view(1, -1)
    recursive_preds = []

    for _ in range(len(x_test)):
        with torch.no_grad():
            pred = model(last_window)
        recursive_preds.append(pred.item())
        last_window = torch.cat((last_window[:, 1:], pred.view(1, 1)), dim=1)
    unscaled_predictions = scaler.inverse_transform(np.array(recursive_preds).reshape(-1, 1))
    results[optimizer_name] = unscaled_predictions

    del model  
    torch.cuda.empty_cache()

# ----------------------------------------------------------------------

#| label:  Plot of optimiser impact on forecasting
#| message: false
#| echo: false
#| eval: true

# Plotting results of different optimisers
line_styles = ['-', '--', '-.', ':']
style_cycle = itertools.cycle(line_styles)
colour_cycle = itertools.cycle(custom_palette)

for name, preds in results.items():
    test_index = data.index[-len(preds):]
    plt.plot(
        test_index,
        preds,
        label=f'{name}',
        color=next(colour_cycle),
        linestyle=next(style_cycle),
        linewidth=4
    )
plt.plot(
    data.index,
    data['passengers'],
    label='Actual',
    color='black',
    linewidth=2.0
)
plt.title('Airline Passengers - Predictions with Different Optimizers')
plt.legend()
plt.show()

# ----------------------------------------------------------------------

#| label: Loss epochs
#| message: false
#| echo: false
#| eval: true

optimizers = ['SGD', 'Adam', 'Adadelta', 'RMSprop', 'Adagrad']

base_dir = Path(r"C:\Users\Graeme\Documents\github\tsfwpt\das_buch\lightning_logs")

plt.figure(figsize=(CFG.img_dim1 , CFG.img_dim2))

for opt in optimizers:
    metrics_path = base_dir / opt / "version_0" / "metrics.csv"
    df = pd.read_csv(metrics_path)

    # keep rows where validation loss exists
    val_df = df[df["val_loss"].notna()].copy()

    plt.plot(
        val_df["epoch"],
        val_df["val_loss"],
        label=opt,
        linewidth=2
    )

plt.title("Validation Loss by Optimizer")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ----------------------------------------------------------------------

#| label: Impact of optimisers on accuracy 
#| message: false
#| echo: false
#| eval: true


errors = {'Model': [], 'MAE': [], 'MSE': [], 'RMSE': [], 'MAPE': [], 'sMAPE': []}

for name, preds in results.items():
    mae_val = mean_absolute_error(unscaled_actuals, preds)
    mse_val = mean_squared_error(unscaled_actuals, preds)
    rmse_val = rmse(unscaled_actuals, preds)
    mape_val = mape(unscaled_actuals, preds)
    smape_val = smape(unscaled_actuals, preds)

    errors['Model'].append(name)
    errors['MAE'].append(mae_val)
    errors['MSE'].append(mse_val)
    errors['RMSE'].append(rmse_val)
    errors['MAPE'].append(mape_val)
    errors['sMAPE'].append(smape_val)

fcst_err_df = pd.DataFrame(errors)
print(fcst_err_df)

# ----------------------------------------------------------------------

#| label: Learning rate scheduler example
#| message: false
#| echo: true
#| eval: false

def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'monitor': 'val_loss'
        }
    }

# ----------------------------------------------------------------------

#| label: Learning rate range test
#| message: false
#| echo: true
#| eval: false

from lightning.pytorch.tuner import Tuner

trainer = L.Trainer(max_epochs=100)
tuner = Tuner(trainer)

# runs LR range test and updates model.learning_rate
lr_finder = tuner.lr_find(model, datamodule=data_module)

# plot loss curve
fig = lr_finder.plot(suggest=True)
fig.show()

# suggested LR
print(f"Suggested LR: {lr_finder.suggestion()}")

# ----------------------------------------------------------------------

#| label: FFN with dropout added
#| message: false
#| echo: true
#| eval: false

class ffnetwork(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers=1, output_dim=1, learning_rate=0.0001, dropout_rate=0.5, activation_func=nn.ReLU()):
        super(ffnetwork, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        self.dropout = nn.Dropout(dropout_rate)
        
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.activation = activation_func
        self.learning_rate = learning_rate

    def forward(self, x):
        x = x.squeeze(-1)
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
            x = self.dropout(x)  # Apply dropout after each hidden layer
        x = self.layers[-1](x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# ----------------------------------------------------------------------

#| label: Weight decay example
#| message: false
#| echo: true
#| eval: false

# Adam with weight decay (L2 regularisation)
optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)

# AdamW separates weight decay from gradient updates, generally preferred
optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)

# ----------------------------------------------------------------------

#| label: Optuna full example  
#| message: false
#| echo: true
#| eval: false

def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3)
    hidden_dim = trial.suggest_categorical('hidden_dim', [6, 10, 20, 40])
    num_layers = trial.suggest_int('num_layers', 1, 6)

    model = ffnetwork(input_dim, hidden_dim, num_layers=num_layers, output_dim=output_dim, 
                      learning_rate=learning_rate, dropout_rate=dropout_rate)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=False, mode='min')
    trainer = L.Trainer(max_epochs=100, callbacks=[early_stop_callback], 
                        logger=False, enable_checkpointing=False)

    trainer.fit(model, data_module)

    model.eval()

    last_window = x_val[-1].view(1, -1)
    recursive_preds = []
    for _ in range(len(x_test)):
        with torch.no_grad():
            pred = model(last_window)
        recursive_preds.append(pred.item())
        last_window = torch.cat((last_window[:, 1:], pred.view(1, 1)), dim=1)

    recursive_preds = np.array(recursive_preds).reshape(-1, 1)
    y_test_np = y_test.numpy().reshape(-1, 1)
    mse_val = np.mean((recursive_preds - y_test_np) ** 2)

    trial.set_user_attr("MSE", mse_val)
    
    return mse_val

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Extract best results
print(f"Best hyperparameters: {study.best_params}")
print(f"Best MSE: {study.best_value:.4f}")