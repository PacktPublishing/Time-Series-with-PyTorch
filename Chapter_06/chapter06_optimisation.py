# =============================================================================
# Chapter 6: Optimisation
# =============================================================================
# Code companion for Chapter 6
# Covers: activation functions (sigmoid, tanh, ReLU, leaky ReLU, swish),
#         hidden layer experiments, loss functions, optimiser comparison,
#         hyperparameter tuning with Optuna, dropout, weight decay
# =============================================================================

import os
import sys
import platform
import numpy as np
import pandas as pd
import itertools
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

import optuna

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    data_folder = Path.cwd()  / "Data" # / "Time-Series-with-PyTorch"
    img_dim1 = 12
    img_dim2 = 6
    fontsize = 18

plt.rcParams.update({'figure.figsize': (CFG.img_dim1, CFG.img_dim2)})


# =============================================================================
# Helper: error metrics
# =============================================================================

def mae(actual, predicted):
    return np.mean(np.abs(predicted - actual))

def mse(actual, predicted):
    return np.mean(np.square(predicted - actual))

def rmse(actual, predicted):
    return np.sqrt(np.mean(np.square(predicted - actual)))

def mape(actual, predicted):
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def smape(actual, predicted):
    return 100 / len(actual) * np.sum(
        2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))


# =============================================================================
# Helper: plot activation function and its derivative
# =============================================================================

def plot_activation_fn_and_derivative(fn, x, xlabel='x', ylabel='y',
                                       title='Activation Function', linewidth=2.5,
                                       figsize=(CFG.img_dim1, CFG.img_dim2), ylim=None):
    plt.figure(figsize=figsize)
    x = x.detach().requires_grad_(True)
    y = fn(x)
    y.backward(torch.ones_like(x))
    dy_dx = x.grad
    plt.plot(x.detach().numpy(), y.detach().numpy(), linewidth=linewidth, label='Function')
    plt.plot(x.detach().numpy(), dy_dx.detach().numpy(), linewidth=linewidth,
             linestyle='--', label='Derivative')
    plt.axvline(0, color='black', linewidth=1.5, linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.show()


# =============================================================================
# 6.1  Activation Functions — Visualisation
# =============================================================================

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)

# --- Sigmoid ---
sigmoid = torch.nn.Sigmoid()
plot_activation_fn_and_derivative(sigmoid, x, xlabel='x', ylabel='Sigmoid(x)',
                                   title='Sigmoid Activation Function')

# --- Tanh ---
y_tanh = torch.nn.Tanh()
plot_activation_fn_and_derivative(y_tanh, x, xlabel='x', ylabel='Tanh(x)',
                                   title='Tanh Activation Function')

# --- ReLU ---
y_relu = torch.nn.ReLU()
plot_activation_fn_and_derivative(y_relu, x, xlabel='x', ylabel='ReLU(x)',
                                   title='ReLU Activation Function')

# --- Leaky ReLU ---
plot_activation_fn_and_derivative(torch.nn.LeakyReLU(negative_slope=0.1), x,
                                   xlabel='x', ylabel='Leaky ReLU(x)',
                                   title='Leaky ReLU Activation Function', ylim=(-2, 6))


# --- ReLU vs Leaky ReLU side-by-side ---
x = torch.linspace(-10, 10, 1000, requires_grad=True)

activation_functions = {
    'ReLU': torch.nn.ReLU(),
    'Leaky ReLU': torch.nn.LeakyReLU(negative_slope=0.1)
}

for name, fn in activation_functions.items():
    y = fn(x)
    grad_y = torch.ones_like(y)
    y.backward(grad_y, retain_graph=True)
    dy_dx = x.grad.clone()

    plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
    plt.plot(x.detach().numpy(), y.detach().numpy(), label=f'{name} function')
    plt.plot(x.detach().numpy(), dy_dx.detach().numpy(), linestyle='--',
             label=f'{name} derivative')
    plt.xlim([-10, 10])
    plt.ylim([-1.5, 1.5])
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
    plt.xlabel('x')
    plt.ylabel(f'{name}(x)')
    plt.title(f'{name} Activation Function and its Derivative')
    plt.legend()
    plt.show()

x.grad.zero_()


# --- Swish (trainable) ---
class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

swish_fn = Swish(beta=1.0)
plot_activation_fn_and_derivative(swish_fn, x, xlabel='x', ylabel='Swish(x)',
                                   title='Swish Activation Function')


# =============================================================================
# 6.2  Activation Function Comparison on Airline Passengers
# =============================================================================

np.random.seed(42)

data = pd.read_csv(CFG.data_folder / 'passengers.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

train_size = int(len(data) * 0.6)
val_size = int(len(data) * 0.2)
train = data[:train_size]
val = data[train_size:train_size + val_size]
test = data[train_size + val_size:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
val_scaled = scaler.transform(val)
test_scaled = scaler.transform(test)

def create_time_windows(data, window_size):
    inputs, targets = [], []
    for i in range(len(data) - window_size):
        inputs.append(data[i:i + window_size])
        targets.append(data[i + window_size])
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

# Determine num_workers for DataLoader
_num_workers = 0 if platform.system() == 'Windows' else min(os.cpu_count() or 1, 4)


# --- Lightning data module ---
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                         shuffle=False, num_workers=_num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                         shuffle=False, num_workers=_num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                         shuffle=False, num_workers=_num_workers)


# --- FFN with configurable activation ---
class ffnetwork_act(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, activation_func, output_dim=1,
                 learning_rate=0.0001):
        super().__init__()
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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


# --- Hyperparameters ---
input_dim = 12
hidden_dim = 50
output_dim = 1
learning_rate = 0.0001
batch_size = 64

data_module = TimeSeriesDataModule(
    train_data=(x_train, y_train),
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

# --- Train with each activation function ---
results = {}
for name, activation_func in activations.items():
    print(f"Training with {name} activation...")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = ffnetwork_act(input_dim, hidden_dim, activation_func, output_dim,
                          learning_rate)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10,
                                         verbose=False)
    trainer = L.Trainer(max_epochs=2000, callbacks=[early_stop_callback],
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

    unscaled_predictions = scaler.inverse_transform(
        np.array(recursive_preds).reshape(-1, 1))
    results[name] = unscaled_predictions
    del model
    torch.cuda.empty_cache()

# --- Activation comparison plot ---
plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
for idx, (name, preds) in enumerate(results.items()):
    test_index = data.index[-len(preds):]
    plt.plot(test_index, preds, label=f'{name}',
             color=custom_palette[(idx + 1) % len(custom_palette)], linewidth=2.0)
plt.plot(data.index, data['passengers'], label='Actual',
         color=custom_palette[0], linewidth=2.0)
plt.title('Airline Passengers - Predictions with Different Activation Functions')
plt.legend()
plt.show()

# --- Activation comparison accuracy ---
unscaled_actuals = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
errors = {'Activation': [], 'MAE': [], 'MSE': [], 'RMSE': [], 'MAPE': [], 'sMAPE': []}
for name, preds in results.items():
    errors['Activation'].append(name)
    errors['MAE'].append(mean_absolute_error(unscaled_actuals, preds))
    errors['MSE'].append(mean_squared_error(unscaled_actuals, preds))
    errors['RMSE'].append(rmse(unscaled_actuals, preds))
    errors['MAPE'].append(mape(unscaled_actuals, preds))
    errors['sMAPE'].append(smape(unscaled_actuals, preds))
print(pd.DataFrame(errors).to_string(index=False))


# =============================================================================
# 6.3  Hidden Layers Experiment
# =============================================================================

class ffnetwork_layers(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers=1, output_dim=1,
                 learning_rate=0.0001, activation_func=nn.ReLU()):
        super().__init__()
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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# --- Train with 1-6 hidden layers ---
results = {}
hidden_layer_configs = range(1, 7)

for num_layers in hidden_layer_configs:
    print(f"Training model with {num_layers} hidden layers...")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = ffnetwork_layers(input_dim, hidden_dim, num_layers=num_layers,
                              output_dim=output_dim, learning_rate=learning_rate)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10,
                                         verbose=False)
    trainer = L.Trainer(max_epochs=2000, callbacks=[early_stop_callback],
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

    unscaled_predictions = scaler.inverse_transform(
        np.array(recursive_preds).reshape(-1, 1))
    results[f'{num_layers} layers'] = unscaled_predictions
    del model
    torch.cuda.empty_cache()

# --- Hidden layers comparison plot ---
plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
for idx, (name, preds) in enumerate(results.items()):
    test_index = data.index[-len(preds):]
    plt.plot(test_index, preds, label=name,
             color=custom_palette[(idx + 1) % len(custom_palette)], linewidth=2.0)
plt.plot(data.index, data['passengers'], label='Actual',
         color=custom_palette[0], linewidth=2.0)
plt.title('Airline Passengers - Impact of Hidden Layers')
plt.legend()
plt.show() 

# --- Hidden layers accuracy ---
errors = {'Model': [], 'MAE': [], 'MSE': [], 'RMSE': [], 'MAPE': [], 'sMAPE': []}
for name, preds in results.items():
    errors['Model'].append(name)
    errors['MAE'].append(mean_absolute_error(unscaled_actuals, preds))
    errors['MSE'].append(mean_squared_error(unscaled_actuals, preds))
    errors['RMSE'].append(rmse(unscaled_actuals, preds))
    errors['MAPE'].append(mape(unscaled_actuals, preds))
    errors['sMAPE'].append(smape(unscaled_actuals, preds))
print(pd.DataFrame(errors).to_string(index=False))


# =============================================================================
# 6.4  Loss Functions — Generalisation vs Training
# =============================================================================

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


# --- Local minima illustration ---
def fd(x):
    return f(x) + 0.3 * torch.cos(4 * np.pi * x)

plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
plt.plot(x, fd(x), label='fd (training)', color=custom_palette[2])
plt.xlabel('x')
plt.ylabel('Value')
plt.legend()
plt.show()


# =============================================================================
# 6.5  Optimiser Comparison
# =============================================================================

class ffnetwork_opt(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers=1, output_dim=1,
                 learning_rate=0.0001, dropout_rate=0.00, activation_func=nn.ReLU(),
                 optimizer_name='Adam'):
        super().__init__()
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


# --- Train with each optimiser, log to local directory ---
optimizer_names = ['SGD', 'Adam', 'Adadelta', 'RMSprop', 'Adagrad']
log_base = Path.cwd() / "lightning_logs"
results = {}

for optimizer_name in optimizer_names:
    print(f"Training with {optimizer_name} optimizer...")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = ffnetwork_opt(input_dim, hidden_dim, num_layers=2, output_dim=output_dim,
                          learning_rate=0.001, optimizer_name=optimizer_name)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10,
                                         verbose=False, mode='min')
    logger = CSVLogger(str(log_base), name=optimizer_name)

    trainer = L.Trainer(max_epochs=2000, callbacks=[early_stop_callback],
                        logger=logger, enable_checkpointing=False)
    trainer.fit(model, data_module)

    model.eval()
    last_window = x_val[-1].view(1, -1)
    recursive_preds = []
    for _ in range(len(x_test)):
        with torch.no_grad():
            pred = model(last_window)
        recursive_preds.append(pred.item())
        last_window = torch.cat((last_window[:, 1:], pred.view(1, 1)), dim=1)
    unscaled_predictions = scaler.inverse_transform(
        np.array(recursive_preds).reshape(-1, 1))
    results[optimizer_name] = unscaled_predictions
    del model
    torch.cuda.empty_cache()

# --- Optimiser predictions plot ---
style_cycle = itertools.cycle(line_styles)
colour_cycle = itertools.cycle(custom_palette)

for name, preds in results.items():
    test_index = data.index[-len(preds):]
    plt.plot(test_index, preds, label=f'{name}',
             color=next(colour_cycle), linestyle=next(style_cycle), linewidth=4)
plt.plot(data.index, data['passengers'], label='Actual',
         color='black', linewidth=2.0)
plt.title('Airline Passengers - Predictions with Different Optimizers')
plt.legend()
plt.show() 

# --- Validation loss by optimiser (from CSV logs) ---
plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
for opt in optimizer_names:
    metrics_path = log_base / opt / "version_0" / "metrics.csv"
    if metrics_path.exists():
        df_log = pd.read_csv(metrics_path)
        val_df = df_log[df_log["val_loss"].notna()].copy()
        plt.plot(val_df["epoch"], val_df["val_loss"], label=opt, linewidth=2)
plt.title("Validation Loss by Optimizer")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# --- Optimiser accuracy ---
errors = {'Model': [], 'MAE': [], 'MSE': [], 'RMSE': [], 'MAPE': [], 'sMAPE': []}
for name, preds in results.items():
    errors['Model'].append(name)
    errors['MAE'].append(mean_absolute_error(unscaled_actuals, preds))
    errors['MSE'].append(mean_squared_error(unscaled_actuals, preds))
    errors['RMSE'].append(rmse(unscaled_actuals, preds))
    errors['MAPE'].append(mape(unscaled_actuals, preds))
    errors['sMAPE'].append(smape(unscaled_actuals, preds))
print(pd.DataFrame(errors).to_string(index=False))


# =============================================================================
# 6.6  Hyperparameter Tuning with Optuna (example)
# =============================================================================

def objective(trial):
    lr = trial.suggest_float('learning_rate', 0.0001, 0.1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3)
    hd = trial.suggest_categorical('hidden_dim', [6, 10, 20, 40])
    nl = trial.suggest_int('num_layers', 1, 6)

    model = ffnetwork_opt(input_dim, hd, num_layers=nl, output_dim=output_dim,
                          learning_rate=lr, dropout_rate=dropout_rate)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10,
                                         verbose=False, mode='min')
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
print(f"Best hyperparameters: {study.best_params}")
print(f"Best MSE: {study.best_value:.4f}")