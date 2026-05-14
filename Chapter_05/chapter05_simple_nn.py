# =============================================================================
# Chapter 5: Simple Neural Architecture
# =============================================================================
# Code companion for Chapter 5
# Covers: perceptrons, artificial neurons, activation functions, feedforward
#         neural networks (FFN), window embeddings, early stopping
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD, Adam

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

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
    data_folder = Path.cwd() /  "Data" # "Time-Series-with-PyTorch" /
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
# 5.1  Perceptron
# =============================================================================

class true_perceptron:
    def __init__(self, input_dim, learning_rate=0.1):
        self.weights = np.zeros(input_dim + 1)  # +1 for the bias
        self.learning_rate = learning_rate

    def predict(self, inputs):
        # Unit step activation function
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels, epochs=20):
        for _ in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)


# Train perceptron on AND function
training_inputs = []
training_inputs.append(np.array([1, 1]))
training_inputs.append(np.array([1, 0]))
training_inputs.append(np.array([0, 1]))
training_inputs.append(np.array([0, 0]))

labels = np.array([1, 0, 0, 0])

perceptron = true_perceptron(2)
perceptron.train(training_inputs, labels)

# Test: should be 1
inputs = np.array([1, 1])
print(perceptron.predict(inputs))

# Test: should be 0
inputs = np.array([0, 1])
print(perceptron.predict(inputs))


# --- Decision boundary plot ---
def plot_decision_boundary(perceptron, inputs, labels):
    df = pd.DataFrame(
        np.concatenate([inputs, labels.reshape(-1, 1)], axis=1),
        columns=['x', 'y', 'label'])
    sns.scatterplot(x='x', y='y', hue='label', style='label', data=df)

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    xs = np.linspace(xmin, xmax, 30)
    ys = np.linspace(ymin, ymax, 30)
    Y, X = np.meshgrid(ys, xs)

    predictions = np.array(
        [perceptron.predict(np.array([x, y]))
         for x, y in zip(np.ravel(X), np.ravel(Y))])
    predictions = predictions.reshape(X.shape)

    ax = plt.gca()
    ax.contourf(X, Y, predictions,
                colors=[custom_palette[2], custom_palette[1]],
                alpha=0.2, levels=range(-1, 2))
    plt.show()

plot_decision_boundary(perceptron, training_inputs, labels)


# --- Perceptron applied to regression (fails) ---
np.random.seed(42)

x_train = np.linspace(-10, 10, num=500)
y_train = 3 * x_train + 7 + np.random.normal(0, 10, size=x_train.shape)

perceptron = true_perceptron(1)

x_train_reshaped = np.array([np.array([x]) for x in x_train])
y_train_reshaped = np.where(y_train > 0, 1, 0)
perceptron.train(x_train_reshaped, y_train_reshaped, epochs=1000)

df = pd.DataFrame({
    'X values': x_train.flatten(),
    'Y values': y_train.flatten(),
    'Prediction': [perceptron.predict(np.array([x])) for x in x_train.flatten()],
    'label': ['Training data' for _ in range(len(x_train))]
})

sns.scatterplot(x='X values', y='Y values', hue='label', data=df,
                palette=[custom_palette[1]])
sns.lineplot(x='X values', y='Prediction', data=df, color=custom_palette[2])

legend_elements = [
    Patch(facecolor=custom_palette[1], edgecolor=custom_palette[1], label='Training data'),
    Patch(facecolor=custom_palette[2], edgecolor=custom_palette[2], label='Predictions')
]
plt.legend(handles=legend_elements, title='Data', loc='best')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.show()


# =============================================================================
# 5.2  Differentiable Perceptron (Artificial Neuron)
# =============================================================================

class Diff_Perceptron:
    def __init__(self, input_dim, learning_rate=0.1, activation_function=lambda x: x):
        self.weights = np.zeros(input_dim + 1)
        self.learning_rate = learning_rate
        self.activation_function = activation_function

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation_function(summation)

    def train(self, training_inputs, labels, epochs=20):
        for _ in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
                prediction = self.activation_function(summation)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error


# --- Linear vs ReLU activation on regression data ---
np.random.seed(42)

nn_linear = Diff_Perceptron(1, activation_function=lambda x: x, learning_rate=0.001)
nn_linear.train(x_train_reshaped, y_train, epochs=200)

relu = lambda x: np.maximum(0, x)
sigmoid = lambda x: 1 / (1 + np.exp(-x))
tanh = lambda x: np.tanh(x)

nn_relu = Diff_Perceptron(1, activation_function=relu, learning_rate=0.00001)
nn_relu.train(x_train_reshaped, y_train, epochs=1000)

plt.scatter(x_train, y_train, color=custom_palette[5], label='Training data')

y_pred_linear = [nn_linear.predict(np.array([x])) for x in x_train]
plt.plot(x_train, y_pred_linear, color=custom_palette[2], label='Linear Activation')

y_pred_relu = [nn_relu.predict(np.array([x])) for x in x_train]
plt.plot(x_train, y_pred_relu, color=custom_palette[3], label='ReLU Activation')

plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Single-Layer Neural Network with Different Activation Functions')
plt.legend(title='Data', loc='best')
plt.show()


# =============================================================================
# 5.3  Single Neuron on Airline Passengers
# =============================================================================

np.random.seed(42)

# Load dataset
data = pd.read_csv(CFG.data_folder / 'passengers.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Sequential split
train_size = int(len(data) * 0.6)
val_size = int(len(data) * 0.2)
train = data[:train_size]
val = data[train_size:train_size + val_size]
test = data[train_size + val_size:]

# Scaling
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
val_scaled = scaler.transform(val)
test_scaled = scaler.transform(test)

# Prepare data with lags
def prepare_data(data, max_lag=12):
    dataX, dataY = [], []
    for i in range(max_lag, len(data) - 1):
        dataX.append(data[i - max_lag:i, 0])
        dataY.append(data[i, 0])
    return np.array(dataX), np.array(dataY)

lags = 12
trainX, trainY = prepare_data(train_scaled, lags)
valX, valY = prepare_data(val_scaled, lags)
testX, testY = prepare_data(test_scaled, lags)

# Initialize model with ReLU activation function
def relu_fn(x):
    return np.maximum(0, x)

diff_perceptron = Diff_Perceptron(
    input_dim=lags, learning_rate=0.001, activation_function=relu_fn)

# Early stopping parameters
best_val_loss = float('inf')
patience, wait = 10, 0

# Train model with early stopping
for epoch in range(200):
    diff_perceptron.train(trainX, trainY, epochs=1)
    val_predictions = diff_perceptron.predict(valX)
    val_loss = mean_squared_error(valY, val_predictions)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping on epoch {epoch}")
            break

# Generate predictions
scaled_predictions = diff_perceptron.predict(testX)

train_predictions = diff_perceptron.predict(trainX)
unscaled_train_predictions = scaler.inverse_transform(train_predictions.reshape(-1, 1))
train_pred_index = data.index[lags:lags + len(unscaled_train_predictions)]

val_predictions_plot = diff_perceptron.predict(valX)
unscaled_val_predictions = scaler.inverse_transform(val_predictions_plot.reshape(-1, 1))
val_pred_index = data.index[train_size + lags:train_size + lags + len(unscaled_val_predictions)]

unscaled_predictions = scaler.inverse_transform(scaled_predictions.reshape(-1, 1))
test_index = data.index[-len(unscaled_predictions):]

# --- Single neuron predictions plot ---
plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
plt.plot(data.index, data['passengers'], label='Actual', color=custom_palette[0])
plt.plot(train_pred_index, unscaled_train_predictions,
         label='In-sample', color=custom_palette[1], linestyle='-.')
plt.plot(val_pred_index, unscaled_val_predictions,
         label='Validation', color=custom_palette[3], linestyle='-.')
plt.plot(test_index, unscaled_predictions,
         label='Test', color=custom_palette[2], linestyle='--')
plt.title('Airline Passengers — Single Neuron')
plt.legend()
plt.show()

# Single neuron accuracy
target = scaler.inverse_transform(testY.reshape(-1, 1))
predictions = unscaled_predictions

fcst_err_df = pd.DataFrame({
    'MAE': [mae(target, predictions)],
    'MSE': [mse(target, predictions)],
    'RMSE': [rmse(target, predictions)],
    'MAPE': [mape(target, predictions)],
    'sMAPE': [smape(target, predictions)]
}, index=['Single Neuron'])
print(fcst_err_df.to_string())


# =============================================================================
# 5.4  Feedforward Neural Network (FFN)
# =============================================================================

# Window embedding function
def create_time_windows(data, window_size):
    inputs = []
    targets = []
    for i in range(len(data) - window_size):
        inputs.append(data[i:i + window_size])
        targets.append(data[i + window_size])
    return np.array(inputs), np.array(targets)

window_size = 12

train_inputs, train_targets = create_time_windows(train_scaled, window_size)
val_inputs, val_targets = create_time_windows(val_scaled, window_size)
test_inputs, test_targets = create_time_windows(test_scaled, window_size)

# Convert to tensors
x_train = torch.FloatTensor(train_inputs)
y_train = torch.FloatTensor(train_targets)
x_val = torch.FloatTensor(val_inputs)
y_val = torch.FloatTensor(val_targets)
x_test = torch.FloatTensor(test_inputs)
y_test = torch.FloatTensor(test_targets)

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)


# Define FFN model
class ff_network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(ff_network, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Instantiate and train
np.random.seed(42)

input_dim = 12
hidden_dim = 100
output_dim = 1
model = ff_network(input_dim, hidden_dim, output_dim)

criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.0001)

best_val_loss = float('inf')
patience, wait = 10, 0

train_losses = []
val_losses = []
best_model_state = None

for epoch in range(2000):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train.view(-1, window_size))
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_outputs = model(x_val.view(-1, window_size))
        val_loss = criterion(val_outputs, y_val)

    val_losses.append(val_loss.item())

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        wait = 10
        best_model_state = model.state_dict()
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping on epoch {epoch}")
            break

if best_model_state:
    model.load_state_dict(best_model_state)


# --- Training and validation loss ---
plt.plot(range(epoch + 1), train_losses, label='Train', color=custom_palette[0])
plt.plot(range(epoch + 1), val_losses, label='Validation', color=custom_palette[3])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('FFN Training and Validation Loss')
plt.legend()
plt.show()


# --- FFN predictions ---
model.eval()
with torch.no_grad():
    train_preds = model(x_train.view(-1, window_size)).numpy()
    val_preds = model(x_val.view(-1, window_size)).numpy()
    test_preds = model(x_test.view(-1, window_size)).numpy()

train_preds = scaler.inverse_transform(train_preds)
val_preds = scaler.inverse_transform(val_preds)
test_preds = scaler.inverse_transform(test_preds)
all_actuals = scaler.inverse_transform(
    np.concatenate([y_train, y_val, y_test]).reshape(-1, 1))

all_preds = np.concatenate([train_preds, val_preds, test_preds])
pred_index = data.index[window_size:window_size + len(all_preds)]

insample_end = len(train_preds) + len(val_preds)

plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
plt.plot(pred_index, all_actuals, label='Actual', color=custom_palette[1])
plt.plot(pred_index[:insample_end], all_preds[:insample_end],
         label='Predicted (in-sample)', color=custom_palette[2], linestyle='--')
plt.plot(pred_index[insample_end:], all_preds[insample_end:],
         label='Predicted (test)', color=custom_palette[3], linestyle='--')
plt.axvline(pred_index[insample_end], color=custom_palette[0],
            linestyle=':', label='Test split')
plt.xlabel('Year')
plt.ylabel('Passengers (count)')
plt.title('Airline Passengers FFN Predictions')
plt.legend()
plt.show()


# --- FFN accuracy ---
unscaled_actuals = scaler.inverse_transform(y_test.detach().numpy().reshape(-1, 1))

ffn_err_df = pd.DataFrame({
    'MAE': [mae(unscaled_actuals, test_preds)],
    'MSE': [mse(unscaled_actuals, test_preds)],
    'RMSE': [rmse(unscaled_actuals, test_preds)],
    'MAPE': [mape(unscaled_actuals, test_preds)],
    'sMAPE': [smape(unscaled_actuals, test_preds)]
}, index=['FFN'])
print(ffn_err_df.to_string())