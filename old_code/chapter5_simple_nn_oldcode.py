# Extracted from chapter5_Simple_nn.qmd
# Do not edit the source .qmd file directly.

#| label: Chapter 4 libraries
#| message: false
#| echo: false
#| eval: true
from pathlib import Path
import pandas as pd
import seaborn as sns
import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD
#from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Define palette
custom_palette = ["#000000", "#0072B2", "#D55E00","#009E73","#CC79A7", "#56B4E9","#E69F00"]

# general settings
class CFG:
    data_folder = Path.cwd().parent / "data"
    img_dim1 = 12
    img_dim2 = 6
    fontsize = 18
    
    
# adjust the parameters for displayed figures    
plt.rcParams.update({'figure.figsize': (CFG.img_dim1,CFG.img_dim2)}) 
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Source Sans Pro', 'Arial']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titlesize'] = 18

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

# ----------------------------------------------------------------------

#| label: Perceptron as a class
#| message: false
#| echo: true
#| eval: true

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

# ----------------------------------------------------------------------

#| label: Perceptron training
#| message: false
#| echo: true
#| eval: true

# Create training inputs for AND function
training_inputs = []
training_inputs.append(np.array([1, 1]))
training_inputs.append(np.array([1, 0]))
training_inputs.append(np.array([0, 1]))
training_inputs.append(np.array([0, 0]))

# Labels for AND function
labels = np.array([1, 0, 0, 0])

# Create a Perceptron object
perceptron = true_perceptron(2)

# Train perceptron
perceptron.train(training_inputs, labels)

# Test perceptron
# Output should be 1 
inputs = np.array([1, 1])
print(perceptron.predict(inputs))  

# Output should be 0
inputs = np.array([0, 1])
print(perceptron.predict(inputs))

# ----------------------------------------------------------------------

#| label: Perceptron decision boundary
#| message: false
#| echo: true
#| eval: true
def plot_decision_boundary(perceptron, inputs, labels):
    plt.style.use('fivethirtyeight')

    # Scatterplot of data points
    df = pd.DataFrame(np.concatenate([inputs, labels.reshape(-1,1)], axis=1), columns=['x', 'y', 'label'])
    sns.scatterplot(x='x', y='y', hue='label', style='label', data=df)

    # Generate grid over input space to plot colour of classification
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    xs = np.linspace(xmin, xmax, 30)
    ys = np.linspace(ymin, ymax, 30)
    Y, X = np.meshgrid(ys, xs)

    # Compute predictions over grid
    predictions = np.array([perceptron.predict(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))])
    predictions = predictions.reshape(X.shape)

    # Show boundary line
    ax = plt.gca()
    ax.contourf(X, Y, predictions, colors=['#D55E00', '#0072B2'], alpha=0.2, levels=range(-1, 2))
    plt.show()

# Call decision boundary function with trained perceptron
plot_decision_boundary(perceptron, training_inputs, labels)

# ----------------------------------------------------------------------

#| label: Perceptron ('true') applied to regression
#| message: false
#| echo: false
#| eval: true
plt.style.use('fivethirtyeight')

# Create some data for regression 
np.random.seed(42)  # for reproducibility

# Create simple linearly increasing data with noise
x_train = np.linspace(-10, 10, num=500)
y_train = 3 * x_train + 7 + np.random.normal(0, 10, size=x_train.shape)

# Create a Perceptron object
perceptron = true_perceptron(1)

# Train perceptron
# Reshape x_train and y_train to fit the perceptron's expected input shape
x_train_reshaped = np.array([np.array([x]) for x in x_train])
y_train_reshaped = np.where(y_train > 0, 1, 0)  # Binary labels
perceptron.train(x_train_reshaped, y_train_reshaped, epochs=1000)

# Create DataFrame to hold the train data and predictions
df = pd.DataFrame({
    'X values': x_train.flatten(),  # Renamed the x-axis label
    'Y values': y_train.flatten(),  # Renamed the y-axis label
    'Prediction': [perceptron.predict(np.array([x])) for x in x_train.flatten()],
    'label': ['Training data' for _ in range(len(x_train))]
})

# Scatterplot of train data
sns.scatterplot(x='X values', y='Y values', hue='label', data=df, palette=['#0F95D7'])

# lineplot of perceptron predictions
sns.lineplot(x='X values', y='Prediction', data=df, color='#FF2700')

# Add legend
legend_elements = [
    Patch(facecolor='#0F95D7', edgecolor='#0F95D7', label='Training data'),
    Patch(facecolor='#FF2700', edgecolor='#FF2700', label='Predictions')
]

plt.legend(handles=legend_elements, title='Data', loc='best')

# Change the x-axis and y-axis labels
plt.xlabel('X values')
plt.ylabel('Y values')
plt.show()

# ----------------------------------------------------------------------

#| label: Perceptron class with activation argument
#| message: false
#| echo: true
#| eval: true

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

# ----------------------------------------------------------------------

#| label: Perceptron modelling with linear function only and ReLU
#| message: false
#| echo: true
#| eval: true

# set seed
np.random.seed(42)

# Instantiate Perceptron class with linear activation 
nn_linear = Diff_Perceptron(1, activation_function=lambda x: x, learning_rate= 0.001)

# Train Perceptron
nn_linear.train(x_train_reshaped, y_train, epochs=200)

# Some activation functions to test 
relu = lambda x: np.maximum(0, x)
sigmoid = lambda x: 1 / (1 + np.exp(-x))
tanh = lambda x: np.tanh(x)

# Instantiate Perceptron class with ReLU activation 
nn_relu = Diff_Perceptron(1, activation_function=relu, learning_rate= 0.00001)

# Train Perceptron
nn_relu.train(x_train_reshaped, y_train, epochs=1000)

# Scatterplot of training data
plt.scatter(x_train, y_train, color='#0F95D7', label='Training data')

# Lineplot of predictions for linear activation
y_pred_linear = [nn_linear.predict(np.array([x])) for x in x_train]
plt.plot(x_train, y_pred_linear, color='#FF2700', label='Linear Activation')

# Lineplot of predictions for ReLU activation
y_pred_relu = [nn_relu.predict(np.array([x])) for x in x_train]
plt.plot(x_train, y_pred_relu, color='#5FA613', label='ReLU Activation')
plt.xlabel('X values')
plt.ylabel('Y values')

plt.title('Single-Layer Neural Network with Different Activation Functions')
plt.legend(title='Data', loc='best')

plt.show()

# ----------------------------------------------------------------------

#| label: Single neuron model with ReLU activation of Airline Passengers 
#| message: false
#| echo: true
#| eval: true

# Set seed
np.random.seed(42)

# Load dataset
data = pd.read_csv(CFG.data_folder / 'passengers.csv')

# Preprocess data
# Convert 'date' to datetime
data['date'] = pd.to_datetime(data['date'])
# Set 'date' as the index
data.set_index('date', inplace=True)

# Sequential split
train_size = int(len(data) * 0.6)
val_size = int(len(data) * 0.2)
train, val, test = data[:train_size], data[train_size:train_size + val_size], data[train_size + val_size:]

# Scaling: Min-max 
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
def relu(x):
    return np.maximum(0, x)

diff_perceptron = Diff_Perceptron(input_dim=lags, learning_rate=0.001, activation_function=relu)

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

# Generate test predictions
scaled_predictions = diff_perceptron.predict(testX)

# In-sample predictions
train_predictions = diff_perceptron.predict(trainX)
unscaled_train_predictions = scaler.inverse_transform(train_predictions.reshape(-1, 1))
train_pred_index = data.index[lags:lags + len(unscaled_train_predictions)]

# Validation predictions
val_predictions_plot = diff_perceptron.predict(valX)
unscaled_val_predictions = scaler.inverse_transform(val_predictions_plot.reshape(-1, 1))
val_pred_index = data.index[train_size + lags:train_size + lags + len(unscaled_val_predictions)]

# Inverse Scaling 
unscaled_predictions = scaler.inverse_transform(scaled_predictions.reshape(-1, 1))
test_index = data.index[-len(unscaled_predictions):]

# ----------------------------------------------------------------------

#| label: Graphing single neuron model 
#| message: false
#| echo: false
#| eval: true

# Plot passenegrs and predictions
plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
plt.plot(data.index, data['passengers'], label='Actual', color=custom_palette[0])
plt.plot(train_pred_index, unscaled_train_predictions, label='In-sample', color=custom_palette[1], linestyle=('-.'))
plt.plot(val_pred_index, unscaled_val_predictions, label='Validation', color=custom_palette[3], linestyle='-.')
plt.plot(test_index, unscaled_predictions, label='Test', color=custom_palette[2], linestyle='--')
plt.title('Airline Passengers')
plt.legend()
plt.show()

# ----------------------------------------------------------------------

#| label: Accuracy of single neuron model 
#| message: false
#| echo: false
#| eval: true

target = scaler.inverse_transform(testY.reshape(-1, 1))
predictions = unscaled_predictions

# Calculate errors
mae_error = mae(target, predictions)
mse_error = mse(target, predictions)
rmse_error = rmse(target, predictions)
mape_error = mape(target, predictions)
smape_error = smape(target, predictions)

# Create a dataframe
fcst_err_df = pd.DataFrame({'MAE': [mae_error], 'MSE': [mse_error], 'RMSE': [rmse_error], 'MAPE': [mape_error], 'sMAPE': [smape_error]})
#print(fcst_err_df)

# ----------------------------------------------------------------------

#| label: Pytorch Libraries for neural network
#| message: false
#| echo: true
#| eval: true

import torch
from torch import nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------------------

#| label: Create window embeddings
#| message: false
#| echo: true
#| eval: true

# Define a function to create time window embeddings
def create_time_windows(data, window_size):
    inputs = []
    targets = []
    for i in range(len(data) - window_size):
        inputs.append(data[i:i+window_size])
        targets.append(data[i+window_size])
    return np.array(inputs), np.array(targets)

window_size = 12 # good parameter to play with to understand effect

train_inputs, train_targets = create_time_windows(train_scaled, window_size)
val_inputs, val_targets = create_time_windows(val_scaled, window_size)
test_inputs, test_targets = create_time_windows(test_scaled, window_size)

# ----------------------------------------------------------------------

#| label: Prepare data
#| message: false
#| echo: true
#| eval: true

# Convert numpy arrays to PyTorch tensors
x_train = torch.FloatTensor(train_inputs)
y_train = torch.FloatTensor(train_targets)
x_val = torch.FloatTensor(val_inputs)
y_val = torch.FloatTensor(val_targets)
x_test = torch.FloatTensor(test_inputs)
y_test = torch.FloatTensor(test_targets)

# Print shapes
x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape

# ----------------------------------------------------------------------

#| label: Feedforward Neural Net
#| message: false
#| echo: false
#| eval: true

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

# ----------------------------------------------------------------------

#| label: Instantiate FFN, and train network
#| message: false
#| echo: true
#| eval: true

np.random.seed(42)

# Initialize FFN 
input_dim = 12 
hidden_dim = 100 # Tunable 
output_dim = 1 # We are predicting a single output 
model = ff_network(input_dim, hidden_dim, output_dim)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.0001)


# Early stopping parameters
best_val_loss = float('inf')
patience, wait = 10, 0

# Lists to hold loss values for plotting
train_losses = []
val_losses = []

# Best model checkpoint
best_model_state = None

# Training loop with early stopping
for epoch in range(2000):
    model.train()
    optimizer.zero_grad()
    # Flatten time window dimension
    outputs = model(x_train.view(-1, window_size))
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Store train loss
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_outputs = model(x_val.view(-1, window_size))
        val_loss = criterion(val_outputs, y_val)

    # Store validation loss
    val_losses.append(val_loss.item())

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        wait = 10
        # Save best model's state
        best_model_state = model.state_dict()
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping on epoch {epoch}")
            break

# Load best model's state
if best_model_state:
    model.load_state_dict(best_model_state)

# ----------------------------------------------------------------------

#| label: Training and validation loss graph
#| message: false
#| echo: false
#| eval: true

# Plot training and validation losses
plt.plot(range(epoch+1), train_losses, label='Train', color = custom_palette[0])
plt.plot(range(epoch+1), val_losses, label='Validation', color = custom_palette[3])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ----------------------------------------------------------------------

#| label: FFN predictions
#| message: false
#| echo: false
#| eval: true

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
pred_index = data.index[window_size : window_size + len(all_preds)]

# split point based on actual array lengths not original row counts
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

# ----------------------------------------------------------------------

#| label: FFN accuracy
#| message: false
#| echo: false
#| eval: true

# use recursive_preds already generated in the plot block above
unscaled_actuals = scaler.inverse_transform(y_test.detach().numpy().reshape(-1, 1))

mae_error_val = mae(unscaled_actuals, test_preds)
mse_error_val = mse(unscaled_actuals, test_preds)
rmse_error_val = rmse(unscaled_actuals, test_preds)
mape_error_val = mape(unscaled_actuals, test_preds)
smape_error_val = smape(unscaled_actuals, test_preds)

fcst_err_df = pd.DataFrame({'MAE': [mae_error_val],
                            'MSE': [mse_error_val],
                            'RMSE': [rmse_error_val],
                            'MAPE': [mape_error_val],
                            'sMAPE': [smape_error_val]},
                           index=['Test'])

print(fcst_err_df)