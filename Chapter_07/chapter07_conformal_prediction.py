# =============================================================================
# Chapter 7: Conformal Prediction
# =============================================================================
# Code companion for Chapter 7
# Covers: uncertainty quantification intuition, conformity ladders,
#         non-conformity scores, split conformal method, quantile intervals,
#         full conformal refitting problem, EnbPI with MAPIE on a PyTorch model
# =============================================================================

import os
import platform
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    data_folder = Path.cwd()  / "Data" # / "Time-Series-with-PyTorch"
    img_dim1 = 12
    img_dim2 = 6
    fontsize = 18

plt.rcParams.update({'figure.figsize': (CFG.img_dim1, CFG.img_dim2)})


# =============================================================================
# 7.1  Point Prediction Error
# =============================================================================

# --- Figure 7.1: Point prediction relative to actual value ---
x = np.linspace(0.1, 0.9, 100)
y = 2 * x + 1

x_pred = 0.6
y_pred = 2 * x_pred + 1
y_true = 2.5

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, label='Linear prediction (Model)', color=custom_palette[0], linestyle='--')
ax.scatter([x_pred], [y_pred], color=custom_palette[1], zorder=5)
ax.scatter([x_pred], [y_true + 0.12], color=custom_palette[2], zorder=5)
ax.annotate('Predicted Value', (x_pred, y_pred), textcoords="offset points",
            xytext=(-57, -3), ha='center', color=custom_palette[1])
ax.annotate('True Value', (x_pred, y_true), textcoords="offset points",
            xytext=(-42, 15), ha='center', color=custom_palette[2])
ax.arrow(x_pred, y_pred, 0, y_true - y_pred, head_width=0.02,
         head_length=0.1, fc=custom_palette[1], ec=custom_palette[2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Point Prediction relative to actual value')
ax.legend()
plt.grid(True)
plt.show()


# =============================================================================
# 7.2  Uncertainty Types
# =============================================================================

# --- Figure 7.2: Aleatoric and epistemic uncertainty ---
np.random.seed(0)
x_high_aleatoric = np.linspace(-4, -2, 50)
y_high_aleatoric = 0.3 * x_high_aleatoric + np.random.normal(loc=0, scale=1.5, size=50)

x_low_aleatoric = np.linspace(2, 4, 50)
y_low_aleatoric = 0.5 * x_low_aleatoric + np.random.normal(loc=0, scale=0.3, size=50)

x_line = np.linspace(-10, 10, 200)
y_line = 0.5 * x_line

fig, ax = plt.subplots(figsize=(CFG.img_dim1, CFG.img_dim2))
ax.scatter(x_high_aleatoric, y_high_aleatoric, color=custom_palette[5],
           label='High Aleatoric \n Uncertainty')
ax.scatter(x_low_aleatoric, y_low_aleatoric, color=custom_palette[6],
           alpha=0.6, label='Aleatoric \n Uncertainty')
ax.plot(x_line, y_line, 'k--', label='Real Function')
ax.annotate('High Aleatoric \n Uncertainty', xy=(-3.0, 1.5), xytext=(-7.5, 4.5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'), fontsize=14)
ax.annotate('Low Aleatoric \n Uncertainty', xy=(2.5, 1.5), xytext=(2.5, 4.5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'), fontsize=14)
ax.annotate('High Epistemic \n Uncertainty', xy=(-7.5, -4), xytext=(-5, -5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'), fontsize=14)
ax.annotate('High Epistemic \n Uncertainty', xy=(6.5, 3), xytext=(5, -5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'), fontsize=14)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Data with Uncertainty \n')
ax.legend(loc='right', fontsize=12)
plt.show()


# =============================================================================
# 7.3  Non-Conformity and Distance
# =============================================================================

# --- Figure 7.3: Absolute error from linear model ---
np.random.seed(235987)
x = np.linspace(0, 10, 30)
y = 2 * x + 1 + np.random.normal(scale=2, size=30)

p, residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
fitted_y = np.poly1d(p)(x)
absolute_errors = np.abs(y - fitted_y)

sorted_indices = np.argsort(absolute_errors)
point_near_index = sorted_indices[3]
point_far_index = sorted_indices[-2]

point_near = (x[point_near_index], y[point_near_index])
point_far = (x[point_far_index], y[point_far_index])
fitted_near = fitted_y[point_near_index]
fitted_far = fitted_y[point_far_index]
ae_near = absolute_errors[point_near_index]
ae_far = absolute_errors[point_far_index]

fig, ax = plt.subplots()
ax.plot(x, y, 'o', label='Data points', color=custom_palette[6])
ax.plot(x, fitted_y, '--', label='Regression Line', color=custom_palette[1])
ax.scatter(*point_near, color=custom_palette[3], label='Predicted point (low AE)', zorder=5)
ax.scatter(*point_far, color=custom_palette[2], label='Predicted point (high AE)', zorder=5)
ax.plot([point_near[0], point_near[0]], [point_near[1], fitted_near],
        color=custom_palette[3], linestyle='--')
ax.plot([point_far[0], point_far[0]], [point_far[1], fitted_far],
        color=custom_palette[2], linestyle='--')
ax.text(point_near[0] - 0.1, (point_near[1] + fitted_near) / 1.7,
        f'AE: {ae_near:.2f}', ha='right')
ax.text(point_far[0] - 0.1, (point_far[1] + fitted_far) / 1.85,
        f'AE: {ae_far:.2f}', ha='right')
ax.set_xlabel('X values')
ax.set_ylabel('Y values')
ax.set_title('Distance (absolute error) from linear model')
ax.legend()
plt.grid(True)
plt.show()


# =============================================================================
# 7.4  Conformity Ladder
# =============================================================================

np.random.seed(42)
x = np.random.normal(50, 15, 20)
y_noisy = x * 3 + np.random.normal(0, 20, 20)

model_noisy = LinearRegression().fit(x.reshape(-1, 1), y_noisy)
y_pred_noisy = model_noisy.predict(x.reshape(-1, 1))
abs_errors_noisy = np.abs(y_pred_noisy - y_noisy)

# Shift one point to an extreme
x_modified = x.copy()
y_modified = y_noisy.copy()
y_modified[10] += 55

y_pred_modified = model_noisy.predict(x_modified.reshape(-1, 1))
abs_errors_modified = np.abs(y_pred_modified - y_modified)

conformity_scores_noisy = 1 - np.argsort(np.argsort(abs_errors_noisy)) / len(abs_errors_noisy)
conformity_scores_modified = 1 - np.argsort(np.argsort(abs_errors_modified)) / len(abs_errors_modified)

sorted_indices_noisy = np.argsort(-conformity_scores_noisy)
sorted_indices_modified = np.argsort(-conformity_scores_modified)

colors_noisy = [sns.color_palette("RdYlGn_r", n_colors=len(abs_errors_noisy))[i]
                for i in range(len(abs_errors_noisy))]
colors_modified = [sns.color_palette("RdYlGn_r", n_colors=len(abs_errors_modified))[i]
                   for i in range(len(abs_errors_modified))]

highlight_index = 10
highlight_color = "white"

# --- Figure 7.4: Conformity ladder (original vs modified) ---
fig, ax = plt.subplots(2, 2, figsize=(18, 16))

sns.scatterplot(x=x, y=y_noisy, ax=ax[0, 0], color="black", label="Data Points")
ax[0, 0].plot(x[10], y_noisy[10], marker='D', markersize=12, markerfacecolor='none',
              markeredgecolor=custom_palette[2], markeredgewidth=2.5, label="Predicted value", zorder=5)
sns.lineplot(x=x, y=y_pred_noisy, ax=ax[0, 0], color="blue", label="Regression line")
ax[0, 0].set_title("Original data with regression line")
ax[0, 0].set_xlabel("Price"); ax[0, 0].set_ylabel("Sales"); ax[0, 0].legend()

bar_colors = colors_noisy.copy()
bar_colors[sorted_indices_noisy.tolist().index(highlight_index)] = highlight_color
bars = ax[0, 1].barh(np.arange(len(conformity_scores_noisy)),
                      abs_errors_noisy[sorted_indices_noisy], color=bar_colors)
bar_idx = sorted_indices_noisy.tolist().index(highlight_index)
bars[bar_idx].set_hatch('///'); bars[bar_idx].set_edgecolor(custom_palette[0])
bars[bar_idx].set_facecolor('none')
ax[0, 1].set_yticks(np.arange(len(conformity_scores_noisy)))
ax[0, 1].set_yticklabels(np.round(conformity_scores_noisy[sorted_indices_noisy], 2))
ax[0, 1].set_title("Conformity ladder (original)")
ax[0, 1].set_xlabel("Non-conformity (AE)"); ax[0, 1].set_ylabel("Conformity ranking")
ax[0, 1].invert_yaxis()

sns.scatterplot(x=x_modified, y=y_modified, ax=ax[1, 0], color="black", label="Data Points")
ax[1, 0].plot(x_modified[10], y_modified[10], marker='D', markersize=12, markerfacecolor='none',
              markeredgecolor=custom_palette[2], markeredgewidth=2.5, label="Predicted point", zorder=5)
sns.lineplot(x=x_modified, y=y_pred_modified, ax=ax[1, 0], color="blue", label="Regression line")
ax[1, 0].set_title("Modified data with regression line")
ax[1, 0].set_xlabel("Price"); ax[1, 0].set_ylabel("Sales"); ax[1, 0].legend()

bar_colors_mod = colors_modified.copy()
bar_colors_mod[sorted_indices_modified.tolist().index(highlight_index)] = highlight_color
bars_mod = ax[1, 1].barh(np.arange(len(conformity_scores_modified)),
                          abs_errors_modified[sorted_indices_modified], color=bar_colors_mod)
bar_idx_mod = sorted_indices_modified.tolist().index(highlight_index)
bars_mod[bar_idx_mod].set_hatch('///'); bars_mod[bar_idx_mod].set_edgecolor(custom_palette[0])
bars_mod[bar_idx_mod].set_facecolor('none')
ax[1, 1].set_yticks(np.arange(len(conformity_scores_modified)))
ax[1, 1].set_yticklabels(np.round(conformity_scores_modified[sorted_indices_modified], 2))
ax[1, 1].set_title("Conformity ladder (modified)")
ax[1, 1].set_xlabel("Non-conformity (AE)"); ax[1, 1].set_ylabel("Conformity ranking")
ax[1, 1].invert_yaxis()

plt.tight_layout()
plt.show()


# =============================================================================
# 7.5  Full Conformal Refitting Problem
# =============================================================================

np.random.seed(39285)

x = np.random.normal(50, 15, 20)
y_noisy = x * 3 + np.random.normal(0, 20, 20)

poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x.reshape(-1, 1))
model_noisy = LinearRegression().fit(x_poly, y_noisy)
y_pred_noisy = model_noisy.predict(x_poly)
abs_errors_noisy = np.abs(y_pred_noisy - y_noisy)

highest_x_index = np.argmax(x)
x_modified = x.copy()
y_modified = y_noisy.copy()
y_modified[highest_x_index] += 120

x_poly_modified = poly.transform(x_modified.reshape(-1, 1))
model_modified = LinearRegression().fit(x_poly_modified, y_modified)
y_pred_modified = model_modified.predict(x_poly_modified)
abs_errors_modified = np.abs(y_pred_modified - y_modified)

conformity_scores_noisy = np.argsort(np.argsort(abs_errors_noisy)) / (len(abs_errors_noisy) - 1)
conformity_scores_modified = np.argsort(np.argsort(abs_errors_modified)) / (len(abs_errors_modified) - 1)

norm_noisy = plt.Normalize(conformity_scores_noisy.min(), conformity_scores_noisy.max())
sorted_indices_noisy = np.argsort(conformity_scores_noisy)
colors_noisy = plt.cm.RdYlGn_r(norm_noisy(conformity_scores_noisy[sorted_indices_noisy]))

norm_modified = plt.Normalize(conformity_scores_modified.min(), conformity_scores_modified.max())
sorted_indices_modified = np.argsort(conformity_scores_modified)
colors_modified = plt.cm.RdYlGn_r(norm_modified(conformity_scores_modified[sorted_indices_modified]))

highlight_index = highest_x_index

# --- Figure 7.5: Full conformal refitting problem ---
fig, ax = plt.subplots(2, 2, figsize=(18, 16))

sns.scatterplot(x=x, y=y_noisy, ax=ax[0, 0], color="black", label="Data Points")
ax[0, 0].plot(x[highest_x_index], y_noisy[highest_x_index], marker='D', markersize=12,
              markerfacecolor='none', markeredgecolor=custom_palette[2], markeredgewidth=2.5,
              label="High Price Data Point", zorder=5)
sns.lineplot(x=np.sort(x), y=model_noisy.predict(poly.transform(np.sort(x).reshape(-1, 1))),
             ax=ax[0, 0], color="blue", label="Polynomial Regression Line")
ax[0, 0].set_title("Regression line with reasonable prediction")
ax[0, 0].set_xlabel("Price discount"); ax[0, 0].set_ylabel("Sales"); ax[0, 0].legend(fontsize=16)

bar_colors_noisy = ["white" if i == highlight_index else colors_noisy[j]
                     for j, i in enumerate(sorted_indices_noisy)]
bars = ax[0, 1].barh(np.arange(len(conformity_scores_noisy)),
                      abs_errors_noisy[sorted_indices_noisy], color=bar_colors_noisy)
bar_idx = sorted_indices_noisy.tolist().index(highlight_index)
bars[bar_idx].set_hatch('///'); bars[bar_idx].set_edgecolor(custom_palette[0])
bars[bar_idx].set_facecolor('none')
ax[0, 1].set_yticks(np.arange(len(conformity_scores_noisy)))
ax[0, 1].set_yticklabels(np.round(conformity_scores_noisy[sorted_indices_noisy], 2))
ax[0, 1].set_title("Conformity ladder with reasonable prediction")
ax[0, 1].set_xlabel("Non-Conformity (Absolute Error)"); ax[0, 1].set_ylabel("Conformity Ranking")
ax[0, 1].invert_yaxis()

sns.scatterplot(x=x_modified, y=y_modified, ax=ax[1, 0], color="black", label="Data Points")
ax[1, 0].plot(x_modified[highest_x_index], y_modified[highest_x_index], marker='D', markersize=12,
              markerfacecolor='none', markeredgecolor=custom_palette[2], markeredgewidth=2.5,
              label="Modified High Price Data Point", zorder=5)
sns.lineplot(x=np.sort(x_modified),
             y=model_modified.predict(poly.transform(np.sort(x_modified).reshape(-1, 1))),
             ax=ax[1, 0], color="blue", label="New Polynomial Regression Line")
ax[1, 0].set_title("Regression line with extreme prediction")
ax[1, 0].set_xlabel("Price discount"); ax[1, 0].set_ylabel("Sales"); ax[1, 0].legend(fontsize=16)

bar_colors_modified = ["white" if i == highlight_index else colors_modified[j]
                        for j, i in enumerate(sorted_indices_modified)]
bars_mod = ax[1, 1].barh(np.arange(len(conformity_scores_modified)),
                          abs_errors_modified[sorted_indices_modified], color=bar_colors_modified)
bar_idx_mod = sorted_indices_modified.tolist().index(highlight_index)
bars_mod[bar_idx_mod].set_hatch('///'); bars_mod[bar_idx_mod].set_edgecolor(custom_palette[0])
bars_mod[bar_idx_mod].set_facecolor('none')
ax[1, 1].set_yticks(np.arange(len(conformity_scores_modified)))
ax[1, 1].set_yticklabels(np.round(conformity_scores_modified[sorted_indices_modified], 2))
ax[1, 1].set_title("Conformity Ladder with extreme prediction")
ax[1, 1].set_xlabel("Non-Conformity (Absolute Error)"); ax[1, 1].set_ylabel("Conformity Ranking")
ax[1, 1].invert_yaxis()

plt.tight_layout()
plt.show()


# =============================================================================
# 7.6  Split Conformal Method
# =============================================================================

np.random.seed(39285)

x = np.random.normal(50, 15, 20)
y = x * 3 + np.random.normal(0, 20, 20)

x_train, x_calib, y_train, y_calib = train_test_split(x, y, test_size=0.4, random_state=42)

poly = PolynomialFeatures(degree=3)
x_train_poly = poly.fit_transform(x_train.reshape(-1, 1))
x_calib_poly = poly.transform(x_calib.reshape(-1, 1))

model = LinearRegression().fit(x_train_poly, y_train)
y_train_pred = model.predict(x_train_poly)
y_calib_pred = model.predict(x_calib_poly)

abs_errors_calib = np.abs(y_calib_pred - y_calib)
ranks = np.argsort(np.argsort(abs_errors_calib))
conformity_scores = ranks / (len(abs_errors_calib) - 1)
sorted_indices_calib = np.argsort(conformity_scores)

# --- Figure 7.6: Split conformal method ---
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

sns.scatterplot(x=x_train, y=y_train, ax=axs[0], marker='x', color=custom_palette[2],
                s=100, label="Training Data")
sns.scatterplot(x=x_calib, y=y_calib, ax=axs[0], marker='o', color=custom_palette[1],
                s=100, label="Calibration Data")
sns.lineplot(x=np.linspace(min(x), max(x), 100),
             y=model.predict(poly.transform(np.linspace(min(x), max(x), 100).reshape(-1, 1))),
             ax=axs[0], color=custom_palette[3], label="Polynomial Regression Line")
axs[0].set_title("Training and Calibration Data with Polynomial Fit")
axs[0].set_xlabel("Price discount"); axs[0].set_ylabel("Sales"); axs[0].legend(fontsize=16)

norm = plt.Normalize(conformity_scores.min(), conformity_scores.max())
colors = plt.cm.RdYlGn_r(norm(conformity_scores[sorted_indices_calib]))
axs[1].barh(np.arange(len(conformity_scores)),
            abs_errors_calib[sorted_indices_calib], color=colors)
axs[1].set_yticks(np.arange(len(conformity_scores)))
axs[1].set_yticklabels(np.round(conformity_scores[sorted_indices_calib], 2))
axs[1].set_title("Conformity Ladder (calibration only)")
axs[1].set_xlabel("Non-Conformity (Absolute Error)")
axs[1].set_ylabel("Conformity Ranking")
axs[1].invert_yaxis()

plt.tight_layout()
plt.show()


# =============================================================================
# 7.7  Quartile Sub-Intervals
# =============================================================================

np.random.seed(39285)
x = np.random.normal(50, 15, 30)
y = x * 3 + np.random.normal(0, 20, 30)

x_train, x_calib, y_train, y_calib = train_test_split(x, y, test_size=0.4, random_state=42)

poly = PolynomialFeatures(degree=3)
x_train_poly = poly.fit_transform(x_train.reshape(-1, 1))
x_calib_poly = poly.transform(x_calib.reshape(-1, 1))
model = LinearRegression().fit(x_train_poly, y_train)
y_calib_pred = model.predict(x_calib_poly)
abs_errors_calib = np.abs(y_calib_pred - y_calib)
ranks = np.argsort(np.argsort(abs_errors_calib))
conformity_scores = ranks / (len(abs_errors_calib) - 1)
sorted_indices_calib = np.argsort(conformity_scores)
quartile_bounds = np.percentile(conformity_scores, [20, 40, 60, 80])

# --- Figure 7.7: Quartile sub-intervals ---
fig, ax = plt.subplots(figsize=(10, 8))
norm = plt.Normalize(conformity_scores.min(), conformity_scores.max())
colors = plt.cm.RdYlGn_r(norm(conformity_scores[sorted_indices_calib]))
bars = ax.barh(np.arange(len(conformity_scores)),
               abs_errors_calib[sorted_indices_calib], color=colors)
ax.set_yticks(np.arange(len(conformity_scores)))
ax.set_yticklabels(np.round(conformity_scores[sorted_indices_calib], 2))
ax.set_title("Conformity Ladder for Calibration Data")
ax.set_xlabel("Non-Conformity (Absolute Error)")
ax.set_ylabel("Conformity Ranking")
ax.invert_yaxis()
for quartile in quartile_bounds:
    quartile_index = np.where(np.sort(conformity_scores) >= quartile)[0][0]
    quartile_position = len(conformity_scores) - quartile_index
    ax.axhline(y=quartile_position, color='r', linestyle='--')
plt.tight_layout()
plt.show()


# =============================================================================
# 7.8  Applied: EnbPI on a PyTorch FFN
# =============================================================================

# --- Data loading ---
pseudo_sales_path = CFG.data_folder / 'pseudo_sales.csv'

data = pd.read_csv(pseudo_sales_path)
data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y')
data = data.sort_values('date').reset_index(drop=True)

def add_datetime_features(df, datetime_column):
    df = df.copy()
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df['year'] = df[datetime_column].dt.year
    df['month'] = df[datetime_column].dt.month
    df['week'] = df[datetime_column].dt.isocalendar().week.astype(int)
    df['day'] = df[datetime_column].dt.day
    return df

df = add_datetime_features(data, "date")

date_col = df['date'].copy()
target = df['sales'].copy()
features = df.drop(columns=['sales', 'date']).copy()

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

features_scaled = feature_scaler.fit_transform(features)
target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1))

def create_time_windows(features, target, window_size):
    inputs, targets = [], []
    for i in range(len(features) - window_size):
        inputs.append(features[i:i + window_size])
        targets.append(target[i + window_size])
    return np.array(inputs), np.array(targets)

window_size = 32
inputs, targets = create_time_windows(features_scaled, target_scaled, window_size)


# --- Dataset and splits ---
class TimeSeriesDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

train_size = int(0.6 * len(inputs))
val_size = int(0.2 * len(inputs))

train_dataset = TimeSeriesDataset(inputs[:train_size], targets[:train_size])
val_dataset = TimeSeriesDataset(inputs[train_size:train_size + val_size],
                                 targets[train_size:train_size + val_size])
test_dataset = TimeSeriesDataset(inputs[train_size + val_size:],
                                  targets[train_size + val_size:])

_num_workers = 0 if platform.system() == 'Windows' else min(os.cpu_count() or 1, 4)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=_num_workers)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=_num_workers)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=_num_workers)


# --- FFN model ---
class FFNetwork(L.LightningModule):
    def __init__(self, input_dim, sequence_length, hidden_dim,
                 num_layers=1, output_dim=1, learning_rate=0.0001,
                 dropout_rate=0.5, activation_func=nn.ReLU()):
        super(FFNetwork, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim * sequence_length, hidden_dim)])
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation_func
        self.learning_rate = learning_rate

    def forward(self, x):
        batch_size, sequence_length, input_dim = x.size()
        x = x.view(batch_size, sequence_length * input_dim)
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self.forward(x), y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self.forward(x), y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = inputs.shape[2]
sequence_length = inputs.shape[1]

torch.manual_seed(3527)

model = FFNetwork(
    input_dim=input_dim, sequence_length=sequence_length,
    hidden_dim=63, num_layers=4, output_dim=1,
    learning_rate=0.000689, dropout_rate=0.170971
).to(device)

trainer = L.Trainer(
    accelerator="auto", devices=1, max_epochs=300,
    callbacks=[EarlyStopping(monitor='val_loss', patience=50, mode='min')],
    logger=False, enable_checkpointing=False,
    enable_progress_bar=False, enable_model_summary=False
)
trainer.fit(model, train_loader, val_loader)


# --- Sklearn wrapper for MAPIE ---
class SklearnWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model, window_size, n_features, device='cpu'):
        self.model = model
        self.window_size = window_size
        self.n_features = n_features
        self.device = device
        self.model.to(self.device)

    def fit(self, X, y):
        return self  # model is pre-trained

    def predict(self, X):
        self.model.eval()
        X_3d = X.reshape(-1, self.window_size, self.n_features)
        X_tensor = torch.tensor(X_3d, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions.squeeze()

n_features = inputs.shape[2]
wrapped_model = SklearnWrapper(model, window_size=window_size,
                                n_features=n_features, device='cpu')

def extract_data_from_loader(loader):
    inputs_list, targets_list = [], []
    for batch in loader:
        inp, tgt = batch
        inputs_list.append(inp.cpu().numpy())
        targets_list.append(tgt.cpu().numpy())
    return np.vstack(inputs_list), np.concatenate(targets_list)

X_train, y_train = extract_data_from_loader(train_loader)
X_test, y_test = extract_data_from_loader(test_loader)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
y_train = y_train.ravel()
y_test = y_test.ravel()


# --- EnbPI conformal intervals ---
from mapie.regression import TimeSeriesRegressor
from mapie.subsample import BlockBootstrap

confidence_level = 0.9

cv_mapiets = BlockBootstrap(
    n_resamplings=30, n_blocks=30, overlapping=False, random_state=59)

mapie_enbpi = TimeSeriesRegressor(
    wrapped_model, method="enbpi", cv=cv_mapiets,
    agg_function="mean", n_jobs=1)
mapie_enbpi.fit(X_train, y_train)

y_pred_enbpi, y_pis_enbpi = mapie_enbpi.predict(
    X_test, confidence_level=confidence_level, ensemble=True)


# --- Evaluation ---
y_lower = y_pis_enbpi[:, 0, 0]
y_upper = y_pis_enbpi[:, 1, 0]

in_interval = (y_test >= y_lower) & (y_test <= y_upper)
coverage = float(in_interval.mean())
width = float(np.mean(y_upper - y_lower))

print(f"Target coverage: {confidence_level:.0%}")
print(f"Achieved coverage: {coverage:.3f}")
print(f"Mean interval width: {width:.3f}")


# --- Figure 7.8: EnbPI conformal intervals ---
y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
y_pred_orig = target_scaler.inverse_transform(y_pred_enbpi.reshape(-1, 1)).ravel()
y_lower_orig = target_scaler.inverse_transform(y_pis_enbpi[:, 0, 0].reshape(-1, 1)).ravel()
y_upper_orig = target_scaler.inverse_transform(y_pis_enbpi[:, 1, 0].reshape(-1, 1)).ravel()

in_interval = (y_test_orig >= y_lower_orig) & (y_test_orig <= y_upper_orig)

fig, ax = plt.subplots(figsize=(CFG.img_dim1, CFG.img_dim2))
ax.plot(y_test_orig, color=custom_palette[0], alpha=0.7, label="Actual")
ax.plot(y_pred_orig, color=custom_palette[1], label="Predicted")
ax.fill_between(range(len(y_test_orig)), y_lower_orig, y_upper_orig,
                alpha=0.2, color=custom_palette[5],
                label=f"{confidence_level:.0%} EnbPI interval")
ax.scatter(np.where(~in_interval)[0], y_test_orig[~in_interval],
           color=custom_palette[2], marker='x', s=60, linewidths=2,
           zorder=5, label="Missed")
ax.set_xlabel("Test time step")
ax.set_ylabel("Sales")
ax.set_title(f"EnbPI conformal intervals (coverage={coverage:.2f})")
ax.legend()
plt.tight_layout()
plt.show()