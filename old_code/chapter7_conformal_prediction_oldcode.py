# Extracted from chapter7_Conformal_prediction.qmd
# Do not edit the source .qmd file directly.

#| label: Chapter 7 libraries
#| message: false
#| echo: false
#| eval: true

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.pyplot import figure

import matplotlib.colors as mcolors
from matplotlib.patches import Patch
plt.style.use('seaborn-v0_8')

import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Get working directory
cwd = os.getcwd()

# Get parent directory
parent_dir = os.path.dirname(cwd)

# Add parent directory to system path
sys.path.insert(0, parent_dir)


# Define palette
custom_palette = ["#000000", "#0072B2", "#D55E00","#009E73","#CC79A7", "#56B4E9","#E69F00"]
line_styles = ['-', '--', '-.', ':']

plt.rcParams['figure.figsize'] = (6, 12)
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

#| label: fig-point-prediction-error
#| fig-cap: "Point prediction relative to actual value. The vertical arrow shows the distance (error) between the model's prediction and the true observed value."
#| fig-alt: "Scatter plot showing a linear regression line with a predicted point and true value, and an arrow indicating the absolute error between them"
#| message: false
#| echo: false
#| eval: true

# Create linear data
x = np.linspace(0.1, 0.9, 100)
y = 2 * x + 1  

# Predicted point
x_pred = 0.6
y_pred = 2 * x_pred + 1

# True value for demonstration
y_true = 2.5

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, label='Linear prediction (Model)', color=custom_palette[0], linestyle='--')  # Model line
ax.scatter([x_pred], [y_pred], color=custom_palette[1], zorder=5)  # Predicted point
ax.scatter([x_pred], [y_true + 0.12], color=custom_palette[2], zorder=5)  # True value

# Annotating true and predicted value
ax.annotate('Predicted Value', (x_pred, y_pred), textcoords="offset points", xytext=(-57,-3), ha='center', color=custom_palette[1])
ax.annotate('True Value', (x_pred, y_true), textcoords="offset points", xytext=(-42,15), ha='center', color=custom_palette[2])

# Draw error arrow
ax.arrow(x_pred, y_pred, 0, y_true - y_pred, head_width=0.02, head_length=0.1, fc=custom_palette[1], ec=custom_palette[2])

# Labeling axes and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Point Prediction relative to actual value')
ax.legend()

plt.grid(True)
plt.show()

# ----------------------------------------------------------------------

#| label: fig-uncertainty-types
#| fig-cap: "Aleatoric and epistemic uncertainty. The left cluster shows high aleatoric uncertainty from noise around a linear model. The right cluster shows low aleatoric uncertainty. Voids with no data represent high epistemic uncertainty."
#| fig-alt: "Scatter plot showing two data clusters with differing noise levels and annotated regions of high epistemic uncertainty"
#| message: false
#| echo: false
#| eval: true

# Synthetic data
np.random.seed(0)
x_high_aleatoric = np.linspace(-4, -2, 50)
y_high_aleatoric = 0.3 * x_high_aleatoric + np.random.normal(loc=0, scale=1.5, size=50)

x_low_aleatoric = np.linspace(2, 4, 50)
y_low_aleatoric = 0.5 * x_low_aleatoric + np.random.normal(loc=0, scale=0.3, size=50)

x_line = np.linspace(-10, 10, 200)
y_line = 0.5 * x_line  # Real function

plt.style.use('seaborn-v0_8-ticks')
fig, ax = plt.subplots(figsize=(CFG.img_dim1, CFG.img_dim2))
ax.scatter(x_high_aleatoric, y_high_aleatoric, color=custom_palette[5], label='High Aleatoric \n Uncertainty')
ax.scatter(x_low_aleatoric, y_low_aleatoric, color=custom_palette[6], alpha=0.6, label='Aleatoric \n Uncertainty')

# Plotting real function
ax.plot(x_line, y_line, 'k--', label='Real Function')

# Annotating areas of uncertainty
ax.annotate('High Aleatoric \n Uncertainty', xy=(-3.0, 1.5), xytext=(-7.5, 4.5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'), fontsize=14)
ax.annotate('Low Aleatoric \n Uncertainty', xy=(2.5, 1.5), xytext=(2.5, 4.5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'), fontsize=14)
ax.annotate('High Epistemic \n Uncertainty', xy=(-7.5, -4), xytext=(-5, -5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'), fontsize=14)
ax.annotate('High Epistemic \n Uncertainty', xy=(6.5, 3), xytext=(5, -5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'), fontsize=14)

# Labeling and aesthetics
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Data with Uncertainty \n')
ax.legend(loc='right', fontsize=12)

plt.show()

# ----------------------------------------------------------------------

#| label: fig-regression-nonconformity
#| fig-cap: "Distance (absolute error) from a linear model. Two predicted points are highlighted: one close to the regression line with low absolute error (green), and one far from it with high absolute error (red)."
#| fig-alt: "Scatter plot with regression line showing two predicted points and their absolute errors as dashed vertical lines"
#| message: false
#| echo: false
#| eval: true

# linear regression
plt.style.use('seaborn-v0_8-ticks')
np.random.seed(235987)
x = np.linspace(0, 10, 30)
y = 2 * x + 1 + np.random.normal(scale=2, size=30)

# Fit regression 
p, residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
fitted_y = np.poly1d(p)(x)

# calculate Abs error
absolute_errors = np.abs(y - fitted_y)

# Find indices and select based on rank
sorted_indices = np.argsort(absolute_errors)
point_near_index = sorted_indices[3]  # 4th lowest AE
point_far_index = sorted_indices[-2]  # 2nd highest AE

# Get idx values for highlighting
point_near = (x[point_near_index], y[point_near_index])
point_far = (x[point_far_index], y[point_far_index])
fitted_near = fitted_y[point_near_index]
fitted_far = fitted_y[point_far_index]

# calc AE for highlighted points
ae_near = absolute_errors[point_near_index]
ae_far = absolute_errors[point_far_index]

# Plotting
fig, ax = plt.subplots()
ax.plot(x, y, 'o', label='Data points', color=custom_palette[6])
ax.plot(x, fitted_y, '--', label='Regression Line', color=custom_palette[1])
ax.scatter(*point_near, color=custom_palette[3], label='Predicted point (low AE)', zorder=5)
ax.scatter(*point_far, color=custom_palette[2], label='Predicted point (high AE)', zorder=5)

# Draw AE lines
ax.plot([point_near[0], point_near[0]], [point_near[1], fitted_near], 'g--')
ax.plot([point_far[0], point_far[0]], [point_far[1], fitted_far], 'r--')

# Annotation etc
ax.text(point_near[0] - 0.1, (point_near[1] + fitted_near) / 1.7, f'AE: {ae_near:.2f}', ha='right')
ax.text(point_far[0] -0.1, (point_far[1] + fitted_far) / 1.85, f'AE: {ae_far:.2f}', ha='right')
ax.set_xlabel('X values (i.e. chocolate sales)')
ax.set_ylabel('Y values (i.e. Field medals)')
ax.set_title('Distance (absolute error) from linear model')
ax.legend()
plt.grid(True)
plt.show()

# ----------------------------------------------------------------------

#| label: fig-conformity-ladder-data
#| message: false
#| echo: false
#| eval: true

# Data
np.random.seed(42)
x = np.random.normal(50, 15, 20)
y_noisy = x * 3 + np.random.normal(0, 20, 20) 

# OLS regression 
model_noisy = LinearRegression().fit(x.reshape(-1,1), y_noisy)
y_pred_noisy = model_noisy.predict(x.reshape(-1, 1))
abs_errors_noisy = np.abs(y_pred_noisy - y_noisy)

# Increase sales value of one point 
x_modified = x.copy()
y_modified = y_noisy.copy()
y_modified[10] += 50  # 11th data has 50 added to it

# We recalculate preds and errors 
y_pred_modified = model_noisy.predict(x_modified.reshape(-1, 1))
abs_errors_modified = np.abs(y_pred_modified - y_modified)

# Calc conformity scores for both scenarios


conformity_scores_noisy = 1 - np.argsort(np.argsort(abs_errors_noisy)) / len(abs_errors_noisy)
conformity_scores_modified = 1 - np.argsort(np.argsort(abs_errors_modified)) / len(abs_errors_modified)

# Adj colours for ladder ranking (green down to red) based on conformity scores
sorted_conformity_scores_noisy = np.sort(conformity_scores_noisy)
sorted_conformity_scores_modified = np.sort(conformity_scores_modified)

sorted_indices_noisy = np.argsort(-conformity_scores_noisy)
sorted_indices_modified = np.argsort(-conformity_scores_modified)

colors_noisy = [sns.color_palette("RdYlGn_r", n_colors=len(abs_errors_noisy))[i] for i in range(len(abs_errors_noisy))]
colors_modified = [sns.color_palette("RdYlGn_r", n_colors=len(abs_errors_modified))[i] for i in range(len(abs_errors_modified))]

# Highlighting the modified point in the ladder
highlight_index = 10
highlight_color = "violet"

# ----------------------------------------------------------------------

#| label: fig-conformity-ladder
#| fig-cap: "Conformity ranking ladders for original and modified data. Top row: a data point (violet) sits near the regression line and ranks highly. Bottom row: the same point is shifted to an extreme position, dropping its ranking and shifting all others."
#| fig-alt: "Four-panel figure comparing scatter plots with regression lines alongside horizontal bar charts of conformity rankings"
#| message: false
#| echo: false
#| eval: true

# Plotting the corrected visualizations with highlights
fig, ax = plt.subplots(2, 2, figsize=(18, 16))

# Original noisy data with regression line
sns.scatterplot(x=x, y=y_noisy, ax=ax[0,0], color="black", label="Data Points")
sns.scatterplot(x=[x[10]], y=[y_noisy[10]], ax=ax[0,0], color="violet", s=100, label="Predicted value")
sns.lineplot(x=x, y=y_pred_noisy, ax=ax[0,0], color="blue", label="Regression line average predicted value")
ax[0,0].set_title("Original Noisy Data with Regression Line")
ax[0,0].set_xlabel("Price discount", fontsize=16)
ax[0,0].set_ylabel("Sales", fontsize=16)
ax[0,0].legend(fontsize=16)

# Conformity ranking ladder for original noisy data
bar_colors_noisy = colors_noisy.copy()
bar_colors_noisy[sorted_indices_noisy.tolist().index(highlight_index)] = highlight_color
ax[0,1].barh(np.arange(len(conformity_scores_noisy)), abs_errors_noisy[sorted_indices_noisy], color=bar_colors_noisy)
ax[0,1].set_yticks(np.arange(len(conformity_scores_noisy)))
ax[0,1].set_yticklabels(np.round(conformity_scores_noisy[sorted_indices_noisy], 2))
ax[0,1].set_title("Conformity Ranking Ladder for Original Data")
ax[0,1].set_xlabel("Non-Conformity (Absolute Error)")
ax[0,1].set_ylabel("Conformity Ranking")
ax[0,1].invert_yaxis()

# Modified data with regression line
sns.scatterplot(x=x_modified, y=y_modified, ax=ax[1,0], color="black", label="Data Points")
sns.scatterplot(x=[x_modified[10]], y=[y_modified[10]], ax=ax[1,0], color="violet", s=100, label="Predicted Point")
sns.lineplot(x=x_modified, y=y_pred_modified, ax=ax[1,0], color="blue", label="Regression line extreme predicted value")
ax[1,0].set_title("Modified Data with Regression Line")
ax[1,0].set_xlabel("Price discount", fontsize=16)
ax[1,0].set_ylabel("Sales", fontsize=16)
ax[1,0].legend(fontsize=16)

# Conformity ranking ladder for modified data
bar_colors_modified = colors_modified.copy()
bar_colors_modified[sorted_indices_modified.tolist().index(highlight_index)] = highlight_color
ax[1,1].barh(np.arange(len(conformity_scores_modified)), abs_errors_modified[sorted_indices_modified], color=bar_colors_modified)
ax[1,1].set_yticks(np.arange(len(conformity_scores_modified)))
ax[1,1].set_yticklabels(np.round(conformity_scores_modified[sorted_indices_modified], 2))
ax[1,1].set_title("Conformity Ranking Ladder for Modified Data")
ax[1,1].set_xlabel("Non-Conformity (Absolute Error)")
ax[1,1].set_ylabel("Conformity Ranking")
ax[1,1].invert_yaxis()

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: fig-full-conformal-refit
#| fig-cap: "The refitting problem in full conformal prediction. Top row: a data point (violet) sits at a reasonable position with a stable polynomial fit. Bottom row: the same point is shifted to an extreme position, distorting the fit and reshuffling the conformity ladder."
#| fig-alt: "Four-panel figure showing how moving a validation point changes the polynomial regression fit and conformity rankings"
#| message: false
#| echo: false
#| eval: true

np.random.seed(39285)
plt.rcParams['font.size'] = 20  
plt.rcParams['axes.labelsize'] = 20  
plt.rcParams['axes.titlesize'] = 20  
plt.rcParams['xtick.labelsize'] = 20 
plt.rcParams['ytick.labelsize'] = 20 

# synthetic data
x = np.random.normal(50, 15, 20)
y_noisy = x * 3 + np.random.normal(0, 20, 20)

# Polynomial transformation
poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x.reshape(-1,1))

# Polynomial regression model
model_noisy = LinearRegression().fit(x_poly, y_noisy)
y_pred_noisy = model_noisy.predict(x_poly)
abs_errors_noisy = np.abs(y_pred_noisy - y_noisy)

# Identify highest x value index
highest_x_index = np.argmax(x)

# Modify y-value associated with highest x value
x_modified = x.copy()
y_modified = y_noisy.copy()
adjustment_value = 120  
y_modified[highest_x_index] += adjustment_value  # Add to make it higher

# Fit polynomial regression model on modified data
x_poly_modified = poly.transform(x_modified.reshape(-1, 1))
model_modified = LinearRegression().fit(x_poly_modified, y_modified)
y_pred_modified = model_modified.predict(x_poly_modified)
abs_errors_modified = np.abs(y_pred_modified - y_modified)

# Calculate conformity values
conformity_scores_noisy = np.argsort(np.argsort(abs_errors_noisy)) / (len(abs_errors_noisy) - 1)
conformity_scores_modified = np.argsort(np.argsort(abs_errors_modified)) / (len(abs_errors_modified) - 1)

# Colors for conformity ranking
norm_noisy = plt.Normalize(conformity_scores_noisy.min(), conformity_scores_noisy.max())
sorted_indices_noisy = np.argsort(conformity_scores_noisy)
colors_noisy = plt.cm.RdYlGn_r(norm_noisy(conformity_scores_noisy[sorted_indices_noisy]))

norm_modified = plt.Normalize(conformity_scores_modified.min(), conformity_scores_modified.max())
sorted_indices_modified = np.argsort(conformity_scores_modified)
colors_modified = plt.cm.RdYlGn_r(norm_modified(conformity_scores_modified[sorted_indices_modified]))

highlight_index = highest_x_index  
highlight_color = "violet"  

# Plotting
fig, ax = plt.subplots(2, 2, figsize=(18, 16))

# Regression line with reasonable prediction
sns.scatterplot(x=x, y=y_noisy, ax=ax[0,0], color="black", label="Data Points")
sns.scatterplot(x=[x[highest_x_index]], y=[y_noisy[highest_x_index]], ax=ax[0,0], color="violet", s=100, label="High Price Data Point")
sns.lineplot(x=np.sort(x), y=model_noisy.predict(poly.transform(np.sort(x).reshape(-1, 1))), ax=ax[0,0], color="blue", label="Polynomial Regression Line")
ax[0,0].set_title("Regression line with reasonable prediction")
ax[0,0].set_xlabel("Price discount")
ax[0,0].set_ylabel("Sales")
ax[0,0].legend(fontsize=16)

# Conformity ladder for reasonable prediction
bar_colors_noisy = [highlight_color if i == highlight_index else colors_noisy[j] for j, i in enumerate(sorted_indices_noisy)]
ax[0,1].barh(np.arange(len(conformity_scores_noisy)), abs_errors_noisy[sorted_indices_noisy], color=bar_colors_noisy)
ax[0,1].set_yticks(np.arange(len(conformity_scores_noisy)))
ax[0,1].set_yticklabels(np.round(conformity_scores_noisy[sorted_indices_noisy], 2))
ax[0,1].set_title("Conformity ladder with reasonable prediction")
ax[0,1].set_xlabel("Non-Conformity (Absolute Error)")
ax[0,1].set_ylabel("Conformity Ranking")
ax[0,1].invert_yaxis()

# Regression line with extreme prediction
sns.scatterplot(x=x_modified, y=y_modified, ax=ax[1,0], color="black", label="Data Points")
sns.scatterplot(x=[x_modified[highest_x_index]], y=[y_modified[highest_x_index]], ax=ax[1,0], color="violet", s=100, label="Modified High Price Data Point")
sns.lineplot(x=np.sort(x_modified), y=model_modified.predict(poly.transform(np.sort(x_modified).reshape(-1, 1))), ax=ax[1,0], color="blue", label="New Polynomial Regression Line")
ax[1,0].set_title("Regression line with extreme prediction")
ax[1,0].set_xlabel("Price discount")
ax[1,0].set_ylabel("Sales")
ax[1,0].legend(fontsize=16)

# Conformity ladder for extreme prediction
bar_colors_modified = [highlight_color if i == highlight_index else colors_modified[j] for j, i in enumerate(sorted_indices_modified)]
ax[1,1].barh(np.arange(len(conformity_scores_modified)), abs_errors_modified[sorted_indices_modified], color=bar_colors_modified)
ax[1,1].set_yticks(np.arange(len(conformity_scores_modified)))
ax[1,1].set_yticklabels(np.round(conformity_scores_modified[sorted_indices_modified], 2))
ax[1,1].set_title("Conformity Ladder with extreme prediction")
ax[1,1].set_xlabel("Non-Conformity (Absolute Error)")
ax[1,1].set_ylabel("Conformity Ranking")
ax[1,1].invert_yaxis()

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: fig-split-conformal
#| fig-cap: "Split conformal method. Training data (red crosses) is used to fit the polynomial regression, while calibration data (blue circles) is held out to compute non-conformity scores and build the conformity ladder."
#| fig-alt: "Two-panel figure showing training and calibration data with polynomial fit alongside the calibration conformity ladder"
#| message: false
#| echo: false
#| eval: true
from sklearn.model_selection import train_test_split

np.random.seed(39285)

plt.rcParams['font.size'] = 20  
plt.rcParams['axes.labelsize'] = 20  
plt.rcParams['axes.titlesize'] = 20  
plt.rcParams['xtick.labelsize'] = 20 
plt.rcParams['ytick.labelsize'] = 20 

# synthetic data
x = np.random.normal(50, 15, 20)  
y = x * 3 + np.random.normal(0, 20, 20)

# Split data into training and calibration sets
x_train, x_calib, y_train, y_calib = train_test_split(x, y, test_size=0.4, random_state=42)

# Polynomial transformation for both sets
poly = PolynomialFeatures(degree=3)
x_train_poly = poly.fit_transform(x_train.reshape(-1,1))
x_calib_poly = poly.transform(x_calib.reshape(-1, 1))

# Fit polynomial regression model using training data
model = LinearRegression().fit(x_train_poly, y_train)

# Predict on both training and calibration sets
y_train_pred = model.predict(x_train_poly)
y_calib_pred = model.predict(x_calib_poly)

# Calculate absolute errors on calibration data to form conformity scores
abs_errors_calib = np.abs(y_calib_pred - y_calib)

# Sort ranking for conformity scores: 0 for lowest AE and 1 for highest AE
ranks = np.argsort(np.argsort(abs_errors_calib))
conformity_scores = ranks / (len(abs_errors_calib) - 1)

# Sorting indices for calibration data based on conformity scores
sorted_indices_calib = np.argsort(conformity_scores)  

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

# Training data with polynomial regression line
sns.scatterplot(x=x_train, y=y_train, ax=axs[0], marker='x', color=custom_palette[2], s=100, label="Training Data")
sns.scatterplot(x=x_calib, y=y_calib, ax=axs[0], marker='o', color=custom_palette[1], s=100, label="Calibration Data")
sns.lineplot(x=np.linspace(min(x), max(x), 100), y=model.predict(poly.transform(np.linspace(min(x), max(x), 100).reshape(-1, 1))), ax=axs[0], color="green", label="Polynomial Regression Line")
axs[0].set_title("Training and Calibration Data with Polynomial Fit")
axs[0].set_xlabel("Price discount")
axs[0].set_ylabel("Sales")
axs[0].legend(fontsize = 16)

# Conformity ladder for calibration data
norm = plt.Normalize(conformity_scores.min(), conformity_scores.max())
colors = plt.cm.RdYlGn_r(norm(conformity_scores[sorted_indices_calib]))  # Use reversed colormap
axs[1].barh(np.arange(len(conformity_scores)), abs_errors_calib[sorted_indices_calib], color=colors)
axs[1].set_yticks(np.arange(len(conformity_scores)))
axs[1].set_yticklabels(np.round(conformity_scores[sorted_indices_calib], 2))
axs[1].set_title("Conformity Ladder for Calibration Data")
axs[1].set_xlabel("Non-Conformity (Absolute Error)")
axs[1].set_ylabel("Conformity Ranking")
axs[1].invert_yaxis() 

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: fig-quartile-intervals
#| fig-cap: "Conformity ladder divided into quartile sub-intervals. Red dashed lines mark the boundaries at confidence levels 0.2, 0.4, 0.6, and 0.8."
#| fig-alt: "Horizontal bar chart of conformity scores with red dashed lines marking quartile boundaries"
#| message: false
#| echo: false
#| eval: true

# synthetic data
x = np.random.normal(50, 15, 30)  
y = x * 3 + np.random.normal(0, 20, 30)

# Split data into training and calibration sets
x_train, x_calib, y_train, y_calib = train_test_split(x, y, test_size=0.4, random_state=42)

# Polynomial transformation for both sets
poly = PolynomialFeatures(degree=3)
x_train_poly = poly.fit_transform(x_train.reshape(-1, 1))
x_calib_poly = poly.transform(x_calib.reshape(-1, 1))

# Fit polynomial regression model using only training data
model = LinearRegression().fit(x_train_poly, y_train)

# Predict on both training and calibration sets
y_train_pred = model.predict(x_train_poly)
y_calib_pred = model.predict(x_calib_poly)

# Calculate absolute errors on calibration data to form conformity scores
abs_errors_calib = np.abs(y_calib_pred - y_calib)

# Conformity scores: higher score = higher error = less conforming
# Note: this convention is inverted from earlier ladder plots,
# where we used (1 - rank/n) so that higher = more conforming.
# Both are valid; here we use the raw rank for the quartile construction.
ranks = np.argsort(np.argsort(abs_errors_calib))
conformity_scores = ranks / (len(abs_errors_calib) - 1)

# Sorting indices for calibration data based on conformity scores
sorted_indices_calib = np.argsort(conformity_scores) 

# Calculate quartiles - set quartiles here
quartile_bounds = np.percentile(conformity_scores, [20, 40, 60, 80]) 

# Conformity ladder for calibration data
fig, ax = plt.subplots(figsize=(10, 8))
norm = plt.Normalize(conformity_scores.min(), conformity_scores.max())
colors = plt.cm.RdYlGn_r(norm(conformity_scores[sorted_indices_calib])) 
bars = ax.barh(np.arange(len(conformity_scores)), abs_errors_calib[sorted_indices_calib], color=colors)
ax.set_yticks(np.arange(len(conformity_scores)))
ax.set_yticklabels(np.round(conformity_scores[sorted_indices_calib], 2))
ax.set_title("Conformity Ladder for Calibration Data")
ax.set_xlabel("Non-Conformity (Absolute Error)")
ax.set_ylabel("Conformity Ranking")
ax.invert_yaxis()  

# Draw lines for quartile boundaries
for quartile in quartile_bounds:
    quartile_index = np.where(np.sort(conformity_scores) >= quartile)[0][0]
    quartile_position = len(conformity_scores) - quartile_index
    ax.axhline(y=quartile_position, color='r', linestyle='--')

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: enbpi-data-loading
#| message: false
#| echo: true
#| eval: true

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv(CFG.data_folder / 'pseudo_sales.csv')
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

# ----------------------------------------------------------------------

#| label: enbpi-dataset-splits
#| message: false
#| echo: true
#| eval: true

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
val_dataset = TimeSeriesDataset(inputs[train_size:train_size + val_size], targets[train_size:train_size + val_size])
test_dataset = TimeSeriesDataset(inputs[train_size + val_size:], targets[train_size + val_size:])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ----------------------------------------------------------------------

#| label: enbpi-model-training
#| message: false
#| echo: true
#| eval: true

class FFNetwork(L.LightningModule):
    def __init__(self, input_dim, sequence_length, hidden_dim, 
                 num_layers=1, output_dim=1, learning_rate=0.0001, 
                 dropout_rate=0.5, activation_func=nn.ReLU()):
        super(FFNetwork, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim * sequence_length, hidden_dim)]
        )
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

input_dim = inputs.shape[2]
sequence_length = inputs.shape[1]

torch.manual_seed(755863)

model = FFNetwork(
    input_dim=input_dim, sequence_length=sequence_length,
    hidden_dim=63, num_layers=4, output_dim=1,
    learning_rate=0.000689, dropout_rate=0.170971
).to(device)

trainer = L.Trainer(
    accelerator="auto", devices=1, max_epochs=300,
    callbacks=[EarlyStopping(monitor='val_loss', patience=50, mode='min')],
    logger=False,
    enable_progress_bar=False,
    enable_model_summary=False
)
trainer.fit(model, train_loader, val_loader)

# ----------------------------------------------------------------------

#| label: enbpi-sklearn-wrapper
#| message: false
#| echo: true
#| eval: true
 
from sklearn.base import BaseEstimator, RegressorMixin
 
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
        # MAPIE passes flattened 2D (samples, window*features)
        # FFNetwork expects 3D (samples, window, features)
        X_3d = X.reshape(-1, self.window_size, self.n_features)
        X_tensor = torch.tensor(X_3d, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions.squeeze()
 
n_features = inputs.shape[2]
wrapped_model = SklearnWrapper(
    model, window_size=window_size, 
    n_features=n_features, device='cpu'
)
 
def extract_data_from_loader(loader):
    inputs_list, targets_list = [], []
    for batch in loader:
        inp, tgt = batch
        inputs_list.append(inp.cpu().numpy())
        targets_list.append(tgt.cpu().numpy())
    return np.vstack(inputs_list), np.concatenate(targets_list)
 
X_train, y_train = extract_data_from_loader(train_loader)
X_test, y_test = extract_data_from_loader(test_loader)
 
# Flatten windowed features for MAPIE's sklearn interface
# (samples, window_size, features) → (samples, window_size * features)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
y_train = y_train.ravel()
y_test = y_test.ravel()

# ----------------------------------------------------------------------

#| label: enbpi-conformal-intervals
#| message: false
#| echo: true
#| eval: true
 
from mapie.regression import TimeSeriesRegressor
from mapie.subsample import BlockBootstrap
 
confidence_level = 0.9  # 90% prediction intervals
 
cv_mapiets = BlockBootstrap(
    n_resamplings=30, n_blocks=30, overlapping=False, random_state=59
)
mapie_enbpi = TimeSeriesRegressor(
    wrapped_model, method="enbpi", cv=cv_mapiets, 
    agg_function="mean", n_jobs=1
)
mapie_enbpi.fit(X_train, y_train)
 
y_pred_enbpi, y_pis_enbpi = mapie_enbpi.predict(
    X_test, confidence_level=confidence_level, ensemble=True
)

# ----------------------------------------------------------------------

#| label: enbpi-evaluation
#| message: false
#| echo: true
#| eval: true
 
y_lower = y_pis_enbpi[:, 0, 0] 
y_upper = y_pis_enbpi[:, 1, 0]

in_interval = (y_test >= y_lower) & (y_test <= y_upper)
coverage = float(in_interval.mean())
width = float(np.mean(y_upper - y_lower))

print(f"Target coverage: {confidence_level:.0%}")
print(f"Achieved coverage: {coverage:.3f}")
print(f"Mean interval width: {width:.3f}")

# ----------------------------------------------------------------------

#| label: fig-enbpi-results
#| fig-cap: "EnbPI conformal intervals on the pseudo sales test set. The 90% prediction 
#|   interval is shaded, with points falling outside marked in orange. Misses cluster at 
#|   peaks where the model systematically underpredicts."
#| fig-alt: "Time series plot showing actual vs predicted sales with a shaded 90% 
#|   conformal prediction interval and orange markers for missed points"
#| message: false
#| echo: false
#| eval: true

y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
y_pred_orig = target_scaler.inverse_transform(y_pred_enbpi.reshape(-1, 1)).ravel()
y_lower_orig = target_scaler.inverse_transform(y_pis_enbpi[:, 0, 0].reshape(-1, 1)).ravel()
y_upper_orig = target_scaler.inverse_transform(y_pis_enbpi[:, 1, 0].reshape(-1, 1)).ravel()

in_interval = (y_test_orig >= y_lower_orig) & (y_test_orig <= y_upper_orig)

fig, ax = plt.subplots(figsize=(CFG.img_dim1, CFG.img_dim2))

ax.plot(y_test_orig, color=custom_palette[0], alpha=0.7, label="Actual")
ax.plot(y_pred_orig, color=custom_palette[1], label="Predicted")
ax.fill_between(
    range(len(y_test_orig)),
    y_lower_orig, y_upper_orig,
    alpha=0.2, color=custom_palette[5],
    label=f"{confidence_level:.0%} EnbPI interval"
)
ax.scatter(
    np.where(~in_interval)[0], y_test_orig[~in_interval],
    color=custom_palette[2], s=20, zorder=5, label="Missed"
)
ax.set_xlabel("Test time step")
ax.set_ylabel("Sales")
ax.set_title(f"EnbPI conformal intervals (coverage={coverage:.2f})")
ax.legend()
plt.tight_layout()
plt.show()