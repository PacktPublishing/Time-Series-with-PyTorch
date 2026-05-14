# =============================================================================
# Chapter 10: Other Neural Structures
# =============================================================================
# Code companion for Chapter 10
# Covers: M5 data loading/analysis/visualisation, N-BEATS basis expansion
#         intuition, N-HiTS MaxPool demonstration, foundation model phase
#         space diagram (Lag-Llama intuition)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

class CFG:
    data_folder = Path.cwd()  / "Data" # / "Time-Series-with-PyTorch"
    img_dim1 = 12
    img_dim2 = 6
    fontsize = 18

plt.rcParams.update({'figure.figsize': (CFG.img_dim1, CFG.img_dim2)})


# =============================================================================
# 10.1  Loading and Preprocessing M5 Data
# =============================================================================

def load_m5_subset(data_folder: Path) -> pd.DataFrame:
    filepath = data_folder / 'M5_t20_ABC.csv'
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found at {filepath}")

    df = pd.read_csv(filepath, index_col=0)
    df['date'] = pd.to_datetime(df['date'])

    # Remove last 28 days of each series
    df_sorted = df.sort_values(['item_id', 'date'])
    def remove_tail_values(group):
        return group.iloc[:-28]
    df = df_sorted.groupby('item_id', group_keys=False).apply(
        remove_tail_values).reset_index(drop=True)

    # Add time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week
    return df


# =============================================================================
# 10.2  Dataset Analysis
# =============================================================================

def analyze_dataset(df: pd.DataFrame) -> dict:
    abc_dist = df.groupby('item_id')['ABC_class'].first().value_counts().to_dict()
    analysis = {
        'n_series': df['item_id'].nunique(),
        'n_stores': df['state_id'].nunique(),
        'date_range': (df['date'].min(), df['date'].max()),
        'total_sales': df['sold'].sum(),
        'mean_price': df['sell_price'].mean(),
        'abc_distribution': abc_dist,
        'sales_by_class': df.groupby('ABC_class')['sold'].sum().to_dict(),
        'avg_price_by_class': df.groupby('ABC_class')['sell_price'].mean().to_dict()
    }
    return analysis


def plot_sales_analysis(series: pd.DataFrame, figsize=(15, 10)):
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    axes[0].plot(series['date'], series['sold'], color=custom_palette[1], alpha=0.7)
    axes[0].set_title(f"Sales Over Time for {series['item_id'].iloc[0]}")
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Units Sold')

    sns.boxplot(data=series, x='dayofweek', y='sold', ax=axes[1],
                color=custom_palette[2])
    axes[1].set_title('Sales Distribution by Day of Week')
    axes[1].set_xlabel('Day of Week')
    axes[1].set_ylabel('Units Sold')

    plt.tight_layout()
    plt.show()

    print(f"\nSummary Statistics: {series['sold'].describe()}")


# Load and analyse
df = load_m5_subset(CFG.data_folder)
analysis = analyze_dataset(df)
for k, v in analysis.items():
    print(f"{k}: {v}")


# =============================================================================
# 10.3  Exploring Individual Series
# =============================================================================

series_to_plot = ['FOODS_3_090', 'FOODS_3_714', 'HOUSEHOLD_2_437', 'HOUSEHOLD_1_448']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.tight_layout(pad=3.0)
axes = axes.flatten()

for idx, (series_id, ax) in enumerate(zip(series_to_plot, axes)):
    series = df[df['item_id'] == series_id].copy()
    series = series.sort_values('date')

    last_date = series['date'].max()
    start_date = last_date - pd.DateOffset(months=36)
    series = series[series['date'] >= start_date]

    ax.plot(series['date'], series['sold'],
            color=custom_palette[idx], alpha=0.7)
    ax.set_title(f"Sales for {series_id}", fontsize=14)
    ax.set_xlabel('Date', fontsize=16)
    ax.set_ylabel('Units Sold', fontsize=16)
    ax.tick_params(axis='x', rotation=45)

plt.subplots_adjust(hspace=0.4, wspace=0.2)
plt.show()


# =============================================================================
# 10.4  N-BEATS — Basis Expansion Intuition
# =============================================================================

# Linear regression on temperature vs ice cream sales
temp = np.array([14, 16, 18, 20, 22, 24, 26, 28, 30]).reshape((-1, 1))
sales = np.array([50, 55, 60, 80, 110, 160, 180, 190, 200])

model = LinearRegression()
model.fit(temp, sales)

# --- Figure: Linear fit ---
plt.scatter(temp, sales, color=custom_palette[0])
plt.plot(temp, model.predict(temp), color=custom_palette[1])
plt.xlabel('Temperature')
plt.ylabel('Sales')
plt.title('Linear Regression: Temperature vs Ice Cream Sales')
plt.tight_layout(pad=3.0)
plt.show()


# --- Figure: Polynomial basis expansion ---
poly = PolynomialFeatures(degree=3, include_bias=False)
temp_poly = poly.fit_transform(temp)

model = LinearRegression()
model.fit(temp_poly, sales)

plt.scatter(temp, sales, color=custom_palette[0])
plt.plot(temp, model.predict(temp_poly), color=custom_palette[1])
plt.xlabel('Temperature')
plt.ylabel('Sales')
plt.title('Polynomial (degree=3) Basis Expansion')
plt.show()


# =============================================================================
# 10.5  N-HiTS — MaxPool Demonstration
# =============================================================================

data = np.array([3, 6, 4, 0, 8, 5, 3, 2, 7, 9, 6, 0, 5, 3, 2])
x = np.arange(len(data))

def max_pool(data, kernel_size):
    output = []
    for i in range(0, len(data), kernel_size):
        end = min(i + kernel_size, len(data))
        output.append(np.max(data[i:end]))
    return np.array(output)

pool_3 = max_pool(data, 3)
pool_5 = max_pool(data, 5)

x_pool_3 = np.arange(1, len(pool_3) * 3, 3)[:len(pool_3)]
x_pool_5 = np.arange(2, len(pool_5) * 5, 5)[:len(pool_5)]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

# Top: original data with pooling windows
ax1.plot(x, data, 'o-', color=custom_palette[0], linewidth=1.5, markersize=6, zorder=5)

colours_k3 = [custom_palette[1], custom_palette[5]]
for idx, i in enumerate(range(0, len(data), 3)):
    end = min(i + 3, len(data)) - 1
    ax1.axvspan(i - 0.5, end + 0.5, alpha=0.15, color=colours_k3[idx % 2],
                label='Kernel Size 3 windows' if idx == 0 else None)

bar_y = 9.8
for idx, i in enumerate(range(0, len(data), 5)):
    end = min(i + 5, len(data)) - 1
    ax1.plot([i, end], [bar_y, bar_y], color=custom_palette[6], linewidth=4,
             solid_capstyle='butt',
             label='Kernel Size 5 windows' if idx == 0 else None)
    ax1.plot([i, i], [bar_y - 0.3, bar_y + 0.3], color=custom_palette[6], linewidth=2)
    ax1.plot([end, end], [bar_y - 0.3, bar_y + 0.3], color=custom_palette[6], linewidth=2)

ax1.set_ylim(-0.5, 10.5)
ax1.set_title('Original Time Series with Pooling Windows')
ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)

# Bottom: pooled results
ax2.plot(x_pool_3, pool_3, 'o-', label='Kernel Size 3',
         color=custom_palette[1], linewidth=2)
ax2.plot(x_pool_5, pool_5, 'o-', label='Kernel Size 5',
         color=custom_palette[6], linewidth=2)
ax2.legend(loc='upper right', fontsize=11)
ax2.set_title('Max Pooled Results')

plt.tight_layout()
plt.show()


# =============================================================================
# 10.6  Foundation Models — Phase Space Diagram (Lag-Llama Intuition)
# =============================================================================

# The swing angle dataset lives alongside the chapter data
swing_angle_path = CFG.data_folder / 'swing_angle_ts.csv'
if not swing_angle_path.exists():
    # Try chapter-specific location
    swing_angle_path = (Path.cwd() / "Time-Series-with-PyTorch"
                        / "Chapter_10" / "data" / "swing_angle_ts.csv")

if swing_angle_path.exists():
    df_swing_angle = pd.read_csv(swing_angle_path)

    _, ax = plt.subplots(figsize=(7, 7))
    df_swing_angle.plot(x="theta", y="theta_1", ax=ax, legend=False)
    ax.set_xlabel(r"$\theta(t)$")
    ax.set_ylabel(r"$\theta(t-1)$")
    ax.set_title("Swing Angle Dynamics")
    plt.show()
else:
    print(f"Swing angle data not found at {swing_angle_path}")
    print("Skipping phase space diagram.")