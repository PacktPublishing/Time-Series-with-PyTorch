# Extracted from chapter10_Other_structures.qmd
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

import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

sns.set_style("white")
# Get working directory
cwd = os.getcwd()

# Get parent directory
parent_dir = os.path.dirname(cwd)

# Add parent directory to system path
sys.path.insert(0, parent_dir)


# Define palette
line_styles = ['-', '--', '-.', ':']

plt.rcParams['figure.figsize'] = (6, 12)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Source Sans Pro', 'Arial']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

# ----------------------------------------------------------------------

#| label: chapter-10-setup
#| message: false
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

custom_palette = ["#000000", "#0072B2", "#D55E00","#009E73","#CC79A7", "#56B4E9","#E69F00"]


class CFG:
    data_folder = Path.cwd().parent / "data"
    img_dim1 = 12
    img_dim2 = 6
    fontsize = 18

plt.rcParams.update({'figure.figsize': (CFG.img_dim1,CFG.img_dim2)})

# ----------------------------------------------------------------------

#| label: load-m5-subset
def load_m5_subset(data_folder: Path) -> pd.DataFrame:

    # Construct filepath
    filepath = data_folder / 'M5_t20_ABC.csv'
    
    # Verify file exists
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found at {filepath}")
        
    # Read data
    df = pd.read_csv(filepath, index_col=0)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Remove last 28 days of each series
    df_sorted = df.sort_values(['item_id', 'date'])
    def remove_tail_values(group):
        return group.iloc[:-28]
    df = df_sorted.groupby('item_id', group_keys=False).apply(remove_tail_values).reset_index(drop=True)
    
    # Add time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week
    return df

# ----------------------------------------------------------------------

#| label: analyze-dataset
def analyze_dataset(df: pd.DataFrame) -> dict:
    # Get ABC distribution by unique products
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

# ----------------------------------------------------------------------

#| label: plot-sales-analysis
def plot_sales_analysis(series: pd.DataFrame, figsize=(15, 10)):
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Sales over time
    axes[0].plot(series['date'], series['sold'], color = custom_palette[1], alpha=0.7)
    axes[0].set_title(f"Sales Over Time for {series['item_id'].iloc[0]}")
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Units Sold')
    
    # Sales by day of week
    sns.boxplot(data=series, x='dayofweek', y='sold', ax=axes[1], color = custom_palette[2])
    axes[1].set_title('Sales Distribution by Day of Week')
    axes[1].set_xlabel('Day of Week')
    axes[1].set_ylabel('Units Sold')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(
        "\nSummary Statistics: "
        f"{series['sold'].describe()}"
    )

# ----------------------------------------------------------------------

#| label: load-data
df = load_m5_subset(CFG.data_folder)

# Analyze dataset
analysis = analyze_dataset(df)
analysis

# ----------------------------------------------------------------------

# Select series to plot
series_to_plot = ['FOODS_3_090', 'FOODS_3_714', 'HOUSEHOLD_2_437', 'HOUSEHOLD_1_448']

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.tight_layout(pad=3.0)

# Flatten axes for easier iteration
axes = axes.flatten()

for idx, (series_id, ax) in enumerate(zip(series_to_plot, axes)):
    # Get single series
    series = df[df['item_id'] == series_id].copy()
    series = series.sort_values('date')
    
    # Get last 36 months
    last_date = series['date'].max()
    start_date = last_date - pd.DateOffset(months=36)
    series = series[series['date'] >= start_date]
    
    # Plot
    ax.plot(series['date'], series['sold'], 
            color=custom_palette[idx], 
            alpha=0.7)
    
    ax.set_title(f"Sales for {series_id}", fontsize=14)
    ax.set_xlabel('Date', fontsize=16)
    ax.set_ylabel('Units Sold', fontsize=16)
    ax.tick_params(axis='x', rotation=45)
plt.subplots_adjust(hspace=0.4, wspace=0.2)
plt.show()

# ----------------------------------------------------------------------

#| label: nbeats-linear-example
# Generate example data
temp = np.array([14, 16, 18, 20, 22, 24, 26, 28, 30]).reshape((-1, 1))  
sales = np.array([50, 55, 60, 80, 110, 160, 180, 190, 200])  

# Linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(temp, sales)

plt.scatter(temp, sales, color=custom_palette[0])
plt.plot(temp, model.predict(temp), color=custom_palette[1])
plt.xlabel('Temperature')
plt.ylabel('Sales')
plt.tight_layout(pad=3.0)
plt.show()

# ----------------------------------------------------------------------

#| label: nbeats-polynomial-example
# Polynomial basis expansion
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=3, include_bias=False)
temp_poly = poly.fit_transform(temp)

model = LinearRegression()
model.fit(temp_poly, sales)

plt.scatter(temp, sales, color=custom_palette[0])
plt.plot(temp, model.predict(temp_poly), color=custom_palette[1])
plt.xlabel('Temperature')
plt.ylabel('Sales')
plt.show()

# ----------------------------------------------------------------------

#| label: nhits-maxpool-example
# Original data
data = np.array([3, 6, 4, 0, 8, 5, 3, 2, 7, 9, 6, 0, 5, 3, 2])
x = np.arange(len(data))

# Function to perform max pooling
def max_pool(data, kernel_size):
    output = []
    for i in range(0, len(data), kernel_size):
        end = min(i + kernel_size, len(data))
        output.append(np.max(data[i:end]))
    return np.array(output)

# Calculate pooled data
pool_3 = max_pool(data, 3)
pool_5 = max_pool(data, 5)

# Create x values for pooled data (centered in their windows)
x_pool_3 = np.arange(1, len(pool_3) * 3, 3)[0:len(pool_3)]
x_pool_5 = np.arange(2, len(pool_5) * 5, 5)[0:len(pool_5)]

# Create the plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

# Top plot - data with alternating kernel=3 bands
ax1.plot(x, data, 'o-', color=custom_palette[0], linewidth=1.5, markersize=6, zorder=5)

colours_k3 = [custom_palette[1], custom_palette[5]]
for idx, i in enumerate(range(0, len(data), 3)):
    end = min(i + 3, len(data)) - 1
    ax1.axvspan(i - 0.5, end + 0.5, alpha=0.15, color=colours_k3[idx % 2],
                label='Kernel Size 3 windows' if idx == 0 else None)

# Horizontal bars for kernel=5 at the top
bar_y = 9.8
for idx, i in enumerate(range(0, len(data), 5)):
    end = min(i + 5, len(data)) - 1
    ax1.plot([i, end], [bar_y, bar_y], color=custom_palette[6], linewidth=4, solid_capstyle='butt',
             label='Kernel Size 5 windows' if idx == 0 else None)
    ax1.plot([i, i], [bar_y - 0.3, bar_y + 0.3], color=custom_palette[6], linewidth=2)
    ax1.plot([end, end], [bar_y - 0.3, bar_y + 0.3], color=custom_palette[6], linewidth=2)

ax1.set_ylim(-0.5, 10.5)
ax1.set_title('Original Time Series with Pooling Windows')
ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)

# Bottom plot - pooled results
ax2.plot(x_pool_3, pool_3, 'o-', label='Kernel Size 3', color=custom_palette[1], linewidth=2)
ax2.plot(x_pool_5, pool_5, 'o-', label='Kernel Size 5', color=custom_palette[6], linewidth=2)
ax2.legend(loc='upper right', fontsize=11)
ax2.set_title('Max Pooled Results')

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| echo: false
#| label: fig-ts-foundation-model-swing-angle-dyn
#| fig-cap: "Angle of a swing comparing to its previous time step. The swing starts from the top right of this so-called phase space diagram. The pattern indicates that the swing angle is oscillating but becoming smaller."
df_swing_angle = pd.read_csv("data/chapter10/swing_angle_ts.csv")

_, ax = plt.subplots(figsize=(7, 7))

df_swing_angle.plot(
    x="theta",
    y="theta_1",
    ax=ax,
    legend=False
)

ax.set_xlabel(rf"$\theta(t)$")
ax.set_ylabel(rf"$\theta(t-1)$")
ax.set_title("Swing Angle Dynamics");