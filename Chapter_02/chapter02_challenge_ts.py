# Extracted from chapter2_Challenge_ts.qmd
# Do not edit the source .qmd file directly.

#| label: gapminder load
#| tbl-cap: "2.1: Gapminder life expectancy data"
#| message: false
#| echo: false
#| eval: true

# Collect Gapminder data
url = "https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv"
df = pd.read_csv(url)

# Group life_expect data by continent and year 
world_lifexp = df.groupby(["continent","year"])["lifeExp"].median().reset_index()

# Print table
print(world_lifexp.head(n=5))

# ----------------------------------------------------------------------

#| label: gapminder plot
#| fig-cap: "2.1: Time series life expectancy by continent over time"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true

# Create figure and axes
fig, ax = plt.subplots(figsize=(CFG.img_dim1, CFG.img_dim2))
sns.lineplot(
    data=world_lifexp, 
    x="year", 
    y="lifeExp", 
    hue="continent", 
    style="continent",  
    palette=custom_palette[:5], 
    markers=True,  
    dashes=True,   
    ax=ax
)
ax.set_title("Mean Life Expectancy Over Time by Continent", fontsize=18, fontweight='bold')
ax.set_xlabel("Year", fontsize=18)
ax.set_ylabel("Life Expectancy", fontsize=18)
ax.legend(title="Continent", bbox_to_anchor=(1, 1), loc='upper left')
plt.rcParams['font.size'] = 18
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: Irregular time series 
#| tbl-cap: "2.2: No sales as missing data "
#| message: false
#| echo: true
#| eval: true
df = pd.read_csv(Path.cwd().parent / "data" / "bugatti_sales.csv")
print(df.head(n=5))

# ----------------------------------------------------------------------

#| label: resampling and indexing
#| tbl-cap: "2.3: Resampled and zero filled"
#| message: false
#| echo: true
#| eval: true

# Convert year value to a string and then combine with month 
df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'], format='%Y-%b')
# Set new month variable to index and resample to get missing dates, filled with 0
df = df.set_index('date').resample('ME').sum().fillna(0)
# Drop used columns
df = df.drop(['year', 'month'], axis=1)
# Reset index
df = df.reset_index()
print(df.head(n=5))

# ----------------------------------------------------------------------

#| label: low frequency time series - bugatti sales
#| fig-cap: "2.2: Bugatti sales time series with missing values highlighted"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true

# Simple time series graph
fig, ax = plt.subplots(figsize=(CFG.img_dim1, CFG.img_dim2))
sns.lineplot(data=df, x="date", y="sales", color=custom_palette[1])
ax.set_xlabel('Date', fontsize=18)
ax.set_ylabel('Sales', fontsize=18)
ax.set_title('Bugatti Sales with Missing Values Highlighted', fontsize=18, fontweight='bold')
# Add scatter points to show missing values filled as 0 
zero_points = df[df['sales'] == 0]
plt.scatter(zero_points['date'], zero_points['sales'], color=custom_palette[4], zorder=5, alpha=0.7, s=80)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: M5 time series data 
#| message: false
#| echo: false
#| eval: true

# Load and prepare data
panel_df = pd.read_csv(CFG.data_folder / 'M5_t20_ABC.csv', index_col=False).drop(columns='Unnamed: 0')
panel_df['date'] = pd.to_datetime(panel_df['date'])
panel_df.set_index('date')

def remove_tail(df, drop_value=28):
    assert 'item_id' in df.columns and 'date' in df.columns
    df_sorted = df.sort_values(['item_id', 'date'])
    def remove_tail_values(group):
        return group.iloc[:-drop_value]
    # Keep the grouping column
    df_filter = df_sorted.groupby('item_id').apply(remove_tail_values, include_groups=True).reset_index(drop=True)
    return df_filter

panel_df = remove_tail(panel_df, drop_value=28).copy()

# ----------------------------------------------------------------------

#| label: M5 Time series plot 
#| fig-cap: "2.4: M5 sales mean imputation of missing period"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true
series_df = panel_df.loc[panel_df.item_id == 'FOODS_3_424'].copy() 

# let's make this 2013 onwards
series_df = series_df[series_df['date'].dt.year >= 2013].copy()


fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=series_df, x="date", y="sold", color="black")
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.set_title('M5 Sales - FOODS_3_424 (2013 onwards)')
plt.rcParams['font.size'] = 16
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: Find gaps in data 
#| message: false
#| echo: true
#| eval: true

# Define the gap period
gap_start = '2014-01-01'
gap_end = '2014-08-01'

# Boolean to identify gap  
gap_mask = (series_df['date'] >= gap_start) & (series_df['date'] < gap_end)

# create dataframe to hold imputed data
df_imputed = series_df.copy()
df_imputed['mean_imputed'] = series_df['sold'].replace(0, df_imputed['sold'].mean())

# ----------------------------------------------------------------------

#| label: function for imputing gap
#| message: false
#| echo: true
#| eval: true
def plot_imputed_sales(df, y_variable, title_suffix=""):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x="date", y=y_variable, color="black")
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.set_title(f'M5 Sales - FOODS_3_424 (2013 onwards) - {title_suffix}')
    plt.rcParams['font.size'] = 18
    plt.xticks(rotation=45)
    # Highlight the imputed region
    gap_start = '2014-01-01'
    gap_end = '2014-08-01'
    ax.axvspan(gap_start, gap_end, color='#b512b8', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------

#| label: M5 Time series mean impute
#| fig-cap: "2.4: M5 sales mean imputation of missing period"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true
plot_imputed_sales(df_imputed, "mean_imputed", "Mean Imputation")

# ----------------------------------------------------------------------

#| label: M5 Time series ffill impute
#| fig-cap: "2.5: M5 sales forward-fill imputation of missing period"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true
df_imputed['ffill_imputed'] = df_imputed['sold'].copy()
df_imputed.loc[gap_mask, 'ffill_imputed'] = df_imputed.loc[~gap_mask, 'sold'].iloc[-1]
plot_imputed_sales(df_imputed, "ffill_imputed", "Forward Fill Imputation")

# ----------------------------------------------------------------------

#| label: M5 Time series linear interpolation impute
#| fig-cap: "2.6: M5 sales linear interpolation imputation of missing period"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true
df_imputed['interp_imputed'] = df_imputed['sold'].copy()
df_imputed.loc[gap_mask, 'interp_imputed'] = np.nan
df_imputed['interp_imputed'] = df_imputed['interp_imputed'].interpolate()
plot_imputed_sales(df_imputed, "interp_imputed", "Linear Interpolation")

# ----------------------------------------------------------------------

#| label: M5 Time series model based impute
#| fig-cap: "2.7: M5 sales MICE imputation with Random Forest"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Add additional features to data to help RF
df_imputed['year'] = df_imputed['date'].dt.year
df_imputed['month'] = df_imputed['date'].dt.month
df_imputed['day'] = df_imputed['date'].dt.day
df_imputed['dayofweek'] = df_imputed['date'].dt.dayofweek
df_imputed['is_weekend'] = df_imputed['dayofweek'].isin([5, 6]).astype(int)

# Instantiate imputer
mice_imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=300),
                                max_iter=12, random_state=3962)

# Prepare data for imputation
columns_for_imputation = ['sold', 'year', 'month', 'day', 'dayofweek', 'is_weekend']
df_mice_impute = df_imputed[columns_for_imputation].copy()
df_mice_impute.loc[gap_mask, 'sold'] = np.nan

# Run imputation
imputed_data = mice_imputer.fit_transform(df_mice_impute)

# Update df_imputed dataframe with MICR imputed values
df_imputed['mice_imputed'] = df_imputed['sold'].copy()
df_imputed.loc[gap_mask, 'mice_imputed'] = imputed_data[gap_mask, 0].astype(df_imputed['sold'].dtype)

# Plot
plot_imputed_sales(df_imputed, "mice_imputed", "Iterative Imputation")

# ----------------------------------------------------------------------

#| label: M5 Time series basic statistics
#| message: false
#| echo: true
#| eval: false

# Check date-range
print("=== Basic Data Information ===")
print(f"Date Range: {df_imputed['date'].min()} to {df_imputed['date'].max()}")
print(f"Number of Observations: {len(df_imputed)}")

# Check for any gaps in date sequence
df_imputed['date_diff'] = df_imputed['date'].diff()
gaps = df_imputed[df_imputed['date_diff'] > pd.Timedelta(days=1)]
if not gaps.empty:
    print("\n=== Gaps in Time Series ===")
    print(f"Number of gaps: {len(gaps)}")

# Check for missing values in time series
missing = df_imputed.isnull().sum()
if missing.any():
    print("\n=== Missing Values ===")
    print(missing[missing > 0])

# Basic descriptive statistics of numerical columns
print("\n=== Descriptive Statistics ===")
print(df_imputed.describe())

# ----------------------------------------------------------------------

#| label: Additive decomposition code
#| message: false
#| echo: true
#| eval: true

# Set date to index
df_imputed = df_imputed.set_index('date') 

# sliced data to aid visualisation of seasonal patterns (e.g., last year of data)
end_date = df_imputed.index[-1]
start_date = end_date - pd.DateOffset(years=1)
df_sliced = df_imputed.loc[start_date:end_date]

# from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = sm.tsa.seasonal_decompose(df_sliced['mice_imputed'],period =28, model='additive')

# Plotting function
def plot_component(data, title, color, plot_type='line'):
    plt.figure(figsize=(12, 4))
    if plot_type == 'line':
        plt.plot(data, color=color, linewidth=2)
    elif plot_type == 'scatter':
        plt.scatter(data.index, data, color=color, s=5)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Plot observed time series
plot_component(decomposition.observed, 'Observed', custom_palette[0])

# ----------------------------------------------------------------------

#| label: Additive decomposition - trend plot
#| fig-cap: "2.8: Additive decomposition trend component"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true
plot_component(decomposition.trend, 'Trend', custom_palette[1])

# ----------------------------------------------------------------------

#| label: Additive decomposition - seasonal plot
#| fig-cap: "2.9: Additive decomposition seasonal component"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true
plot_component(decomposition.seasonal, 'Seasonal', custom_palette[2])

# ----------------------------------------------------------------------

#| label: Additive decomposition - residual plot
#| fig-cap: "2.10: Additive decomposition residuals"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true
plot_component(decomposition.resid, 'Residual', custom_palette[3], plot_type='scatter')

# ----------------------------------------------------------------------

#| label: Additive decomposition plot code
#| message: false
#| echo: true
#| eval: false

figure = decomposition.plot()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: Additive decomposition plot
#| message: false
#| echo: true
#| eval: false

# Multiplicative decomposition

# We need to shift the values in the series due to 0s causing inf values
df_sliced['mice_imputed_shifted'] = df_sliced['mice_imputed'] + 1

mult_decomposition = sm.tsa.seasonal_decompose(df_sliced['mice_imputed_shifted'], period=28, model='multiplicative')

# Plot components
plot_component(mult_decomposition.observed, 'Observed (Multiplicative)', custom_palette[0])
plot_component(mult_decomposition.trend, 'Trend (Multiplicative)', custom_palette[1])
plot_component(mult_decomposition.seasonal, 'Seasonal (Multiplicative)', custom_palette[2])
plot_component(mult_decomposition.resid, 'Residual (Multiplicative)', custom_palette[3], plot_type='scatter')

# ----------------------------------------------------------------------

#| label: HP decomposition  
#| fig-cap: "2.11: Hodrick-Prescott filter decomposition into trend and cycle"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true
from statsmodels.tsa.filters.hp_filter import hpfilter

# Apply HP Filter
cycle, trend = hpfilter(df_sliced['mice_imputed'], lamb=1600)

# Plot
plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
plt.subplot(211)
plt.plot(df_sliced['mice_imputed'], label='Original', color = custom_palette[0])
plt.plot(trend, label='Trend')
plt.legend()
plt.title('Hodrick-Prescott Filter: Trend')

plt.subplot(212)
plt.plot(cycle, label='Cycle',color = custom_palette[1])
plt.legend()
plt.title('Hodrick-Prescott Filter: Cycle')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: jj additive decomposition
#| fig-cap: "2.12: Johnson & Johnson additive decomposition (quarterly data)"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true

jj = sm.datasets.get_rdataset("JohnsonJohnson", "datasets")
jj_data = jj.data

# Convert decimal years to proper quarterly dates
def decimal_year_to_date(decimal_year):
    year = int(decimal_year)
    quarter = int(round((decimal_year - year) * 4)) + 1
    month = (quarter - 1) * 3 + 1
    return pd.Timestamp(year=year, month=month, day=1)

# Apply the conversion
jj_data['proper_date'] = jj_data['time'].apply(decimal_year_to_date)

# Create the time series using the 'value' column and proper dates
jj_series = pd.Series(jj_data['value'].values, index=jj_data['proper_date'])

# Additive decomposition
decomposition_add = sm.tsa.seasonal_decompose(jj_series, period=4, model='additive')
fig = plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2*1.5))  
axes = fig.subplots(4, 1, sharex=True)
decomposition_add.observed.plot(ax=axes[0], title='Observed', color=custom_palette[0])
decomposition_add.trend.plot(ax=axes[1], title='Trend', color=custom_palette[1])
decomposition_add.seasonal.plot(ax=axes[2], title='Seasonal', color=custom_palette[2])
decomposition_add.resid.plot(ax=axes[3], title='Residual', style='o', color=custom_palette[3])
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: jj multiplicative decomposition  
#| fig-cap: "2.13: Johnson & Johnson multiplicative decomposition (quarterly data)"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true

# Multiplicative decomposition  
decomposition_mult = sm.tsa.seasonal_decompose(jj_series, period=4, model='multiplicative')
fig = plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2*1.5))  
axes = fig.subplots(4, 1, sharex=True)
decomposition_mult.observed.plot(ax=axes[0], title='Observed', color=custom_palette[0])
decomposition_mult.trend.plot(ax=axes[1], title='Trend', color=custom_palette[1])  
decomposition_mult.seasonal.plot(ax=axes[2], title='Seasonal', color=custom_palette[2])
decomposition_mult.resid.plot(ax=axes[3], title='Residual', style='o', color=custom_palette[3])
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: M5 data with more seasonality - plus date features  
#| message: false
#| echo: false
#| eval: true

# select FOODS_3_714 for panel_df
series_df = panel_df.loc[panel_df.item_id == 'FOODS_3_714'].copy() 
# impute 0s  

# lets make this 2013 onwards
series_df = series_df[series_df['date'].dt.year >= 2013].copy()

series_df = series_df.reset_index()
# We change some columns which were numeric representations to nominal (name) 
series_df['dayofweek'] = series_df['date'].dt.day_name()
series_df['month'] = series_df['date'].dt.month_name()
series_df['week_of_year'] = series_df['date'].dt.isocalendar().week
series_df['year'] = series_df['date'].dt.isocalendar().year

# ----------------------------------------------------------------------

#| label: Day-of-the-week boxplot
#| fig-cap: "2.14: Sales distribution by day of week"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true
plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
sns.boxplot(x='dayofweek', y='sold', data=series_df, hue='dayofweek', palette=custom_palette, showfliers=False, legend=False)
plt.xlabel('Day of Week', fontsize=18)
plt.ylabel('Sales', fontsize=18)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: month boxplot
#| fig-cap: "2.15: Sales distribution by month"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true

plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
sns.boxplot(x='month', y='sold', data=series_df, hue='month', palette=custom_palette, showfliers=False, legend=False)
plt.xlabel('Month', fontsize=18)  
plt.ylabel('Sales', fontsize=18)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: year boxplot
#| fig-cap: "2.16: Sales distribution by year"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true

plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
sns.boxplot(x='year', y='sold', data=series_df, hue='year', palette=custom_palette[:4], showfliers=False, legend=False)
plt.title('Sales Distribution by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: lagged features and table
#| tbl-cap: "2.3: Example of sales with lagged features (1–7 days)"
#| message: false
#| echo: true
#| eval: true
series_df['lag_1'] = series_df['sold'].shift(1)  # Lag of 1 day
series_df['lag_2'] = series_df['sold'].shift(2)  # Lag of 2 days
series_df['lag_3'] = series_df['sold'].shift(3)  # Lag of 3 day
series_df['lag_4'] = series_df['sold'].shift(4)  # Lag of 4 days
series_df['lag_5'] = series_df['sold'].shift(5)  # Lag of 5 day
series_df['lag_6'] = series_df['sold'].shift(6)  # Lag of 6 days
series_df['lag_7'] = series_df['sold'].shift(7)  # Lag of 7 day i.e. week

print(series_df[['date','sold','lag_1','lag_2','lag_3',
'lag_4','lag_5','lag_6','lag_7']].head(n=10))

# ----------------------------------------------------------------------

#| label: correlational plot of lagged features 
#| fig-cap: "2.17: Scatter plots of sales against lagged features"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
fig.tight_layout(pad=4.0)

lags = [1, 3, 5, 7]
colors = [custom_palette[i] for i in range(4)]  # Use first 4 colors from custom_palette

for i, (lag, ax, color) in enumerate(zip(lags, axs.ravel(), colors)):
    ax.scatter(series_df['sold'], series_df[f'lag_{lag}'], alpha=0.5, color=color)
    ax.set_xlabel('Current Sales')
    ax.set_ylabel(f'Sales {lag} day{"s" if lag > 1 else ""} ago')
    ax.grid(True)

plt.show()

# ----------------------------------------------------------------------

#| label: Plotting acf
#| fig-cap: "2.18: Autocorrelation function (ACF) of sales"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true
from statsmodels.graphics.tsaplots import plot_acf
fig = plot_acf(series_df['sold'], lags=40, color=custom_palette[2])
plt.xlabel('Lag/Shift', fontsize=18)
plt.ylabel('Correlation Coefficient', fontsize=18)
plt.tight_layout()

# ----------------------------------------------------------------------

#| label: Plotting pacf
#| fig-cap: "2.19: Partial autocorrelation function (PACF) of sales"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(series_df['sold'], lags=40, color=custom_palette[2])
plt.xlabel('Lag/Shift', fontsize=18)
plt.ylabel('Correlation Coefficient', fontsize=18)
plt.tight_layout()

# ----------------------------------------------------------------------

#| label: OLS model for features
#| message: false
#| echo: false
#| eval: true
# Prepare the data
# Drop NaN (caused by lagged values)
series_df = series_df.dropna().copy()
X = series_df[['sell_price', 'dayofweek', 'month','week_of_year', 'year', 
                'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5','lag_6', 'lag_7']]
X = pd.get_dummies(X, columns=['month', 'dayofweek'], drop_first=True)
X = X.apply(pd.to_numeric, errors='coerce')
X = sm.add_constant(X)
X = X.astype(float)

y = series_df['sold']

# Fit our model
model = sm.OLS(y, X).fit()
print(model.summary())

# ----------------------------------------------------------------------

#| label: Correlational analysis
#| fig-cap: "2.20: Correlation heatmap of sales and exogenous variables"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: false
series_df = series_df[['sold','sell_price', 'dayofweek', 'month','week_of_year', 'year', 
                'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5','lag_6', 'lag_7']].copy()

numeric_columns = series_df.select_dtypes(include=[np.number]).columns

correlation_matrix = series_df[numeric_columns].corr(method='pearson')
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Pearson Correlation Heatmap')
plt.show()

# ----------------------------------------------------------------------

#| label: Random forest SHAP values
#| fig-cap: "2.21: Random forest feature importance values"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: false
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Feature Importances from Random Forest')
plt.show()

# ----------------------------------------------------------------------

#| label: regularisation for feature selection
#| fig-cap: "2.22: Ridge regression feature coefficients"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: false
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ridge = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
ridge.fit(X_scaled, y)

coef_importance = pd.DataFrame({'feature': X.columns, 'coefficient': ridge.coef_})
coef_importance = coef_importance.sort_values('coefficient', key=abs, ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='coefficient', y='feature', data=coef_importance.head(10))
plt.title('Top 10 Feature Coefficients from Ridge Regression')
plt.show()

# ----------------------------------------------------------------------

#| label: Granger causality
#| fig-cap: "2.23: Granger causality test results for exogenous variables"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: false
from statsmodels.tsa.stattools import grangercausalitytests

def granger_causality(data, target, variables, max_lag=7):
    """Test if variables Granger-cause the target variable"""
    results = {}
    for var in variables:
        try:
            # Create dataframe with target and potential causal variable
            test_data = data[[target, var]].dropna()
            test_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
            p_values = [test_result[i+1][0]['ssr_ftest'][1] for i in range(max_lag)]
            min_p_value = min(p_values)
            results[var] = min_p_value
        except:
            results[var] = 1.0  # No causality if test fails
    return results

# Test if exogenous variables Granger-cause sales (not lagged sales variables)
granger_results = granger_causality(series_df, 'sold', 
                                  ['sell_price', 'week_of_year', 'year'])

granger_df = pd.DataFrame.from_dict(granger_results, orient='index', columns=['min_p_value'])
granger_df = granger_df.sort_values('min_p_value')

plt.figure(figsize=(10, 6))
sns.barplot(x='min_p_value', y=granger_df.index, data=granger_df)
plt.axvline(x=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
plt.title('Granger Causality Test Results (p-values)')
plt.xlabel('Minimum p-value across lags')
plt.legend()
plt.show()

# ----------------------------------------------------------------------

#| label: White noise series
#| fig-cap: "2.24: Gaussian white noise time series"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true

series = pd.DataFrame(data = np.random.normal(0, 1, 10000), columns = ['noise'] )
series.plot()
print()

# ----------------------------------------------------------------------

#| label: white noise acf
#| fig-cap: "2.25: Autocorrelation function of white noise"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true

plot_acf(series['noise'], lags=25, color=custom_palette[2])
plt.xlabel('Lag/Shift', fontsize=18)
plt.ylabel('Correlation Coefficient', fontsize=18)
plt.tight_layout()

# ----------------------------------------------------------------------

#| label: passengers histogram
#| fig-cap: "2.26: Distribution of airline passenger counts"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true

series = pd.read_csv(CFG.data_folder / 'passengers.csv')
series['date'] = pd.to_datetime(series['date'])

plt.figure(figsize=(12, 6))
series['passengers'].plot.hist(bins=25, alpha=0.5, color=custom_palette[3])
plt.xlabel('Passenger Count', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.tight_layout()

# ----------------------------------------------------------------------

#| label: passengers split check
#| tbl-cap: "2.5: Mean and variance comparison for airline passengers split series"
#| message: false
#| echo: false
#| eval: true

X = series.passengers.values
split =  int(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean:')
print('chunk1: %.2f vs chunk2: %.2f' % (mean1, mean2))
print('variance:')
print('chunk1: %.2f vs chunk2: %.2f' % (var1, var2))

# ----------------------------------------------------------------------

#| label: passengers KPSS test
#| tbl-cap: "2.6: KPSS test results for airline passengers series"
#| message: false
#| echo: true
#| eval: true

from statsmodels.tsa.stattools import kpss

X = series.passengers.values
result = kpss(X)
print('KPSS Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[3].items():
    print('\t%s: %.3f' % (key, value))

# ----------------------------------------------------------------------

#| label: function for adf and plotting passengers series against transformation
#| fig-cap: "2.27: KPSS results before and after transformation"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true

series.set_index('date', inplace=True)

def kpss_test(series):
    result = kpss(series.dropna())
    return f'KPSS Statistic: {result[0]:.3f}, p-value: {result[1]:.3f}'

def plot_transformation(original, transformed, title):
    plt.figure(figsize=(12, 6))
    plt.plot(original, color=custom_palette[0], label='Original')
    plt.plot(transformed, color=custom_palette[1], label='Transformed')
    
    original_kpss = kpss_test(original)
    transformed_kpss = kpss_test(transformed)
    
    plt.title(f'{title}\nOriginal: {original_kpss}\nTransformed: {transformed_kpss}')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------

#| label: passengers differenced
#| fig-cap: "2.28: Airline passengers series vs differenced series"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true

diff_series = series.diff().dropna()
plot_transformation(series, diff_series, 'Original vs Differenced Series')

# ----------------------------------------------------------------------

#| label: passengers detrended
#| fig-cap: "2.29: Airline passengers original vs detrended series"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true

from scipy import signal

detrend_series = pd.Series(signal.detrend(series.values.flatten()), index=series.index)
plot_transformation(series, detrend_series, 'Original vs Detrended Series (Linear)')

# ----------------------------------------------------------------------

#| label: passengers logged variance
#| fig-cap: "2.30: Airline passengers log-transformed variance"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true

log_series = np.log(series)
plt.figure(figsize=(12, 6))
plt.plot(log_series, color=custom_palette[2], label='Transformed')

original_kpss = kpss_test(series)
transformed_kpss = kpss_test(log_series)
plt.title(f'{'Log Transformed Series'}\nOriginal: {original_kpss}\nTransformed: {transformed_kpss}')
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: passengers box-cox variance
#| fig-cap: "2.31: Airline passengers box-cox transformation"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true
from scipy import stats

bc_series, lambda_param = stats.boxcox(series.values.flatten())
bc_series = pd.Series(bc_series, index=series.index)

plt.figure(figsize=(12, 6))
plt.plot(bc_series, color=custom_palette[2], label='Transformed')
original_kpss = kpss_test(series)
transformed_adf = kpss_test(bc_series)
plt.title(f'{'Box-Cox Transformed Series'}\nOriginal: {original_kpss}\nTransformed: {transformed_kpss}')
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: passengers power transform
#| fig-cap: "2.32: Airline passengers power transform stabilising variance"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true
bc_series, lambda_param = stats.boxcox(series.values.flatten())
bc_series = pd.Series(bc_series, index=series.index)

bc_detrend_series = pd.Series(signal.detrend(bc_series.values.flatten()), index=bc_series.index)

plt.figure(figsize=(12, 6))
plt.plot(bc_detrend_series, color=custom_palette[2], label='Transformed')
original_kpss = kpss_test(series)
transformed_adf = kpss_test(bc_detrend_series)
plt.title(f'{'Box-Cox and detrended Transformed Series'}\nOriginal: {original_kpss}\nTransformed: {transformed_kpss}')
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: Univariate time series example
#| fig-cap: "2.33: Example of a univariate time series"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true

# Load a univariate time series example
# We'll use the passengers data as a univariate example
# Create a synthetic time series with wave patterns
np.random.seed(539278)
date_rng = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')

t = np.arange(len(date_rng))

# Create wave components with different frequencies
trend = 0.1 * t
seasonal_1 = 10 * np.sin(2 * np.pi * t / 365.25)  # Annual cycle
seasonal_2 = 5 * np.sin(2 * np.pi * t / 30.5)     # Monthly cycle
seasonal_3 = 2 * np.sin(2 * np.pi * t / 7)        # Weekly cycle
noise = np.random.normal(0, 1, len(t))

# Combine components
wave_series = pd.Series(
    trend + seasonal_1 + seasonal_2 + seasonal_3 + noise,
    index=date_rng,
    name='Wave Pattern'
)

plt.figure(figsize=(CFG.img_dim1, CFG.img_dim2))
plt.plot(wave_series, color=custom_palette[0], linewidth=2)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Value', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: Multivariate time series example
#| fig-cap: "2.34: Example of a multivariate time series"
#| fig-cap-location: bottom
#| message: false
#| echo: false
#| eval: true

# Create a multivariate time series example
np.random.seed(231598)
date_rng = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')

t = np.arange(len(date_rng))

# Create different components for multiple variables
# Temperature
temp_trend = 15 + 10 * np.sin(2 * np.pi * t / 365.25)  # Annual cycle with seasonal pattern
temp_noise = np.random.normal(0, 2, len(t))
temperature = temp_trend + temp_noise

# Humidity
humidity_trend = 60 + 20 * np.cos(2 * np.pi * t / 365.25)  # Opposite seasonal pattern to temperature
humidity_noise = np.random.normal(0, 5, len(t))
humidity = humidity_trend + humidity_noise

# Wind Speed
wind_base = 8 + 3 * np.sin(2 * np.pi * t / 182.5)  # Semi-annual cycle
wind_noise = np.random.normal(0, 1.5, len(t))
wind_speed = wind_base + wind_noise

# Create multivariate dataframe
multivariate_df = pd.DataFrame({
    'Temperature': temperature,
    'Humidity': humidity,
    'Wind_Speed': wind_speed
}, index=date_rng)

# Plot the multivariate time series
fig, ax = plt.subplots(3, 1, figsize=(CFG.img_dim1, CFG.img_dim2 * 1.3), sharex=True)

ax[0].plot(multivariate_df.index, multivariate_df['Temperature'], color=custom_palette[0], linewidth=2)
ax[0].set_ylabel('Temperature (°C)', fontsize=18)
ax[0].grid(True, linestyle='--', alpha=0.7)

ax[1].plot(multivariate_df.index, multivariate_df['Humidity'], color=custom_palette[1], linewidth=2)
ax[1].set_ylabel('Humidity (%)', fontsize=18)
ax[1].grid(True, linestyle='--', alpha=0.7)

ax[2].plot(multivariate_df.index, multivariate_df['Wind_Speed'], color=custom_palette[2], linewidth=2)
ax[2].set_ylabel('Wind Speed (m/s)', fontsize=18)
ax[2].set_xlabel('Date', fontsize=18)
ax[2].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

#| label: Multivariable time series example
#| fig-cap: "2.35: Example of a multivariable time series with different frequencies"
#| fig-cap-location: bottom
#| message: false
#| echo: true
#| eval: true

# Create a multivariable time series example with different frequencies
np.random.seed(52747)

# Daily sales data
daily_dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
daily_trend = np.linspace(100, 150, len(daily_dates))
daily_seasonal = 20 * np.sin(2 * np.pi * np.arange(len(daily_dates)) / 7)  # Weekly pattern
daily_noise = np.random.normal(0, 10, len(daily_dates))
daily_sales = daily_trend + daily_seasonal + daily_noise

# Monthly Consumer Price Index (CPI)
monthly_dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='M')
monthly_trend = np.linspace(100, 110, len(monthly_dates))
monthly_seasonal = 2 * np.sin(2 * np.pi * np.arange(len(monthly_dates)) / 12)  # Annual pattern
monthly_noise = np.random.normal(0, 1, len(monthly_dates))
monthly_cpi = monthly_trend + monthly_seasonal + monthly_noise

# Create DataFrames
daily_df = pd.DataFrame({'Sales': daily_sales}, index=daily_dates)
monthly_df = pd.DataFrame({'CPI': monthly_cpi}, index=monthly_dates)

# Plot the multivariable time series
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot daily sales
ax1.plot(daily_df.index, daily_df['Sales'], color=custom_palette[0], linewidth=1)
ax1.set_ylabel('Daily Sales ($)', fontsize=18)
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot monthly CPI
ax2.plot(monthly_df.index, monthly_df['CPI'], color=custom_palette[1], linewidth=2, marker='o')
ax2.set_ylabel('Consumer Price Index', fontsize=18)
ax2.set_xlabel('Date', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.7)

# Annotate the difference in frequency
ax1.annotate('Daily Frequency', xy=(0.02, 0.85), xycoords='axes fraction', 
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
ax2.annotate('Monthly Frequency', xy=(0.02, 0.85), xycoords='axes fraction', 
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.show()

# Example of resampling monthly data to align with daily data
# First, forward fill the monthly CPI to have a value for each day
monthly_filled = monthly_df.resample('D').ffill()

# Now we can create a combined dataset
combined_df = pd.DataFrame({
    'Sales': daily_df['Sales'],
    'CPI': monthly_filled['CPI']
})

# Show the first few rows of the combined dataset
combined_df.head()