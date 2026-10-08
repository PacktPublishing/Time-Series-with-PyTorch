import os
import gc
import warnings
import logging
import time
from pathlib import Path
from functools import partial

import numpy as np
import polars as pl
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx
from neuralforecast.losses.pytorch import MAE, HuberLoss
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import smape, rmse, mae, mase, rmsse

# ============================================================================
# ENVIRONMENT
# ============================================================================

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.cuda.set_device(0)
print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}")

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent   # works on Windows and Linux

DATA_DIR = BASE_DIR / "data"

RESULTS_DIR = BASE_DIR / "results" / "nbeatsx_weekly_phase3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIG
# ============================================================================

FORECAST_HORIZON   = 14          # weeks — matches actual test window
FREQ = "W-WED"    # Wednesday weeks
RANDOM_SEED        = 46335

# Split boundary (week_start date, Wednesday)
FINAL_TEST_START = "2025-04-16"

# Fixed NBEATSx settings — a sensible starting point, not tuned
STACK_TYPES   = ["identity", "trend", "seasonality", "exogenous"]
N_BLOCKS      = [1, 1, 1, 1]
MLP_SIZE      = 512
INPUT_SIZE    = 12
MAX_STEPS     = 15000
DROPOUT       = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE    = 128
WEIGHT_DECAY  = 5e-7

# ============================================================================
# HELPERS
# ============================================================================

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def cast_float32(df: pl.DataFrame) -> pl.DataFrame:
    for col in df.columns:
        if df[col].dtype == pl.Float64:
            df = df.with_columns(pl.col(col).cast(pl.Float32))
        elif df[col].dtype == pl.Int64:
            df = df.with_columns(pl.col(col).cast(pl.Int32))
    return df


def add_weekly_cyclic(df: pl.DataFrame) -> pl.DataFrame:
    """Weekly-cadence cyclic encodings only. No day/weekday/dayofyear features."""
    return df.with_columns([
        (2 * np.pi * pl.col("ds").dt.week() / 53).sin().cast(pl.Float32).alias("week_sin"),
        (2 * np.pi * pl.col("ds").dt.week() / 53).cos().cast(pl.Float32).alias("week_cos"),
        (2 * np.pi * pl.col("ds").dt.month() / 12).sin().cast(pl.Float32).alias("month_sin"),
        (2 * np.pi * pl.col("ds").dt.month() / 12).cos().cast(pl.Float32).alias("month_cos"),
        (2 * np.pi * pl.col("ds").dt.quarter() / 4).sin().cast(pl.Float32).alias("quarter_sin"),
        (2 * np.pi * pl.col("ds").dt.quarter() / 4).cos().cast(pl.Float32).alias("quarter_cos"),
    ])


# ============================================================================
# LOAD DATA
# ============================================================================

weekly = pl.read_parquet(DATA_DIR / "sales_weekly.parquet")
clusters = pl.read_parquet(DATA_DIR / "series_features.parquet")
items = pl.read_csv(DATA_DIR / "items.csv").select(
    ["item_id", "department_id", "class_id", "subclass_id", "item_group_id"]
)

# Standardise column name
weekly = weekly.rename({"week_start": "ds", "sales_units": "y"})

if weekly["ds"].dtype != pl.Date:
    weekly = weekly.with_columns(pl.col("ds").cast(pl.Date))

# Shift y to avoid zeros
weekly = weekly.with_columns((pl.col("y") + 1).alias("y"))

# Cast early
weekly = cast_float32(weekly)

print(f"  Loaded {weekly['unique_id'].n_unique()} series | "
      f"{weekly['ds'].min()} → {weekly['ds'].max()}")

# ============================================================================
# ATTACH HIERARCHY COVARIATES
# ============================================================================

# unique_id is "{item_id}_S1"
weekly = weekly.with_columns(
    pl.col("unique_id")
    .str.split("_")
    .list.first()
    .cast(pl.Int64)
    .alias("item_id")
)

weekly = weekly.join(items, on="item_id", how="left")

n_missing_hier_ids = (
    weekly.filter(pl.col("subclass_id").is_null())
    .select(pl.col("item_id").n_unique())
    .item()
)
print(f"  Hierarchy join check: {n_missing_hier_ids} weekly item_id(s) without a subclass")

cluster_cols = [c for c in clusters.columns if "cluster" in c.lower() or c == "unique_id"]
cluster_slim = clusters.select(cluster_cols).unique("unique_id")
weekly = weekly.join(cluster_slim, on="unique_id", how="left")

# ============================================================================
# PRICE DATA QUALITY — remove series with corrupt prices
# ============================================================================

print("Price data quality checks...")

# Negative prices (refunds/credits/data errors)
neg_price_ids = weekly.filter(pl.col("sale_price") < 0).select("unique_id").unique().to_series().to_list()
if neg_price_ids:
    print(f"  Dropping {len(neg_price_ids)} series with negative prices: {neg_price_ids}")
    weekly = weekly.filter(~pl.col("unique_id").is_in(neg_price_ids))

# Zero prices (free samples/giveaways/errors)
zero_price_ids = weekly.filter(pl.col("sale_price") == 0).select("unique_id").unique().to_series().to_list()
if zero_price_ids:
    print(f"  Dropping {len(zero_price_ids)} series with zero prices")
    weekly = weekly.filter(~pl.col("unique_id").is_in(zero_price_ids))

print(f"  Clean price range: {weekly['sale_price'].min():.2f} → {weekly['sale_price'].max():.2f}")
print(f"  Series remaining: {weekly['unique_id'].n_unique()}")

# ============================================================================
# FEATURE ENGINEERING — RELATIVE PRICE FEATURES
# ============================================================================

print("Engineering price features...")

weekly = weekly.sort(["unique_id", "ds"])

# 4-week rolling mean of sale_price (per series)
weekly = weekly.with_columns(
    pl.col("sale_price")
    .rolling_mean(window_size=4, min_periods=1)
    .over("unique_id")
    .alias("_rolling_price_mean")
)

# price_ratio_to_rolling_mean
weekly = weekly.with_columns(
    (pl.col("sale_price") / pl.col("_rolling_price_mean"))
    .cast(pl.Float32)
    .alias("price_ratio_to_rolling_mean")
)

# price_pct_change_1w
weekly = weekly.with_columns(
    pl.col("sale_price")
    .pct_change()
    .over("unique_id")
    .cast(pl.Float32)
    .alias("price_pct_change_1w")
)

# discount_depth = max(0, 1 - price_ratio)
weekly = weekly.with_columns(
    pl.max_horizontal(
        pl.lit(0.0),
        1.0 - pl.col("price_ratio_to_rolling_mean"),
    )
    .cast(pl.Float32)
    .alias("discount_depth")
)

weekly = weekly.drop("_rolling_price_mean")

# Fill NaN/null in price features
price_feat_cols = ["sale_price", "price_ratio_to_rolling_mean",
                   "price_pct_change_1w", "discount_depth"]
weekly = weekly.with_columns([
    pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan())
    .then(pl.lit(0.0).cast(pl.Float32))
    .otherwise(pl.col(c))
    .alias(c)
    for c in price_feat_cols
])
weekly = weekly.with_columns(
    [pl.col(c).fill_null(0.0) for c in price_feat_cols]
)

# ============================================================================
# CYCLIC DATE FEATURES
# ============================================================================

weekly = weekly.with_columns(pl.col("ds").cast(pl.Datetime))
weekly = add_weekly_cyclic(weekly)
weekly = weekly.with_columns(pl.col("ds").cast(pl.Date))

# ============================================================================
# is_promotional — rename for clarity
# ============================================================================

weekly = weekly.rename({"is_promotional": "promotional_proportion"})
weekly = weekly.with_columns(
    pl.col("promotional_proportion").cast(pl.Float32)
)

# ============================================================================
# PANDAS CONVERSION + ENCODING
# ============================================================================

print("Encoding static covariates...")

df_pd = weekly.to_pandas()
df_pd["ds"] = pd.to_datetime(df_pd["ds"])

# One-hot: sale_class (A–J) and cluster_label
ohe_cols = [c for c in ["sale_class", "cluster_label"] if c in df_pd.columns]
if ohe_cols:
    df_pd = pd.get_dummies(df_pd, columns=ohe_cols,
                            prefix=[c.replace("_label", "") for c in ohe_cols],
                            dtype=np.float32)

# Cast types
for col in df_pd.select_dtypes(include=["float64"]).columns:
    df_pd[col] = df_pd[col].astype(np.float32)
for col in df_pd.select_dtypes(include=["int32", "int64"]).columns:
    if col not in ["unique_id", "item_id"]:
        df_pd[col] = df_pd[col].astype(np.float32)
for col in df_pd.select_dtypes(include=["bool"]).columns:
    df_pd[col] = df_pd[col].astype(np.float32)

# ============================================================================
# DEFINE COVARIATE LISTS
# ============================================================================

FUTR_EXOG_COLS = [
    "sale_price",
    "promotional_proportion",
    "price_ratio_to_rolling_mean",
    "price_pct_change_1w",
    "discount_depth",
    "week_sin", "week_cos",
    "month_sin", "month_cos",
    "quarter_sin", "quarter_cos",
]
FUTR_EXOG_COLS = [c for c in FUTR_EXOG_COLS if c in df_pd.columns]

STATIC_BASE = ["class_id", "subclass_id", "item_group_id"]
STATIC_OHE  = [c for c in df_pd.columns
               if c.startswith("sale_class_") or c.startswith("cluster_")]
STATIC_COLS = [c for c in STATIC_BASE + STATIC_OHE if c in df_pd.columns]

for col in STATIC_COLS:
    df_pd[col] = df_pd[col].fillna(0.0).astype(np.float32)

print(f"  Future exog: {len(FUTR_EXOG_COLS)} cols")
print(f"  Static covariates: {len(STATIC_COLS)} cols")

# ============================================================================
# DIAGNOSTIC — promo-week distribution by sale class
# ============================================================================

_sc_cols = [c for c in df_pd.columns if c.startswith("sale_class_")]
_sc_map = df_pd[["unique_id"] + _sc_cols].drop_duplicates("unique_id").copy()
_sc_map["sale_class"] = (
    _sc_map[_sc_cols].idxmax(axis=1).str.replace("sale_class_", "", regex=False)
)
_sc_map = _sc_map[["unique_id", "sale_class"]]

_diag_src = df_pd[df_pd["ds"] < pd.Timestamp(FINAL_TEST_START)].copy()
_diag_src["_is_promo"] = (_diag_src["promotional_proportion"] > 0).astype(int)

_promo_diag = (
    _diag_src.groupby("unique_id")
    .agg(promo_weeks=("_is_promo", "sum"),
         total_weeks=("_is_promo", "count"))
    .reset_index()
    .merge(_sc_map, on="unique_id", how="left")
)

print("\nPromo-week distribution by sale_class:")
print(_promo_diag.groupby("sale_class")["promo_weeks"].describe(
    percentiles=[.25, .5, .75, .9]
).round(1))

print("\nSeries count by sale_class × promo-week tier:")
print(pd.crosstab(
    _promo_diag["sale_class"],
    pd.cut(_promo_diag["promo_weeks"], [-1, 2, 9, 24, 49, 9999],
           labels=["0-2", "3-9", "10-24", "25-49", "50+"])
))

PROMO_WEEK_TIERS = [
    (10, 25,  4),
    (25, 50,  2),
    (50, 9999, 1),
]

CLASS_WEIGHT = {
    "A": 0.25, "B": 0.5, "C": 1.0, "D": 1.5, "E": 1.5,
    "F": 1.25, "G": 1.0, "H": 0.6, "I": 0.4, "J": 0.25,
}

df_pd['unique_id'].nunique()

# ============================================================================
# PROMOTIONAL OVERSAMPLING
# ============================================================================

def oversample_promotional_graduated(
    df: pd.DataFrame,
    sale_class_map: pd.DataFrame,
    copy_multiplier: float = 1.0,
    max_copies_cap: int | None = None,
) -> tuple[pd.DataFrame, list]:
    promo_counts = (
        df.assign(_is_promo=(df["promotional_proportion"] > 0).astype(int))
        .groupby("unique_id")
        .agg(promo_weeks=("_is_promo", "sum"))
        .reset_index()
        .merge(sale_class_map, on="unique_id", how="left")
    )

    def copies_for(row):
        pw, sc = row["promo_weeks"], row["sale_class"]
        tier_copies = 0
        for lo, hi, n in PROMO_WEEK_TIERS:
            if lo <= pw < hi:
                tier_copies = n
                break
        weight = CLASS_WEIGHT.get(sc, 1.0)
        c = int(round(tier_copies * weight * copy_multiplier))
        if max_copies_cap is not None:
            c = min(c, max_copies_cap)
        return c

    promo_counts["copies"] = promo_counts.apply(copies_for, axis=1)

    print(f"\nGraduated promotional oversampling "
          f"(multiplier={copy_multiplier}, cap={max_copies_cap}):")
    print(f"  Total series: {df['unique_id'].nunique():,}")
    print(f"\n  Copies distribution by sale_class:")
    print(promo_counts.groupby("sale_class")["copies"]
          .agg(["mean", "max", "sum"]).round(2))
    print(f"\n  Total extra series to add: {promo_counts['copies'].sum():,}")

    qualifying_ids = promo_counts[promo_counts["copies"] > 0]["unique_id"].tolist()

    extra = []
    max_copies = int(promo_counts["copies"].max()) if len(promo_counts) else 0
    for n in range(1, max_copies + 1):
        ids_for_this_copy = promo_counts[
            promo_counts["copies"] >= n
        ]["unique_id"].tolist()
        if not ids_for_this_copy:
            continue
        c = df[df["unique_id"].isin(ids_for_this_copy)].copy()
        c["unique_id"] = c["unique_id"] + f"_promo_copy_{n}"
        extra.append(c)

    augmented = pd.concat(
        [df] + extra, ignore_index=True
    ).sort_values(["unique_id", "ds"]).reset_index(drop=True)

    print(f"  Total series after:        {augmented['unique_id'].nunique():,}")
    print(f"  Total rows after:          {len(augmented):,}")

    return augmented, qualifying_ids


df_pd, qualifying_ids = oversample_promotional_graduated(
    df_pd, _sc_map, copy_multiplier=5.0, max_copies_cap=30,
)

# ============================================================================
# STATIC DF
# ============================================================================

static_df = df_pd[["unique_id"] + STATIC_COLS].drop_duplicates("unique_id").copy()

# ============================================================================
# TRAIN / FINAL_TEST SPLIT
# ============================================================================

train_df = df_pd[["unique_id", "ds", "y"] + FUTR_EXOG_COLS].copy()

final_test_start = pd.Timestamp(FINAL_TEST_START)

final_train = train_df[train_df["ds"] < final_test_start].copy()
final_test  = train_df[train_df["ds"] >= final_test_start].copy()

print(f"\nSplit summary:")
print(f"  Final train: {final_train['ds'].min().date()} → {final_train['ds'].max().date()} "
      f"| {final_train['unique_id'].nunique()} series")
print(f"  Final test:  {final_test['ds'].min().date()} → {final_test['ds'].max().date()}")
print(f"  Test weeks per series: {final_test.groupby('unique_id').size().unique().tolist()}")

# Filter helper
real_ids = [uid for uid in df_pd["unique_id"].unique()
            if "_promo_copy_" not in uid]

# ============================================================================
# BASELINES
# ============================================================================

print("\nGenerating baseline forecasts...")

sf = StatsForecast(
    models=[Naive(), SeasonalNaive(season_length=52)],
    freq=FREQ,
    n_jobs=-1,
)
naive_preds = sf.forecast(df=final_train, h=FORECAST_HORIZON)
naive_preds = naive_preds[naive_preds["unique_id"].isin(real_ids)]
naive_preds = naive_preds.reset_index()
print(f"  Baselines done: {naive_preds.shape}")

# ============================================================================
# FIT NBEATSx — single model, fixed hyperparameters
# ============================================================================

print("\nFitting NBEATSx...")
clear_gpu()

fcst_mase  = partial(mase,  seasonality=52)
fcst_rmsse = partial(rmsse, seasonality=52)

model = NBEATSx(
    h=FORECAST_HORIZON,
    input_size=INPUT_SIZE,
    n_harmonics=3,
    n_polynomials=3,
    stack_types=STACK_TYPES,
    n_blocks=N_BLOCKS,
    mlp_units=[[MLP_SIZE, MLP_SIZE]] * len(STACK_TYPES),
    dropout_prob_theta=DROPOUT,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    max_steps=MAX_STEPS,
    val_check_steps=max(100, MAX_STEPS // 150),
    early_stop_patience_steps=200,
    scaler_type="robust",
    futr_exog_list=FUTR_EXOG_COLS,
    stat_exog_list=STATIC_COLS,
    loss=HuberLoss(),
    valid_loss=MAE(),
    optimizer=torch.optim.Adam,
    optimizer_kwargs={"weight_decay": WEIGHT_DECAY},
    accelerator="gpu",
    devices=[0],
    strategy="auto",
    enable_checkpointing=False,
    logger=False,
    enable_model_summary=False,
    gradient_clip_val=1.0,
    random_seed=RANDOM_SEED,
)

nf = NeuralForecast(models=[model], freq=FREQ)
fit_start = time.time()
nf.fit(df=final_train, static_df=static_df,
       val_size=FORECAST_HORIZON, verbose=False)
fit_time = time.time() - fit_start
print(f"  Fit time: {fit_time:.0f}s")

nf.save(path=str(RESULTS_DIR / "nbeatsx_weekly_best_phase3"), overwrite=True)

# Training curves
train_traj = nf.models[0].train_trajectories
valid_traj = nf.models[0].valid_trajectories

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot([x[0] for x in train_traj], [x[1] for x in train_traj], label="Train")
ax.plot([x[0] for x in valid_traj], [x[1] for x in valid_traj], label="Valid")
ax.set_yscale("log")
ax.set_xlabel("Step"); ax.set_ylabel("Loss (log)"); ax.legend(); ax.grid(alpha=0.3)
ax.set_title("NBEATSx Weekly — Training Curves")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "training_curves_nbeatsx_weekly_phase3.png", dpi=150)
plt.close()

# ============================================================================
# EVALUATION
# ============================================================================

nf = NeuralForecast.load(str(RESULTS_DIR / "nbeatsx_weekly_best_phase3"))

print("\nEvaluating on final test set...")
t0 = time.time()
static_df = static_df[~static_df["unique_id"].str.contains("_promo_copy_")].copy()
best_preds = nf.predict(futr_df=final_test, static_df=static_df)
print(f"  Predict time: {time.time()-t0:.1f}s")

nf.get_missing_future(futr_df=final_test)

eval_df = final_test.merge(
    best_preds[["unique_id", "ds", "NBEATSx"]], on=["unique_id", "ds"], how="left"
)
eval_df = eval_df.merge(
    naive_preds[["unique_id", "ds", "Naive", "SeasonalNaive"]],
    on=["unique_id", "ds"], how="left"
)

# Drop promo copies
eval_df = eval_df[~eval_df["unique_id"].str.contains("_promo_copy_")].copy()

# Convert inf to nan
eval_df.replace([np.inf, -np.inf], np.nan, inplace=True)

all_models = ["NBEATSx", "Naive", "SeasonalNaive"]
metrics_df = evaluate(
    eval_df, train_df=final_train,
    metrics=[smape, rmse, mae, fcst_mase, fcst_rmsse],
    models=all_models, target_col="y", id_col="unique_id",
)

# Attach sale_class from the original data
class_map = (
    df_pd[["unique_id", *[c for c in df_pd.columns if c.startswith("sale_class_")]]]
    .drop_duplicates("unique_id")
)
ohe_sc_cols = [c for c in df_pd.columns if c.startswith("sale_class_")]
if ohe_sc_cols:
    class_map = class_map.copy()
    class_map["sale_class"] = (
        class_map[ohe_sc_cols]
        .idxmax(axis=1)
        .str.replace("sale_class_", "", regex=False)
    )
    class_map = class_map[["unique_id", "sale_class"]]
    metrics_df = metrics_df.merge(class_map, on="unique_id", how="left")

# Clean inf/nan metric rows — degenerate J-class series with zero in-sample naive error
for col in all_models:
    n_inf = np.isinf(metrics_df[col]).sum()
    if n_inf > 0:
        print(f"{col}: {n_inf} inf values")

metrics_df.replace([np.inf, -np.inf], np.nan, inplace=True)
inf_ids = metrics_df.loc[metrics_df[all_models].isna().any(axis=1), 'unique_id'].unique().tolist()
print(f"Dropping {len(inf_ids)} series with NaN/inf metrics (degenerate J-class)")
metrics_df = metrics_df[~metrics_df['unique_id'].isin(inf_ids)].copy()
eval_df = eval_df[~eval_df['unique_id'].isin(inf_ids)].copy()

metrics_df.to_csv(RESULTS_DIR / "metrics_detailed_nbeatsx_weekly_phase3.csv", index=False)

summary = metrics_df.groupby("metric")[all_models].agg(["mean", "median", "std"])
summary.to_csv(RESULTS_DIR / "metrics_summary_nbeatsx_weekly_phase3.csv")

print("\n" + "=" * 60)
print("SUMMARY — median across all series")
print("=" * 60)
for metric in ["smape", "rmsse"]:
    row = metrics_df[metrics_df["metric"] == metric][all_models].median()
    print(f"\n{metric}:")
    for m, v in row.items():
        print(f"  {m:20s}: {v:.4f}")

# Win rate vs naive
for metric in ["rmsse"]:
    m_sub = metrics_df[metrics_df["metric"] == metric].copy()
    wins = (m_sub["NBEATSx"] < m_sub["Naive"]).mean()
    print(f"\nNBEATSx win rate vs Naive ({metric}): {wins*100:.1f}%")
    wins_sn = (m_sub["NBEATSx"] < m_sub["SeasonalNaive"]).mean()
    print(f"NBEATSx win rate vs SeasonalNaive ({metric}): {wins_sn*100:.1f}%")

# By sale_class
if "sale_class" in metrics_df.columns:
    abc_summary = (
        metrics_df[metrics_df["metric"].isin(["rmsse", "smape"])]
        .groupby(["metric", "sale_class"])[all_models]
        .median()
    )
    print("\nMedian RMSSE by sale_class:")
    print(abc_summary.loc["rmsse"].to_string())
    abc_summary.to_csv(RESULTS_DIR / "metrics_by_sale_class_nbeatsx_phase3.csv")

# ============================================================================
# PRICE SIMULATION — ±5/10/15/20/25%
# ============================================================================

print("\nRunning price simulation...")
price_changes = [-0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
price_results = []

# Baseline (0%)
base = final_test[["unique_id", "ds", "y", "sale_price", "promotional_proportion"]].copy()
base_preds_vals = best_preds[["unique_id", "ds", "NBEATSx"]].copy()
base = base.merge(base_preds_vals, on=["unique_id", "ds"], how="left")
base["price_change_pct"] = 0.0
base = base.rename(columns={"NBEATSx": "predicted_sales"})
price_results.append(base)

for pct in [p for p in price_changes if p != 0.0]:
    test_mod = final_test.copy()
    test_mod["sale_price"] = test_mod["sale_price"] * (1 + pct)
    test_mod["price_ratio_to_rolling_mean"] = (
        test_mod["price_ratio_to_rolling_mean"] * (1 + pct)
    )
    test_mod["discount_depth"] = np.maximum(
        0, 1 - test_mod["price_ratio_to_rolling_mean"]
    ).astype(np.float32)

    p = nf.predict(futr_df=test_mod, static_df=static_df)
    r = test_mod[["unique_id", "ds", "y", "sale_price", "promotional_proportion"]].copy()
    r = r.merge(p[["unique_id", "ds", "NBEATSx"]], on=["unique_id", "ds"], how="left")
    r["price_change_pct"] = pct
    r = r.rename(columns={"NBEATSx": "predicted_sales"})
    price_results.append(r)
    clear_gpu(); time.sleep(1)
    print(f"  {pct:+.0%} done")

price_df = pd.concat(price_results, ignore_index=True)
price_df = price_df[~price_df["unique_id"].str.contains("_promo_copy_")].copy()

price_df.to_csv(RESULTS_DIR / "price_simulation_nbeatsx_weekly_phase3.csv", index=False)

# Compute per-series arc elasticity — vectorised
base_stats = (
    price_df[price_df["price_change_pct"] == 0.0]
    .groupby("unique_id")
    .agg(base_price=("sale_price", "mean"), base_sales=("predicted_sales", "mean"))
)

scenarios = (
    price_df[price_df["price_change_pct"] != 0.0]
    .groupby(["unique_id", "price_change_pct"])
    .agg(scen_price=("sale_price", "mean"), scen_sales=("predicted_sales", "mean"))
    .reset_index()
    .merge(base_stats, on="unique_id")
)

scenarios["dp"] = (scenarios["scen_price"] - scenarios["base_price"]) / scenarios["base_price"]
scenarios["dq"] = (scenarios["scen_sales"] - scenarios["base_sales"]) / scenarios["base_sales"]
scenarios["elasticity"] = scenarios["dq"] / scenarios["dp"]
scenarios = scenarios[scenarios["dp"] != 0]

elast_df = (
    scenarios.groupby("unique_id")["elasticity"]
    .agg(model_elasticity_median="median", model_elasticity_mean="mean")
    .reset_index()
)

if "sale_class" in metrics_df.columns:
    elast_df = elast_df.merge(
        metrics_df[["unique_id", "sale_class"]].drop_duplicates(),
        on="unique_id", how="left"
    )
elast_df.to_csv(RESULTS_DIR / "model_elasticity_nbeatsx_weekly_phase3.csv", index=False)

# Elasticity summary
print(f"\nElasticity summary (NBEATSx weekly Phase 3):")
print(f"  Median: {elast_df['model_elasticity_median'].median():.3f}")
print(f"  Perverse (>0): {(elast_df['model_elasticity_median'] > 0).sum()}")
print(f"  Highly elastic (<-1): {(elast_df['model_elasticity_median'] < -1).sum()}")

if "sale_class" in elast_df.columns:
    print("\n  Median elasticity by sale_class:")
    print(elast_df.groupby("sale_class")["model_elasticity_median"].median().to_string())

# Elasticity distribution plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
vals = elast_df["model_elasticity_median"].clip(-5, 2)
ax.hist(vals, bins=80, edgecolor="black", linewidth=0.4, color="#0072B2", alpha=0.8)
ax.axvline(vals.median(), color="red", linestyle="--", linewidth=2,
           label=f"Median: {vals.median():.2f}")
ax.set_xlabel("Model Elasticity"); ax.set_ylabel("Count")
ax.set_title("NBEATSx Weekly — Elasticity Distribution"); ax.legend(); ax.grid(alpha=0.3)

if "sale_class" in elast_df.columns:
    ax = axes[1]
    elast_df.boxplot(column="model_elasticity_median", by="sale_class",
                     ax=ax, showfliers=False)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Elasticity by Sale Class"); ax.set_xlabel("Sale Class")
    ax.set_ylabel("Elasticity"); plt.suptitle("")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "elasticity_distribution_nbeatsx_weekly_phase3.png", dpi=150)
plt.close()

print(f"\n✓ NBEATSx weekly Phase 3 complete. Results in: {RESULTS_DIR}")
clear_gpu()
