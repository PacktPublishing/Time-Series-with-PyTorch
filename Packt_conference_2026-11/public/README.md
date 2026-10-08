# Weekly Retail Sales Forecasting Dataset

Weekly unit sales for 18,692 products in a single grocery supermarket, with prices, promotion flags, a product hierarchy and precomputed series features. The data covers 2023-07-05 to 2025-07-16, in weeks that start on Wednesday.

The data has been anonymised:

- Every id is internal to this dataset, numbered 1..N in random order.
- Product descriptions are generic rewrites with no brand names.
- Brands are replaced with pseudonyms.
- The store is labelled `S1`.

Sales, prices and dates are unchanged.

## Files

| File | Rows | Description |
| --- | --- | --- |
| `data/sales_weekly.parquet` | 1,967,797 | One row per series per week |
| `data/series_features.parquet` | 18,692 | Time-series features and cluster label per series |
| `data/items.csv` | 18,692 | Product attributes and hierarchy ids |
| `data/hierarchy.csv` | 1,483 | Subclass → class → department, with names |
| `nbeats_weekly_phase3.py` | | Example global model (NBEATSx, NeuralForecast) |

## Joining

```text
sales_weekly.unique_id  = "{item_id}_S1"
series_features.unique_id = sales_weekly.unique_id
items.item_id           -> parse from unique_id
items.subclass_id       -> hierarchy.subclass_id
```

## sales_weekly.parquet

| Column | Type | Description |
| --- | --- | --- |
| `unique_id` | str | Series id, `{item_id}_S1` |
| `week_start` | date | Week start (Wednesday) |
| `sales_units` | float | Weekly sales quantity (forecast target), as recorded by the source system |
| `is_promotional` | bool | Item was on promotion in the week |
| `sale_price` | float | Average selling price |
| `regular_price` | float | Regular (non-promotional) shelf price |
| `days_in_week` | int | Daily-record count from the source aggregation; constant within most series and not used by the example model |
| `sale_class` | str | Sales-velocity class A–J (A = highest) |
| `has_any_promo` | bool | Series has at least one promotional week |
| `n_promo_weeks` | int | Number of promotional weeks in the series |
| `has_promo_in_forecast` | bool | Promotion falls in the 14-week test window |
| `n_promo_weeks_in_forecast` | int | Promotional weeks in the test window |
| `promo_ratio` | float | Share of weeks on promotion |

## items.csv

| Column | Description |
| --- | --- |
| `item_id` | Product id |
| `item_desc` | Generic product description; empty for about 1,000 products with no source description |
| `brand_id`, `brand_name` | Brand pseudonym, e.g. `Confectionery Brand 07`. The category word is the brand's main department. `Unbranded` covers produce, seasonal and generic lines |
| `own_brand` | True for the retailer's own labels (`Own Brand 1`–`3`) |
| `division_id` | Store division; a coarse grouping that is not strictly nested in the hierarchy |
| `department_id`, `class_id`, `subclass_id` | Merchandise hierarchy; empty for 448 products |
| `item_group_id` | Replenishment grouping, close to subclass |
| `is_delisted` | Product has since been delisted |

## Example model

`nbeats_weekly_phase3.py` trains an NBEATSx global model with:

- price and promotion as future covariates;
- class, subclass, item group, sales class and cluster as static covariates;
- a single fit with fixed hyperparameters (no tuning), followed by a price simulation that estimates per-series elasticity.

It holds out the final 14 weeks (from 2025-04-16) for testing and compares against Naive and Seasonal Naive baselines.

Requirements: `neuralforecast`, `statsforecast`, `utilsforecast`, `polars`, `pandas`, `torch`, `matplotlib`, `seaborn`.

```bash
python nbeats_weekly_phase3.py
```

A CUDA GPU is expected.