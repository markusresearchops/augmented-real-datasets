# Augmented Real Datasets

Real time series dataset collection, curation, and domain adaptation for stock prediction model training.

**Inspired by**: Chronos (Ansari et al., 2024) approach to combining real and synthetic data. This repo complements synthetic-timeseries-generation by providing curated real datasets with domain adaptation techniques.

## Two Key Data Augmentation Approaches

### 1. **Real Dataset Collection & Curation**

Gather diverse real-world time series from multiple sources:

- **Equity data** (Yahoo Finance, IBKR historical)
- **Different market conditions** (bull/bear markets, high/low volatility regimes)
- **Different asset classes** (large-cap, small-cap, indices, sectors)
- **Economic indicators** (FRED: yields, unemployment, inflation)

```python
from augmented_real_datasets import DatasetDownloader, DatasetRegistry

# Download real data
downloader = DatasetDownloader(cache_dir="./data_cache")
data = downloader.download_yfinance(
    symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    start_date="2020-01-01",
    end_date="2024-01-01",
)

# Register for tracking
registry = DatasetRegistry()
registry.register(
    name="sp500_5stocks_2020_2024",
    description="5 large-cap stocks over 4 years",
    source="yfinance",
    symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    start_date="2020-01-01",
    end_date="2024-01-01",
    frequency="1d",
    n_bars=1000,
)
```

### 2. **Domain Adaptation & Augmentation**

Transform datasets to match training domain and improve robustness:

#### **Domain Adaptation Methods:**

- **Price Normalization**: Scale all assets to start at 100 (removes absolute price level bias)
- **RevIN (Reversible Instance Normalization)**: Subtract rolling mean, divide by rolling std to handle non-stationarity
- **Frequency Adjustment**: Upsample daily to intraday (interpolation) or downsample to match training freq

```python
from augmented_real_datasets import DomainAdapter

# Normalize scales
df = DomainAdapter.normalize_price_scale(df, target_scale=100.0)

# Apply RevIN (critical for financial non-stationarity)
df, stats = DomainAdapter.apply_revin(df, window=20)

# Upsample daily data to intraday
df = DomainAdapter.upsample_low_freq(df, source_freq="1D", target_freq="1H")
```

#### **Data Augmentation Methods:**

- **Jitter**: Add Gaussian noise (~0.1% of price) to improve robustness
- **Random Walk Augmentation**: Perturb prices with synthetic random walks to create variants
- **Detrend & Retread**: Remove original trend and apply synthetic trend

```python
from augmented_real_datasets import TimeSeriesAugmentor

# Add noise for robustness
df = TimeSeriesAugmentor.jitter(df, noise_level=0.001)

# Create 3 augmented versions of each symbol
df_aug = TimeSeriesAugmentor.random_walk_augmentation(df, n_augmented_paths=3)

# Detrend and apply new synthetic trend
df = TimeSeriesAugmentor.detrend_and_add_new_trend(df, trend_strength=0.0005)
```

## Integration with tokenized-forecaster

Output format integrates with existing pipeline:

```python
import pandas as pd
from pathlib import Path

# Load augmented data
df = pd.read_parquet("augmented_data.parquet")

# Copy to tokenized-forecaster
output_path = Path("../tokenized-forecaster/data/augmented/augmented_year=2024.parquet")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(output_path)

# Then run tokenized-forecaster pipeline
# cd ../tokenized-forecaster
# pipeline-mh --symbols AAPL,MSFT,GOOGL --force
```

## Installation

```bash
pip install -e .  # with base dependencies
pip install -e ".[datasets]"  # with yfinance and pandas-datareader
pip install -e ".[test]"  # with test suite
```

## Usage

### CLI

```bash
# Download stocks
augment-datasets download-yf --symbols AAPL,MSFT,GOOGL --start-date 2020-01-01 --output ./data

# Apply domain adaptation (RevIN)
augment-datasets adapt --input data.parquet --method revin --output data_adapted.parquet

# Augment with random walks
augment-datasets augment --input data.parquet --method random-walk --n-augmented 5 --output data_aug.parquet

# List registered datasets
augment-datasets registry-list
```

### Python API

```python
from augmented_real_datasets import (
    DatasetDownloader,
    DomainAdapter,
    TimeSeriesAugmentor,
    DatasetRegistry,
)

# Comprehensive workflow
downloader = DatasetDownloader(cache_dir="./cache")

# 1. Download
data = downloader.download_yfinance(
    ["AAPL", "MSFT", "GOOGL"],
    start_date="2022-01-01",
    end_date="2024-01-01",
)

df = pd.concat(data.values(), ignore_index=True)

# 2. Adapt domain
df = DomainAdapter.normalize_price_scale(df)
df, stats = DomainAdapter.apply_revin(df, window=20)

# 3. Augment
df_aug = TimeSeriesAugmentor.jitter(df, noise_level=0.001)
df_aug = TimeSeriesAugmentor.random_walk_augmentation(df_aug, n_augmented_paths=3)

# 4. Register for tracking
registry = DatasetRegistry()
registry.register(
    name="sp500_adapted_augmented",
    description="Real S&P500 stocks with RevIN + augmentation",
    source="yfinance",
    symbols=["AAPL", "MSFT", "GOOGL"],
    start_date="2022-01-01",
    end_date="2024-01-01",
    frequency="1d",
    n_bars=500,
    augmented=True,
    augmentation_method="jitter+random_walk",
    domain_adapted=True,
    adaptation_methods=["normalize", "revin"],
)

# 5. Save
df_aug.to_parquet("sp500_final.parquet")
```

## Module Structure

```
augmented_real_datasets/
  dataset_sources.py          # Data downloaders (yfinance, FRED, local)
  domain_adaptation.py        # Domain adaptation & augmentation
  dataset_registry.py         # Metadata tracking and curation
  cli.py                      # CLI entry point
```

## Why Both Real + Synthetic?

**This Repo (Real Data):**
- Authentic market patterns and dynamics
- Real correlation structures
- Handles actual edge cases (gaps, halts, unusual volume)
- Domain-specific knowledge

**synthetic-timeseries-generation (Synthetic Data):**
- Fills gaps in real data
- Generates controlled variations
- Improves generalization to unseen patterns
- Reduces overfitting to specific assets

**Together:** Foundation models trained on both real + synthetic achieve **comparable zero-shot performance** across held-out test sets (Chronos finding).

## Best Practices

1. **Domain Adapt Before Training**: Apply RevIN to continuous features to handle non-stationarity
2. **Combine Multiple Sources**: Mix different asset classes and time periods
3. **Augment Conservatively**: Small noise/perturbations improve robustness without destroying signal
4. **Track Metadata**: Use DatasetRegistry to document sources and transformations
5. **Balance Real:Synthetic**: ~70% real, ~30% synthetic is a good starting ratio

## Performance Notes

- Download: ~10-50 symbols/min depending on source and connection
- Domain adaptation: ~100K rows/sec (RevIN, normalization)
- Augmentation: ~50K rows/sec (random walk, jitter)
- All operations fit in memory for typical training datasets

## References

- **Chronos** (Ansari et al., 2024): Foundation models trained on diverse real + synthetic data generalize better
- **RevIN** (Kim et al., ICLR 2022): Critical normalization for non-stationary financial time series
- **Mixup** (Zhang et al., 2018): Data augmentation via interpolation

## Future Extensions

- [ ] Integration with more data sources (Alpaca, Polygon, IQFeed)
- [ ] Automated market regime detection
- [ ] Intra-day seasonality modeling
- [ ] Multi-asset correlation preservation in augmentation
- [ ] Backtesting harness to validate synthetic data properties
