"""
Real dataset sources and download utilities.

Provides interfaces to various public time series datasets:
- Equity data (yfinance, IBKR historical)
- Commodity futures
- Crypto
- Economic indicators
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

# Optional imports
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    from pandas_datareader import data as pdr
    HAS_PANDAS_DATAREADER = True
except ImportError:
    HAS_PANDAS_DATAREADER = False


class DatasetDownloader:
    """Download and manage real time series datasets."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize downloader.

        Args:
            cache_dir: directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./datasets/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_yfinance(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Download historical data from Yahoo Finance.

        Args:
            symbols: list of ticker symbols
            start_date: start date (YYYY-MM-DD)
            end_date: end date (YYYY-MM-DD)
            interval: "1d", "1wk", "1mo", or "1m" (for last 7 days)
            cache: whether to cache downloads

        Returns:
            dict mapping symbol -> DataFrame with OHLCV
        """
        if not HAS_YFINANCE:
            raise ImportError("Install yfinance: pip install yfinance")

        data = {}
        for symbol in symbols:
            cache_path = self.cache_dir / f"{symbol}_{start_date}_{end_date}_{interval}.parquet"

            if cache and cache_path.exists():
                data[symbol] = pd.read_parquet(cache_path)
                continue

            try:
                df = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False,
                )

                if df.empty:
                    warnings.warn(f"No data for {symbol}")
                    continue

                # Standardize column names
                df.columns = [col.lower() for col in df.columns]
                df = df.rename(columns={"adj close": "adjclose"})
                df = df.reset_index()

                if "date" not in df.columns and "datetime" in df.columns:
                    df = df.rename(columns={"datetime": "date"})

                df["symbol"] = symbol

                if cache:
                    df.to_parquet(cache_path)

                data[symbol] = df

            except Exception as e:
                warnings.warn(f"Failed to download {symbol}: {e}")
                continue

        return data

    def download_fred(
        self,
        series_ids: List[str],
        start_date: str,
        end_date: str,
        cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Download economic indicators from FRED.

        Args:
            series_ids: FRED series identifiers (e.g., "DGS10" for 10-yr yield)
            start_date: start date
            end_date: end date
            cache: whether to cache

        Returns:
            dict mapping series_id -> DataFrame
        """
        if not HAS_PANDAS_DATAREADER:
            raise ImportError("Install pandas-datareader: pip install pandas-datareader")

        data = {}
        for series_id in series_ids:
            cache_path = self.cache_dir / f"fred_{series_id}_{start_date}_{end_date}.parquet"

            if cache and cache_path.exists():
                data[series_id] = pd.read_parquet(cache_path)
                continue

            try:
                df = pdr.get_data_fred(series_id, start=start_date, end=end_date)
                df = df.reset_index()
                df = df.rename(columns={"index": "date", series_id: "value"})
                df["series_id"] = series_id

                if cache:
                    df.to_parquet(cache_path)

                data[series_id] = df

            except Exception as e:
                warnings.warn(f"Failed to download FRED {series_id}: {e}")
                continue

        return data

    def load_local_parquet(self, path: Path, resample_freq: Optional[str] = None) -> pd.DataFrame:
        """Load local parquet file (e.g., from tokenized-forecaster data/).

        Args:
            path: path to parquet file
            resample_freq: optionally resample to freq (e.g., "5min", "1h")

        Returns:
            DataFrame with standardized columns
        """
        df = pd.read_parquet(path)

        if resample_freq:
            # Resample OHLCV data
            if "date" in df.columns:
                df = df.set_index("date")

            agg_dict = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
            agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

            df = df.resample(resample_freq).agg(agg_dict)
            df = df.reset_index()

        return df


def load_yfinance_data(
    symbols: List[str],
    start_date: str = "2020-01-01",
    end_date: str = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Convenience function to load yfinance data.

    Args:
        symbols: list of ticker symbols
        start_date: start date
        end_date: end date (default: today)
        interval: "1d", "1wk", "1mo", or "1m"

    Returns:
        concatenated DataFrame
    """
    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    downloader = DatasetDownloader()
    data_dict = downloader.download_yfinance(symbols, start_date, end_date, interval)

    dfs = [df for df in data_dict.values() if not df.empty]

    if not dfs:
        raise ValueError("No data downloaded for any symbols")

    return pd.concat(dfs, ignore_index=True)


def create_synthetic_indices(
    constituent_symbols: List[str],
    weights: Optional[List[float]] = None,
    start_date: str = "2020-01-01",
    end_date: str = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Create synthetic index from constituent stocks.

    Can serve as proxy for broader market data.

    Args:
        constituent_symbols: list of ticker symbols
        weights: portfolio weights (default: equal-weight)
        start_date: start date
        end_date: end date

    Returns:
        (index_df, metadata)
    """
    if weights is None:
        weights = [1.0 / len(constituent_symbols)] * len(constituent_symbols)

    data = load_yfinance_data(constituent_symbols, start_date, end_date)

    # Normalize by open price (log returns)
    data["log_return"] = data.groupby("symbol")["close"].apply(
        lambda x: np.log(x / x.iloc[0])
    ).values

    # Weight the returns
    weighted_returns = {}
    for symbol, weight in zip(constituent_symbols, weights):
        symbol_data = data[data["symbol"] == symbol]
        weighted_returns[symbol] = weight * symbol_data["log_return"].values

    # Combine
    index_return = np.sum(list(weighted_returns.values()), axis=0)
    index_price = 100 * np.exp(index_return)

    # Use first symbol's dates
    dates = data[data["symbol"] == constituent_symbols[0]]["date"].values

    index_df = pd.DataFrame({
        "symbol": "INDEX",
        "date": dates,
        "close": index_price,
        "open": index_price,  # Simplified
        "high": index_price,
        "low": index_price,
        "volume": 0,
    })

    metadata = {
        "constituents": constituent_symbols,
        "weights": weights,
        "type": "equal_weight_index",
    }

    return index_df, metadata
