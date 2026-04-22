"""
Domain adaptation and augmentation for time series data.

Handles non-stationarity, distribution shifts, and data augmentation
to improve model generalization.
"""

import numpy as np
import pandas as pd
from scipy import signal
from typing import Tuple, Optional


class DomainAdapter:
    """Adapt datasets to match training domain or normalize distributions."""

    @staticmethod
    def normalize_price_scale(df: pd.DataFrame, target_scale: float = 100.0) -> pd.DataFrame:
        """Normalize price levels to standard scale (e.g., all start at 100).

        Useful when combining datasets with different absolute price levels.

        Args:
            df: DataFrame with OHLCV columns
            target_scale: target starting price

        Returns:
            DataFrame with scaled prices
        """
        df = df.copy()

        for symbol in df["symbol"].unique():
            mask = df["symbol"] == symbol
            first_close = df.loc[mask, "close"].iloc[0]
            scale_factor = target_scale / first_close

            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df.loc[mask, col] *= scale_factor

        return df

    @staticmethod
    def apply_revin(
        df: pd.DataFrame,
        window: int = 20,
        learnable: bool = False,
    ) -> Tuple[pd.DataFrame, dict]:
        """Apply Reversible Instance Normalization (RevIN).

        Subtracts rolling mean and divides by rolling std to handle non-stationarity.
        Critical for financial time series (Kim et al., ICLR 2022).

        Args:
            df: DataFrame with OHLCV
            window: rolling window size
            learnable: if True, include parameters for learnable affine transform

        Returns:
            (normalized_df, stats_dict for denormalization)
        """
        df = df.copy()
        stats = {}

        for symbol in df["symbol"].unique():
            mask = df["symbol"] == symbol
            symbol_df = df.loc[mask]

            for col in ["open", "high", "low", "close"]:
                if col not in symbol_df.columns:
                    continue

                values = symbol_df[col].values
                mean = pd.Series(values).rolling(window, min_periods=1).mean().values
                std = pd.Series(values).rolling(window, min_periods=1).std().values
                std = np.where(std < 1e-6, 1.0, std)

                normalized = (values - mean) / std

                df.loc[mask, col] = normalized
                stats[f"{symbol}_{col}"] = {"mean": mean, "std": std}

        return df, stats

    @staticmethod
    def upsample_low_freq(
        df: pd.DataFrame,
        source_freq: str,
        target_freq: str,
        method: str = "interpolate",
    ) -> pd.DataFrame:
        """Upsample lower-frequency data to match training frequency.

        E.g., convert daily data to intraday via interpolation.

        Args:
            df: DataFrame with date column
            source_freq: source frequency (e.g., "1D", "1H")
            target_freq: target frequency (e.g., "1min")
            method: "interpolate" or "forward_fill"

        Returns:
            upsampled DataFrame
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        results = []

        for symbol in df["symbol"].unique():
            symbol_df = df[df["symbol"] == symbol].set_index("date")

            # Resample to target frequency
            resampled = symbol_df.resample(target_freq)

            if method == "interpolate":
                # Interpolate OHLC, forward-fill volume
                for col in ["open", "high", "low", "close"]:
                    if col in resampled.columns:
                        resampled[col] = resampled[col].interpolate(method="linear")

                resampled["volume"] = resampled["volume"].fillna(method="ffill")
            else:
                resampled = resampled.fillna(method="ffill")

            resampled["symbol"] = symbol
            resampled = resampled.reset_index()
            results.append(resampled)

        return pd.concat(results, ignore_index=True)

    @staticmethod
    def downsample_high_freq(
        df: pd.DataFrame,
        source_freq: str,
        target_freq: str,
    ) -> pd.DataFrame:
        """Downsample high-frequency data to lower frequency.

        E.g., convert 1-min bars to hourly bars.

        Args:
            df: DataFrame with date column and OHLCV
            source_freq: source frequency
            target_freq: target frequency

        Returns:
            downsampled DataFrame
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        results = []

        for symbol in df["symbol"].unique():
            symbol_df = df[df["symbol"] == symbol].set_index("date")

            agg_dict = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
            agg_dict = {k: v for k, v in agg_dict.items() if k in symbol_df.columns}

            resampled = symbol_df.resample(target_freq).agg(agg_dict)
            resampled["symbol"] = symbol
            resampled = resampled.reset_index()
            results.append(resampled)

        return pd.concat(results, ignore_index=True)


class TimeSeriesAugmentor:
    """Data augmentation techniques for time series."""

    @staticmethod
    def jitter(
        df: pd.DataFrame,
        noise_level: float = 0.001,
        columns: Optional[list] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Add Gaussian noise (jitter) to prices.

        Args:
            df: DataFrame
            noise_level: std of noise as fraction of price
            columns: which columns to jitter (default: OHLC)
            seed: random seed

        Returns:
            jittered DataFrame
        """
        if seed is not None:
            np.random.seed(seed)

        if columns is None:
            columns = ["open", "high", "low", "close"]

        df = df.copy()

        for col in columns:
            if col in df.columns:
                noise = np.random.normal(0, noise_level, len(df))
                df[col] *= (1 + noise)

        return df

    @staticmethod
    def random_walk_augmentation(
        df: pd.DataFrame,
        n_augmented_paths: int = 3,
        walk_factor: float = 0.1,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate augmented paths by random walk perturbation.

        Args:
            df: DataFrame with close prices
            n_augmented_paths: number of augmented versions per symbol
            walk_factor: magnitude of random walk drift
            seed: random seed

        Returns:
            DataFrame with original + augmented symbols
        """
        if seed is not None:
            np.random.seed(seed)

        augmented_dfs = [df.copy()]

        for aug_idx in range(n_augmented_paths):
            aug_df = df.copy()

            for symbol in aug_df["symbol"].unique():
                mask = aug_df["symbol"] == symbol
                close = aug_df.loc[mask, "close"].values

                # Random walk perturbation
                rw = np.cumsum(np.random.normal(0, walk_factor, len(close)))
                perturbed = close * np.exp(rw)

                aug_df.loc[mask, "close"] = perturbed

                # Adjust OHLC to match perturbed close
                for col in ["open", "high", "low"]:
                    if col in aug_df.columns:
                        ratio = perturbed / close
                        aug_df.loc[mask, col] *= ratio

            aug_df["symbol"] = aug_df["symbol"] + f"_aug{aug_idx+1}"
            augmented_dfs.append(aug_df)

        return pd.concat(augmented_dfs, ignore_index=True)

    @staticmethod
    def mixup(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        alpha: float = 0.5,
    ) -> pd.DataFrame:
        """Create synthetic paths via mixup of two datasets.

        Blends OHLCV of two different symbols to create new training examples.

        Args:
            df1: first dataset (must have same length per symbol)
            df2: second dataset
            alpha: interpolation parameter (0-1)

        Returns:
            mixed DataFrame
        """
        assert len(df1) == len(df2), "DataFrames must have same length"

        mixed = df1.copy()

        for col in ["open", "high", "low", "close", "volume"]:
            if col in mixed.columns:
                mixed[col] = alpha * df1[col].values + (1 - alpha) * df2[col].values

        # Create new symbol names
        symbols_1 = df1["symbol"].unique()
        symbols_2 = df2["symbol"].unique()
        mixed["symbol"] = [f"{s1}_mix_{s2}" for s1, s2 in zip(symbols_1, symbols_2)]

        return mixed

    @staticmethod
    def detrend_and_add_new_trend(
        df: pd.DataFrame,
        trend_strength: float = 0.0005,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Remove original trend and apply synthetic trend for augmentation.

        Args:
            df: DataFrame
            trend_strength: slope of new trend
            seed: random seed

        Returns:
            detrended + retrended DataFrame
        """
        if seed is not None:
            np.random.seed(seed)

        df = df.copy()

        for symbol in df["symbol"].unique():
            mask = df["symbol"] == symbol
            close = df.loc[mask, "close"].values

            # Detrend: remove linear trend using log returns
            n = len(close)
            t = np.arange(n)

            # Fit linear trend to log prices
            log_close = np.log(close)
            trend_coeff = np.polyfit(t, log_close, 1)[0]
            original_trend = trend_coeff * t

            # Detrend: subtract trend
            detrended = log_close - original_trend

            # Add new trend (with same sign to maintain upward/downward bias)
            new_trend = trend_strength * t
            retrended = detrended + new_trend

            # Convert back to prices, ensure positivity
            prices = np.exp(retrended)

            # Normalize to start at original first price
            prices = prices * close[0] / prices[0]

            # Ensure all prices are positive
            prices = np.maximum(prices, close.min() * 0.1)

            df.loc[mask, "close"] = prices

        return df
