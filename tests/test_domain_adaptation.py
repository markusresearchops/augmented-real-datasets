"""Tests for domain adaptation and augmentation."""

import numpy as np
import pandas as pd
import pytest

from augmented_real_datasets.domain_adaptation import DomainAdapter, TimeSeriesAugmentor


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data."""
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="1D")
    close = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, n)))

    df = pd.DataFrame({
        "symbol": "TEST",
        "date": dates,
        "open": close + np.random.normal(0, 1, n),
        "high": close + np.abs(np.random.normal(2, 1, n)),
        "low": close - np.abs(np.random.normal(2, 1, n)),
        "close": close,
        "volume": np.random.lognormal(14, 1, n).astype(int),
    })

    return df


class TestDomainAdapter:
    def test_normalize_price_scale(self, sample_ohlcv):
        """Test price scale normalization."""
        df_norm = DomainAdapter.normalize_price_scale(sample_ohlcv, target_scale=100.0)

        # Check first close is 100
        assert df_norm.iloc[0]["close"] == pytest.approx(100.0)

        # Check that OHLC relationships are preserved
        assert (df_norm["high"] >= df_norm["low"]).all()

    def test_revin(self, sample_ohlcv):
        """Test RevIN normalization."""
        df_revin, stats = DomainAdapter.apply_revin(sample_ohlcv, window=10)

        # Check that output is smaller (normalized)
        assert df_revin["close"].std() < sample_ohlcv["close"].std()

        # Check stats dict
        assert len(stats) > 0
        assert "TEST_close" in stats

    def test_downsample(self, sample_ohlcv):
        """Test downsampling to coarser frequency."""
        df_down = DomainAdapter.downsample_high_freq(
            sample_ohlcv,
            source_freq="1D",
            target_freq="1W"
        )

        # Should have fewer bars
        assert len(df_down) < len(sample_ohlcv)

        # High should be >= low
        assert (df_down["high"] >= df_down["low"]).all()


class TestTimeSeriesAugmentor:
    def test_jitter(self, sample_ohlcv):
        """Test jitter augmentation."""
        df_jitter = TimeSeriesAugmentor.jitter(sample_ohlcv, noise_level=0.01)

        # Should be different but similar
        assert not np.allclose(df_jitter["close"], sample_ohlcv["close"])
        assert np.allclose(df_jitter["close"], sample_ohlcv["close"], rtol=0.1)

    def test_random_walk_augmentation(self, sample_ohlcv):
        """Test random walk augmentation."""
        df_aug = TimeSeriesAugmentor.random_walk_augmentation(
            sample_ohlcv,
            n_augmented_paths=2
        )

        # Should have original + 2 augmented
        assert len(df_aug) == 3 * len(sample_ohlcv)

        # Should have different symbols
        symbols = df_aug["symbol"].unique()
        assert len(symbols) == 3

    def test_detrend_and_add_new_trend(self, sample_ohlcv):
        """Test detrending and retrending."""
        df_retrend = TimeSeriesAugmentor.detrend_and_add_new_trend(sample_ohlcv)

        # Should still have valid prices
        assert (df_retrend["close"] > 0).all()

        # Should be different from original
        assert not np.allclose(df_retrend["close"], sample_ohlcv["close"])
