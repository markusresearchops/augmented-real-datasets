"""Tests for dataset registry."""

import pytest
import tempfile
from pathlib import Path

from augmented_real_datasets.dataset_registry import DatasetRegistry, DatasetMetadata


class TestDatasetRegistry:
    def test_register_dataset(self):
        """Test registering a dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = DatasetRegistry(Path(tmpdir) / "registry.json")

            metadata = registry.register(
                name="test_dataset",
                description="Test dataset",
                source="yfinance",
                symbols=["AAPL", "MSFT"],
                start_date="2024-01-01",
                end_date="2024-12-31",
                frequency="1d",
                n_bars=252,
            )

            assert metadata.name == "test_dataset"
            assert len(registry.datasets) == 1

    def test_get_dataset(self):
        """Test retrieving a dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = DatasetRegistry(Path(tmpdir) / "registry.json")

            registry.register(
                name="test_dataset",
                description="Test",
                source="yfinance",
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-12-31",
                frequency="1d",
                n_bars=252,
            )

            metadata = registry.get("test_dataset")
            assert metadata is not None
            assert metadata.name == "test_dataset"

    def test_list_by_source(self):
        """Test filtering by source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = DatasetRegistry(Path(tmpdir) / "registry.json")

            registry.register(
                name="dataset_yf",
                description="Test",
                source="yfinance",
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-12-31",
                frequency="1d",
                n_bars=252,
            )

            registry.register(
                name="dataset_fred",
                description="Test",
                source="fred",
                symbols=["DGS10"],
                start_date="2024-01-01",
                end_date="2024-12-31",
                frequency="1d",
                n_bars=252,
            )

            yf_datasets = registry.list_by_source("yfinance")
            assert len(yf_datasets) == 1
            assert yf_datasets[0].name == "dataset_yf"

    def test_persistence(self):
        """Test that registry persists to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.json"

            # Create and register
            registry1 = DatasetRegistry(registry_path)
            registry1.register(
                name="test_dataset",
                description="Test",
                source="yfinance",
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-12-31",
                frequency="1d",
                n_bars=252,
            )

            # Load again
            registry2 = DatasetRegistry(registry_path)
            assert len(registry2.datasets) == 1
            assert registry2.get("test_dataset") is not None

    def test_update_dataset(self):
        """Test updating metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = DatasetRegistry(Path(tmpdir) / "registry.json")

            registry.register(
                name="test_dataset",
                description="Original",
                source="yfinance",
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-12-31",
                frequency="1d",
                n_bars=252,
            )

            registry.update("test_dataset", description="Updated")
            metadata = registry.get("test_dataset")
            assert metadata.description == "Updated"

    def test_delete_dataset(self):
        """Test deleting a dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = DatasetRegistry(Path(tmpdir) / "registry.json")

            registry.register(
                name="test_dataset",
                description="Test",
                source="yfinance",
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-12-31",
                frequency="1d",
                n_bars=252,
            )

            registry.delete("test_dataset")
            assert len(registry.datasets) == 0
            assert registry.get("test_dataset") is None
