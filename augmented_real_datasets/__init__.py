"""
Real Time Series Dataset Collection and Augmentation.

Complements synthetic data generation with curated, diverse real datasets
to improve forecasting model robustness (following Chronos approach).
"""

__version__ = "0.1.0"

from .dataset_sources import DatasetDownloader, load_yfinance_data
from .domain_adaptation import DomainAdapter, TimeSeriesAugmentor
from .dataset_registry import DatasetRegistry, DatasetMetadata

__all__ = [
    "DatasetDownloader",
    "load_yfinance_data",
    "DomainAdapter",
    "TimeSeriesAugmentor",
    "DatasetRegistry",
    "DatasetMetadata",
]
