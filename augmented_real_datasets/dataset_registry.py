"""
Dataset registry and metadata management.

Keeps track of downloaded/processed datasets, their sources, and
characteristics for training and evaluation.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from datetime import datetime


@dataclass
class DatasetMetadata:
    """Metadata for a dataset."""

    name: str  # Unique identifier
    description: str
    source: str  # "yfinance", "fred", "local", etc.
    symbols: List[str]  # Symbols in dataset
    start_date: str
    end_date: str
    frequency: str  # "1d", "1h", "1min", etc.
    n_bars: int
    file_path: Optional[str] = None
    augmented: bool = False
    augmentation_method: Optional[str] = None
    domain_adapted: bool = False
    adaptation_methods: Optional[List[str]] = None
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetMetadata":
        """Create from dictionary."""
        return cls(**d)


class DatasetRegistry:
    """Manage dataset catalog and metadata."""

    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize registry.

        Args:
            registry_path: path to registry JSON file
        """
        self.registry_path = Path(registry_path) if registry_path else Path("./dataset_registry.json")
        self.datasets: Dict[str, DatasetMetadata] = {}

        if self.registry_path.exists():
            self._load()

    def register(
        self,
        name: str,
        description: str,
        source: str,
        symbols: List[str],
        start_date: str,
        end_date: str,
        frequency: str,
        n_bars: int,
        file_path: Optional[str] = None,
        augmented: bool = False,
        augmentation_method: Optional[str] = None,
        domain_adapted: bool = False,
        adaptation_methods: Optional[List[str]] = None,
    ) -> DatasetMetadata:
        """Register a dataset.

        Args:
            name: unique dataset name
            description: human-readable description
            source: data source
            symbols: list of symbols
            start_date: start date
            end_date: end date
            frequency: bar frequency
            n_bars: number of bars
            file_path: path to parquet file
            augmented: whether augmented
            augmentation_method: if augmented, the method used
            domain_adapted: whether domain adapted
            adaptation_methods: if adapted, list of methods

        Returns:
            DatasetMetadata object
        """
        metadata = DatasetMetadata(
            name=name,
            description=description,
            source=source,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            n_bars=n_bars,
            file_path=file_path,
            augmented=augmented,
            augmentation_method=augmentation_method,
            domain_adapted=domain_adapted,
            adaptation_methods=adaptation_methods,
        )

        self.datasets[name] = metadata
        self._save()

        return metadata

    def get(self, name: str) -> Optional[DatasetMetadata]:
        """Get dataset metadata by name."""
        return self.datasets.get(name)

    def list_all(self) -> List[DatasetMetadata]:
        """List all registered datasets."""
        return list(self.datasets.values())

    def list_by_source(self, source: str) -> List[DatasetMetadata]:
        """List datasets from a specific source."""
        return [d for d in self.datasets.values() if d.source == source]

    def list_augmented(self) -> List[DatasetMetadata]:
        """List augmented datasets."""
        return [d for d in self.datasets.values() if d.augmented]

    def list_adapted(self) -> List[DatasetMetadata]:
        """List domain-adapted datasets."""
        return [d for d in self.datasets.values() if d.domain_adapted]

    def find_by_symbols(self, symbols: List[str]) -> List[DatasetMetadata]:
        """Find datasets containing specific symbols."""
        results = []
        for metadata in self.datasets.values():
            if any(sym in metadata.symbols for sym in symbols):
                results.append(metadata)
        return results

    def find_by_frequency(self, frequency: str) -> List[DatasetMetadata]:
        """Find datasets with specific frequency."""
        return [d for d in self.datasets.values() if d.frequency == frequency]

    def update(self, name: str, **kwargs) -> Optional[DatasetMetadata]:
        """Update metadata for a dataset."""
        if name not in self.datasets:
            return None

        metadata = self.datasets[name]
        for key, value in kwargs.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)

        self._save()
        return metadata

    def delete(self, name: str) -> bool:
        """Delete a dataset from registry."""
        if name not in self.datasets:
            return False

        del self.datasets[name]
        self._save()
        return True

    def _save(self):
        """Save registry to file."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            name: metadata.to_dict()
            for name, metadata in self.datasets.items()
        }

        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load registry from file."""
        with open(self.registry_path, "r") as f:
            data = json.load(f)

        self.datasets = {
            name: DatasetMetadata.from_dict(metadata)
            for name, metadata in data.items()
        }

    def summary(self) -> str:
        """Get summary of registry."""
        n_datasets = len(self.datasets)
        n_augmented = len(self.list_augmented())
        n_adapted = len(self.list_adapted())

        summary = f"""
Dataset Registry Summary
========================
Total datasets: {n_datasets}
Augmented:     {n_augmented}
Domain-adapted: {n_adapted}

Sources:
"""
        sources = {}
        for metadata in self.datasets.values():
            sources[metadata.source] = sources.get(metadata.source, 0) + 1

        for source, count in sources.items():
            summary += f"  {source}: {count}\n"

        return summary

    def print_summary(self):
        """Print summary to console."""
        print(self.summary())
