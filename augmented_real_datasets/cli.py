"""Command-line interface for dataset collection and augmentation."""

import argparse
import sys
from pathlib import Path

from .dataset_sources import DatasetDownloader, load_yfinance_data
from .domain_adaptation import DomainAdapter, TimeSeriesAugmentor
from .dataset_registry import DatasetRegistry


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Collect, curate, and augment real time series datasets for stock forecasting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Download OHLCV for stocks from Yahoo Finance
  augment-datasets download-yf --symbols AAPL,MSFT,GOOGL --start-date 2020-01-01 --output ./data

  # Normalize price scales and apply RevIN
  augment-datasets adapt --input data.parquet --method revin --output data_adapted.parquet

  # Augment with random walk perturbation
  augment-datasets augment --input data.parquet --method random-walk --output data_aug.parquet

  # Create and manage dataset registry
  augment-datasets registry-list
  augment-datasets registry-register --name my_dataset --source yfinance --description "Test data"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Download command
    download_parser = subparsers.add_parser("download-yf", help="Download from Yahoo Finance")
    download_parser.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="Comma-separated symbols (e.g., AAPL,MSFT,GOOGL)",
    )
    download_parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    download_parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD, default: today)",
    )
    download_parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Interval (1d, 1h, 1m)",
    )
    download_parser.add_argument(
        "--output",
        type=str,
        default="./yfinance_data",
        help="Output directory",
    )

    # Adaptation command
    adapt_parser = subparsers.add_parser("adapt", help="Apply domain adaptation")
    adapt_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input parquet file",
    )
    adapt_parser.add_argument(
        "--method",
        type=str,
        choices=["normalize", "revin", "upsample", "downsample"],
        required=True,
        help="Adaptation method",
    )
    adapt_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output parquet file",
    )
    adapt_parser.add_argument(
        "--target-freq",
        type=str,
        help="Target frequency for upsample/downsample",
    )

    # Augmentation command
    aug_parser = subparsers.add_parser("augment", help="Augment dataset")
    aug_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input parquet file",
    )
    aug_parser.add_argument(
        "--method",
        type=str,
        choices=["jitter", "random-walk", "detrend"],
        required=True,
        help="Augmentation method",
    )
    aug_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output parquet file",
    )
    aug_parser.add_argument(
        "--n-augmented",
        type=int,
        default=3,
        help="Number of augmented paths (for random-walk)",
    )

    # Registry commands
    registry_parser = subparsers.add_parser("registry-list", help="List registered datasets")
    registry_parser.add_argument(
        "--registry-path",
        type=str,
        default="./dataset_registry.json",
        help="Path to registry file",
    )

    args = parser.parse_args()

    if args.command == "download-yf":
        return cmd_download_yf(args)
    elif args.command == "adapt":
        return cmd_adapt(args)
    elif args.command == "augment":
        return cmd_augment(args)
    elif args.command == "registry-list":
        return cmd_registry_list(args)
    else:
        parser.print_help()
        return 0


def cmd_download_yf(args):
    """Execute download-yf command."""
    symbols = args.symbols.split(",")
    print(f"Downloading {len(symbols)} symbols from Yahoo Finance...")
    print(f"  Symbols: {symbols}")
    print(f"  Period: {args.start_date} to {args.end_date or 'today'}")

    downloader = DatasetDownloader(cache_dir=Path(args.output) / "cache")
    data = downloader.download_yfinance(
        symbols,
        args.start_date,
        args.end_date or None,
        interval=args.interval,
    )

    output_path = Path(args.output) / f"yfinance_{args.start_date}.parquet"
    import pandas as pd
    df = pd.concat(data.values(), ignore_index=True)
    df.to_parquet(output_path)

    print(f"✓ Downloaded {len(df)} rows to {output_path}")
    return 0


def cmd_adapt(args):
    """Execute adapt command."""
    import pandas as pd

    print(f"Applying {args.method} adaptation...")

    df = pd.read_parquet(args.input)

    if args.method == "normalize":
        df = DomainAdapter.normalize_price_scale(df)
    elif args.method == "revin":
        df, _ = DomainAdapter.apply_revin(df)
    elif args.method in ["upsample", "downsample"]:
        if not args.target_freq:
            print("Error: --target-freq required", file=sys.stderr)
            return 1
        if args.method == "upsample":
            df = DomainAdapter.upsample_low_freq(df, "1D", args.target_freq)
        else:
            df = DomainAdapter.downsample_high_freq(df, "1min", args.target_freq)

    df.to_parquet(args.output)
    print(f"✓ Saved adapted data to {args.output}")
    return 0


def cmd_augment(args):
    """Execute augment command."""
    import pandas as pd

    print(f"Applying {args.method} augmentation...")

    df = pd.read_parquet(args.input)

    if args.method == "jitter":
        df = TimeSeriesAugmentor.jitter(df)
    elif args.method == "random-walk":
        df = TimeSeriesAugmentor.random_walk_augmentation(df, n_augmented_paths=args.n_augmented)
    elif args.method == "detrend":
        df = TimeSeriesAugmentor.detrend_and_add_new_trend(df)

    df.to_parquet(args.output)
    print(f"✓ Saved augmented data ({len(df)} rows) to {args.output}")
    return 0


def cmd_registry_list(args):
    """Execute registry-list command."""
    registry = DatasetRegistry(Path(args.registry_path))
    registry.print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
