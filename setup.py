from setuptools import setup, find_packages

setup(
    name="augmented-real-datasets",
    version="0.1.0",
    description="Real time series dataset collection, curation, and domain adaptation for stock prediction",
    author="Markus Research Ops",
    author_email="markus.research.ops@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "pyarrow>=6.0.0",
        "scikit-learn>=1.0.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "test": ["pytest>=6.0", "pytest-cov"],
        "datasets": [
            "yfinance>=0.2.0",  # Yahoo Finance
            "pandas-datareader>=0.10.0",  # FRED, other sources
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "augment-datasets=augmented_real_datasets.cli:main",
        ],
    },
)
