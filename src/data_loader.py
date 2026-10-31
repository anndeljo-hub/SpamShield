"""Load and inspect the UCI SMS Spam Collection dataset."""

import csv
from pathlib import Path

import pandas as pd

# Build paths relative to this file, so the project works no matter
# which directory you run it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "SMSSpamCollection"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the tab-separated dataset into a DataFrame with columns
    'label' and 'message'."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download it from https://archive.ics.uci.edu/dataset/228/sms+spam+collection "
            "and place the 'SMSSpamCollection' file in the data/ folder."
        )

    df = pd.read_csv(
        path,
        sep="\t",                 # fields are separated by a tab
        header=None,              # the file has no header row
        names=["label", "message"],
        quoting=csv.QUOTE_NONE,   # SMS text contains " characters; don't treat them as CSV quotes
        encoding="utf-8",
    )
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print a basic overview: shape, sample rows, types, missing values,
    duplicates, and class distribution."""
    print("=== SHAPE (rows, columns) ===")
    print(df.shape)

    print("\n=== FIRST 5 ROWS ===")
    print(df.head())

    print("\n=== DATA TYPES / NON-NULL COUNTS ===")
    df.info()

    print("\n=== MISSING VALUES PER COLUMN ===")
    print(df.isnull().sum())

    print("\n=== DUPLICATE ROWS (identical label + message) ===")
    print(df.duplicated().sum())

    print("\n=== CLASS DISTRIBUTION (counts) ===")
    print(df["label"].value_counts())

    print("\n=== CLASS DISTRIBUTION (percent) ===")
    print((df["label"].value_counts(normalize=True) * 100).round(2))


if __name__ == "__main__":
    data = load_data()
    inspect_data(data)
