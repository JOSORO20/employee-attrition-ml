from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import paths

TARGET_COL = "approved"

def load_data(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or paths.data_raw
    df = pd.read_csv(path)
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[TARGET_COL])
    return train_df, test_df

def save_splits(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path | None = None) -> None:
    out = out_dir or paths.data_processed
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out / "train.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)
