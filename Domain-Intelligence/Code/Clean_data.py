"""
clean_data.py
=============
Battery Consultant AI – Data Cleaning Pipeline
-----------------------------------------------
Reads  : Domain-Intelligence/Dataset/battery.csv
Writes : Domain-Intelligence/Dataset/clean_battery.csv

Run inside the project virtual environment:
    source .venv/bin/activate
    python Domain-Intelligence/Code/clean_data.py
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(                          # Battery-Consultant-AI (repo root)
                  os.path.dirname(                      # Domain-Intelligence
                      os.path.dirname(                  # Code
                          os.path.abspath(__file__))))
DATASET_DIR = os.path.join(BASE_DIR, "Domain-Intelligence", "Dataset")
INPUT_CSV   = os.path.join(DATASET_DIR, "battery.csv")
OUTPUT_CSV  = os.path.join(DATASET_DIR, "clean_battery.csv")

# Core columns that MUST have a value – rows missing any of these are dropped
REQUIRED_COLS = ["property", "name", "value", "doi"]

# Columns to fill with a placeholder instead of dropping
FILLABLE_COLS = ["warning", "tag", "info", "type"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    """Print a visible section separator to the log."""
    log.info("=" * 60)
    log.info("  %s", title)
    log.info("=" * 60)


def _null_report(df: pd.DataFrame, label: str) -> None:
    """Log null counts for every column."""
    log.info("--- Null counts (%s) ---", label)
    for col, n in df.isnull().sum().items():
        if n:
            log.info("  %-20s  %d", col, n)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load(path: str) -> pd.DataFrame:
    _section("STEP 1 – Load CSV")
    log.info("Reading: %s", path)
    df = pd.read_csv(path, low_memory=False)
    log.info("Loaded  : %d rows × %d columns", *df.shape)
    log.info("Columns : %s", list(df.columns))
    return df


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    _section("STEP 2 – Standardise column names")
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "_", regex=True)
    )
    log.info("Columns after: %s", list(df.columns))
    return df


def drop_all_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    _section("STEP 3 – Drop fully-empty columns")
    before = df.shape[1]
    df = df.dropna(axis=1, how="all")
    dropped = before - df.shape[1]
    log.info("Dropped %d all-null column(s); %d remaining", dropped, df.shape[1])
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    _section("STEP 4 – Remove duplicate rows")
    before = len(df)
    df = df.drop_duplicates()
    log.info("Removed %d duplicate row(s); %d remaining", before - len(df), len(df))
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    _section("STEP 5 – Handle missing values")
    _null_report(df, "before")

    # Fill non-critical columns
    for col in FILLABLE_COLS:
        if col in df.columns:
            filled = df[col].isnull().sum()
            df[col] = df[col].fillna("unknown")
            log.info("  Filled %-10s  → %d cell(s) set to 'unknown'", col, filled)

    # Drop rows that are missing any required column
    before = len(df)
    req_present = [c for c in REQUIRED_COLS if c in df.columns]
    df = df.dropna(subset=req_present)
    log.info("Dropped %d row(s) with null in required columns %s; %d remaining",
             before - len(df), req_present, len(df))

    _null_report(df, "after")
    return df


def fix_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    _section("STEP 6 – Fix data types")

    # value → numeric float
    if "value" in df.columns:
        before = len(df)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        coerced = df["value"].isnull().sum()
        if coerced:
            df = df.dropna(subset=["value"])
            log.info("Dropped %d row(s) with non-numeric 'value'; %d remaining",
                     before - len(df), len(df))
        log.info("'value' dtype: %s", df["value"].dtype)

    # date → datetime (best-effort)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        parsed_nulls = df["date"].isnull().sum()
        log.info("'date' parsed; %d unparseable date(s) set to NaT", parsed_nulls)

    return df


def normalise_text(df: pd.DataFrame) -> pd.DataFrame:
    _section("STEP 7 – Normalise text (strip whitespace)")
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    log.info("Stripped whitespace from %d string column(s)", len(str_cols))
    return df


def standardise_correctness(df: pd.DataFrame) -> pd.DataFrame:
    _section("STEP 8 – Standardise 'correctness' column")
    if "correctness" not in df.columns:
        log.info("Column 'correctness' not found – skipping")
        return df

    # Show unique non-null values
    uniq = df["correctness"].dropna().unique()
    log.info("Unique correctness values (non-null): %s", uniq)

    # T → True, anything starting with F (F1, F2, F3 …) → False, null → pd.NA
    def _map(val):
        if pd.isna(val):
            return None          # preserve nulls – don't drop rows for missing correctness
        s = str(val).strip().upper()
        if s == "T":
            return True
        if s.startswith("F"):
            return False
        return None              # unrecognised → None (not dropped)

    df["correctness"] = df["correctness"].map(_map)
    n_true  = (df["correctness"] == True).sum()   # noqa: E712
    n_false = (df["correctness"] == False).sum()  # noqa: E712
    n_null  = df["correctness"].isnull().sum()
    log.info("Correct (True): %d  |  Incorrect (False): %d  |  Unknown (null): %d",
             n_true, n_false, n_null)
    return df



def remove_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    _section("STEP 9 – Remove physically invalid values")
    if "value" not in df.columns:
        return df
    before = len(df)
    df = df[df["value"] > 0]
    log.info("Removed %d row(s) with value ≤ 0; %d remaining", before - len(df), len(df))
    return df


def reset_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    log.info("Index reset to 0…%d", len(df) - 1)
    return df


def export(df: pd.DataFrame, path: str) -> None:
    _section("STEP 10 – Export clean_battery.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    log.info("Saved  : %s", path)
    log.info("Rows   : %d", len(df))
    log.info("Columns: %d  →  %s", len(df.columns), list(df.columns))
    log.info("Size   : %.2f MB", size_mb)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    _section("CLEANING SUMMARY")
    log.info("%-25s  %8s  %8s", "", "BEFORE", "AFTER")
    log.info("%-25s  %8d  %8d", "Rows", len(df_raw), len(df_clean))
    log.info("%-25s  %8d  %8d", "Columns", df_raw.shape[1], df_clean.shape[1])
    log.info("%-25s  %8d  %8d", "Total nulls",
             int(df_raw.isnull().sum().sum()),
             int(df_clean.isnull().sum().sum()))
    kept_pct = 100.0 * len(df_clean) / len(df_raw) if len(df_raw) else 0
    log.info("Rows retained: %.1f%%", kept_pct)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.path.exists(INPUT_CSV):
        log.error("Input file not found: %s", INPUT_CSV)
        sys.exit(1)

    df_raw = load(INPUT_CSV)

    df = (df_raw
          .pipe(standardise_columns)
          .pipe(drop_all_null_columns)
          .pipe(drop_duplicates)
          .pipe(handle_missing_values)
          .pipe(fix_datatypes)
          .pipe(normalise_text)
          .pipe(standardise_correctness)
          .pipe(remove_invalid_values)
          .pipe(reset_index)
          )

    print_summary(df_raw, df)
    export(df, OUTPUT_CSV)
    log.info("Done! clean_battery.csv is ready for the AI consultant.")


if __name__ == "__main__":
    main()
