"""
make the Manufacturing_dataset.csv a bit clean for AI consultant.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

BASE_DIR    = os.path.dirname(                          # Battery-Consultant-AI (repo root)
                  os.path.dirname(                      # Manufacturing
                      os.path.dirname(                  # Code
                          os.path.abspath(__file__))))
DATASET_DIR = os.path.join(BASE_DIR, "Manufacturing", "Dataset")
INPUT_CSV   = os.path.join(DATASET_DIR, "Manufacturing_dataset.csv")
OUTPUT_CSV  = os.path.join(DATASET_DIR, "clean_Manufacturing.csv")

# Sensor columns used for outlier detection and rolling stats
SENSOR_COLS = [
    "temperature_c",
    "machine_speed_rpm",
    "production_quality_score",
    "vibration_level_mm_s",
    "energy_consumption_kwh",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _section(title: str) -> None:
    log.info("=" * 60)
    log.info("  %s", title)
    log.info("=" * 60)


# Pipeline steps

def load(path: str) -> pd.DataFrame:
    _section("STEP 1 – Load CSV")
    log.info("Reading: %s", path)
    df = pd.read_csv(path)
    log.info("Loaded  : %d rows × %d columns", *df.shape)
    log.info("Columns : %s", list(df.columns))
    return df


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    _section("STEP 2 – Standardise column names")
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[°/()]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    # Manual fixes for special characters left over from unit strings
    rename_map = {
        "temperature_c":            "temperature_c",
        "machine_speed_rpm":        "machine_speed_rpm",
        "production_quality_score": "production_quality_score",
        "vibration_level_mms":      "vibration_level_mm_s",
        "energy_consumption_kwh":   "energy_consumption_kwh",
        "optimal_conditions":       "is_optimal",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    log.info("Columns after: %s", list(df.columns))
    return df


def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    _section("STEP 3 – Parse timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    n_bad = df["timestamp"].isnull().sum()
    if n_bad:
        df = df.dropna(subset=["timestamp"])
        log.info("Dropped %d un-parseable timestamp rows", n_bad)
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("Timestamp range: %s  →  %s", df["timestamp"].min(), df["timestamp"].max())
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    _section("STEP 4 – Remove duplicates")
    before = len(df)
    df = df.drop_duplicates(subset=["timestamp"])
    log.info("Removed %d duplicate timestamp row(s); %d remaining", before - len(df), len(df))
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    _section("STEP 5 – Remove IQR outliers")
    before = len(df)
    mask = pd.Series(True, index=df.index)
    for col in SENSOR_COLS:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        col_outliers = (df[col] < lo) | (df[col] > hi)
        log.info("  %-35s  Q1=%.3f  Q3=%.3f  outliers=%d", col, Q1, Q3, col_outliers.sum())
        mask &= ~col_outliers
    df = df[mask].reset_index(drop=True)
    log.info("Removed %d outlier row(s); %d remaining", before - len(df), len(df))
    return df


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    _section("STEP 6 – Extract time features")
    df["hour"]        = df["timestamp"].dt.hour
    df["day"]         = df["timestamp"].dt.day
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["week"]        = df["timestamp"].dt.isocalendar().week.astype(int)

    def _shift(h: int) -> str:
        if 6 <= h < 14:
            return "Morning"
        elif 14 <= h < 22:
            return "Afternoon"
        return "Night"

    df["shift"] = df["hour"].map(_shift)
    log.info("Added: hour, day, day_of_week, week, shift")
    return df


def add_rolling_features(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Merge 60-minute rolling mean + std for every sensor column back onto each row.
    This is the key 'merge' step — each minute-level record gains hourly context.
    """
    _section(f"STEP 7 – Rolling {window}-min window merge (mean + std)")
    df = df.set_index("timestamp").sort_index()

    for col in SENSOR_COLS:
        if col not in df.columns:
            continue
        rolling = df[col].rolling(f"{window}min", min_periods=1)
        df[f"{col}_roll_mean"] = rolling.mean().round(4)
        df[f"{col}_roll_std"]  = rolling.std().fillna(0).round(4)
        log.info("  Added %s_roll_mean / %s_roll_std", col, col)

    df = df.reset_index()
    log.info("Rolling merge complete — %d new columns added", len(SENSOR_COLS) * 2)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    _section("STEP 8 – Derived features")
    # Efficiency ratio: how much quality output per unit of energy consumed
    if "production_quality_score" in df.columns and "energy_consumption_kwh" in df.columns:
        df["efficiency_ratio"] = (
            df["production_quality_score"] / df["energy_consumption_kwh"].replace(0, np.nan)
        ).round(4)
        log.info("Added efficiency_ratio = quality_score / energy_consumption")

    # Energy per RPM (normalised energy load at given speed)
    if "energy_consumption_kwh" in df.columns and "machine_speed_rpm" in df.columns:
        df["energy_per_rpm"] = (
            df["energy_consumption_kwh"] / df["machine_speed_rpm"].replace(0, np.nan)
        ).round(6)
        log.info("Added energy_per_rpm")

    return df


def final_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    _section("STEP 9 – Final cleanup")
    before = len(df)
    # Drop any rows that still have nulls in the original sensor columns
    df = df.dropna(subset=SENSOR_COLS)
    log.info("Dropped %d row(s) with remaining nulls in sensor cols; %d remaining",
             before - len(df), len(df))
    df = df.reset_index(drop=True)
    # Report remaining nulls
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        log.info("Remaining nulls:\n%s", nulls)
    else:
        log.info("No remaining nulls in sensor columns ✓")
    return df


def export(df: pd.DataFrame, path: str) -> None:
    _section("STEP 10 – Export clean_Manufacturing.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    size_kb = os.path.getsize(path) / 1024
    log.info("Saved  : %s", path)
    log.info("Rows   : %d", len(df))
    log.info("Columns: %d  →  %s", len(df.columns), list(df.columns))
    log.info("Size   : %.1f KB", size_kb)


# Summary

def print_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    _section("CLEANING SUMMARY")
    log.info("%-30s  %8s  %8s", "", "BEFORE", "AFTER")
    log.info("%-30s  %8d  %8d", "Rows",    len(df_raw), len(df_clean))
    log.info("%-30s  %8d  %8d", "Columns", df_raw.shape[1], df_clean.shape[1])
    log.info("%-30s  %8d  %8d", "Total nulls",
             int(df_raw.isnull().sum().sum()),
             int(df_clean.isnull().sum().sum()))
    pct = 100.0 * len(df_clean) / len(df_raw) if len(df_raw) else 0
    log.info("Rows retained : %.1f%%", pct)
    log.info("New columns   : %s", [c for c in df_clean.columns if c not in df_raw.columns])


# Main

def main() -> None:
    if not os.path.exists(INPUT_CSV):
        log.error("Input file not found: %s", INPUT_CSV)
        sys.exit(1)

    df_raw = load(INPUT_CSV)

    df = (df_raw.copy()
          .pipe(standardise_columns)
          .pipe(parse_timestamp)
          .pipe(drop_duplicates)
          .pipe(remove_outliers)
          .pipe(extract_time_features)
          .pipe(add_rolling_features, window=60)
          .pipe(add_derived_features)
          .pipe(final_cleanup)
          )

    print_summary(df_raw, df)
    export(df, OUTPUT_CSV)
    log.info("Done! clean_Manufacturing.csv is ready for the AI consultant.")


if __name__ == "__main__":
    main()
