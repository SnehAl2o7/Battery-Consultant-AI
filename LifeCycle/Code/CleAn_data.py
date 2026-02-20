"""
make the Li_S_Battery_Thermal_Failure_Dataset.csv a bit clean for AI consultant.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

BASE_DIR    = os.path.dirname(                          # Battery-Consultant-AI (repo root)
                  os.path.dirname(                      # LifeCycle
                      os.path.dirname(                  # Code
                          os.path.abspath(__file__))))
DATASET_DIR = os.path.join(BASE_DIR, "LifeCycle", "Dataset")
INPUT_CSV   = os.path.join(DATASET_DIR, "Li_S_Battery_Thermal_Failure_Dataset.csv")
OUTPUT_CSV  = os.path.join(DATASET_DIR, "clean_Li_S.csv")

TEMP_SENSORS   = ["temp_sensor_1", "temp_sensor_2", "temp_sensor_3"]
ROLLING_COLS   = ["avg_temp", "charge_rate", "discharge_rate"]
WINDOW_MINUTES = 30

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
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    log.info("Columns after: %s", list(df.columns))
    return df


def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    _section("STEP 3 – Parse timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    n_bad = df["timestamp"].isnull().sum()
    if n_bad:
        df = df.dropna(subset=["timestamp"])
        log.info("Dropped %d unparseable timestamp row(s)", n_bad)
    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info("Timestamp range: %s  →  %s", df["timestamp"].min(), df["timestamp"].max())
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    _section("STEP 4 – Remove duplicates")
    before = len(df)
    df = df.drop_duplicates()
    log.info("Removed %d duplicate row(s); %d remaining", before - len(df), len(df))
    return df


def merge_temp_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the 3 individual temp sensor readings into combined features.
    Raw sensor columns are kept for traceability.
    """
    _section("STEP 5 – Merge temperature sensors")
    present = [c for c in TEMP_SENSORS if c in df.columns]
    if not present:
        log.warning("No temp sensor columns found — skipping merge")
        return df

    df["avg_temp"]    = df[present].mean(axis=1).round(4)
    df["max_temp"]    = df[present].max(axis=1).round(4)
    df["min_temp"]    = df[present].min(axis=1).round(4)
    df["temp_spread"] = (df["max_temp"] - df["min_temp"]).round(4)

    log.info("Merged %s → avg_temp, max_temp, min_temp, temp_spread", present)
    log.info("  avg_temp  range: %.2f – %.2f", df["avg_temp"].min(), df["avg_temp"].max())
    log.info("  temp_spread max: %.4f", df["temp_spread"].max())
    return df


def flag_temp_extremes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag rows where any raw sensor exceeds the IQR upper fence.
    We FLAG (not drop) — high temps ARE the failure signal.
    """
    _section("STEP 6 – Flag temperature extremes (IQR-based)")
    flag = pd.Series(False, index=df.index)
    for col in [c for c in TEMP_SENSORS if c in df.columns]:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        upper  = Q3 + 1.5 * (Q3 - Q1)
        extreme = df[col] > upper
        log.info("  %-20s  IQR upper=%.2f  flagged=%d", col, upper, extreme.sum())
        flag |= extreme

    df["is_temp_extreme"] = flag.astype(int)
    log.info("Total rows flagged as temp extreme: %d / %d", df["is_temp_extreme"].sum(), len(df))
    return df


def add_thermal_stress(df: pd.DataFrame) -> pd.DataFrame:
    _section("STEP 7 – Derived thermal features")
    if "avg_temp" in df.columns and "ambient_temp" in df.columns:
        df["thermal_stress"] = (df["avg_temp"] - df["ambient_temp"]).round(4)
        log.info("Added thermal_stress = avg_temp − ambient_temp")
        log.info("  thermal_stress range: %.4f – %.4f",
                 df["thermal_stress"].min(), df["thermal_stress"].max())

    if "charge_rate" in df.columns and "discharge_rate" in df.columns:
        df["charge_discharge_ratio"] = (
            df["charge_rate"] / df["discharge_rate"].replace(0, np.nan)
        ).round(4)
        log.info("Added charge_discharge_ratio = charge_rate / discharge_rate")

    return df


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    _section("STEP 8 – Extract time features")
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


def add_rolling_features(df: pd.DataFrame, window: int = WINDOW_MINUTES) -> pd.DataFrame:
    """
    30-minute rolling mean + std merged back onto every row.
    Gives the AI consultant per-row access to recent trend context.
    """
    _section(f"STEP 9 – Rolling {window}-min window merge (mean + std)")
    df = df.set_index("timestamp").sort_index()

    for col in ROLLING_COLS:
        if col not in df.columns:
            continue
        rolling = df[col].rolling(f"{window}min", min_periods=1)
        df[f"{col}_roll_mean"] = rolling.mean().round(4)
        df[f"{col}_roll_std"]  = rolling.std().fillna(0).round(4)
        log.info("  Added %s_roll_mean / %s_roll_std", col, col)

    df = df.reset_index()
    log.info("Rolling merge complete — %d new columns added", len(ROLLING_COLS) * 2)
    return df


def clean_failure_label(df: pd.DataFrame) -> pd.DataFrame:
    _section("STEP 10 – Validate failure label consistency")
    mapping = {0: "Normal", 1: "Overheat_Alert", 2: "Thermal_Runaway"}
    if "failure_label" in df.columns and "failure_condition" in df.columns:
        # Ensure numeric label and text label agree
        df["failure_condition_mapped"] = df["failure_label"].map(mapping)
        mismatches = (df["failure_condition"] != df["failure_condition_mapped"]).sum()
        log.info("Label ↔ condition mismatches: %d", mismatches)
        if mismatches == 0:
            df = df.drop(columns=["failure_condition_mapped"])
            log.info("Labels consistent ✓ — dropped temp check column")
        else:
            log.warning("Mismatches found! Keeping both columns for inspection.")

    counts = df.groupby(["failure_label", "failure_condition"]).size()
    log.info("Failure distribution:\n%s", counts.to_string())
    return df


def final_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    _section("STEP 11 – Final cleanup")
    before = len(df)
    core_cols = TEMP_SENSORS + ["ambient_temp", "charge_rate", "discharge_rate"]
    present_core = [c for c in core_cols if c in df.columns]
    df = df.dropna(subset=present_core)
    log.info("Dropped %d row(s) with nulls in core cols; %d remaining",
             before - len(df), len(df))
    df = df.reset_index(drop=True)
    remaining_nulls = df.isnull().sum()
    remaining_nulls = remaining_nulls[remaining_nulls > 0]
    if len(remaining_nulls):
        log.info("Remaining nulls:\n%s", remaining_nulls.to_string())
    else:
        log.info("No remaining nulls ✓")
    return df


def export(df: pd.DataFrame, path: str) -> None:
    _section("STEP 12 – Export clean_Li_S.csv")
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
    orig_cols = set(df_raw.columns.str.lower().str.replace(r"[^a-z0-9]", "_", regex=True))
    new_cols  = [c for c in df_clean.columns if c not in orig_cols]
    log.info("New columns   : %s", new_cols)


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
          .pipe(merge_temp_sensors)
          .pipe(flag_temp_extremes)
          .pipe(add_thermal_stress)
          .pipe(extract_time_features)
          .pipe(add_rolling_features, window=WINDOW_MINUTES)
          .pipe(clean_failure_label)
          .pipe(final_cleanup)
          )

    print_summary(df_raw, df)
    export(df, OUTPUT_CSV)
    log.info("Done! clean_Li_S.csv is ready for the AI consultant.")


if __name__ == "__main__":
    main()
