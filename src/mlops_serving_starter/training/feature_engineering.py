"""Feature engineering for aFRR capacity price time-series models.

Builds lag, rolling, calendar and exogenous features for XGBoost.
"""

from __future__ import annotations

import logging

import holidays
import pandas as pd

logger = logging.getLogger(__name__)

# Exogenous inputs — in production these come from external APIs
FORECAST_FEATURES = [
    "fcr_price_symmetric",
    "consumption_forecast",
    "gas_price_forecast",
    "spot_price_forecast",
    "solar_forecast",
    "wind_onshore_forecast",
    "wind_offshore_forecast",
]

ROLLING_FEATURES: dict[str, tuple[str, int]] = {
    "rolling_mean_7": ("mean", 7),
    "rolling_min_7": ("min", 7),
    "rolling_max_7": ("max", 7),
}

LAG_DAYS = [1, 2, 3, 7, 14, 28]
HOURS_PER_DAY = 24


def prepare_features_for_horizon(
    df: pd.DataFrame,
    target: str,
    horizon: int,
) -> pd.DataFrame:
    """Build all features for one forecast horizon. Filter by hour *after* calling this."""
    df = df.copy()

    df = add_holiday_status(df)
    df = add_weekend_status(df)
    df = add_lag_features(df, target, horizon)
    df = add_rolling_features(df, target, horizon)
    df = forward_fill_forecast_features(df)
    df = select_features(df, target, horizon)

    logger.info("Feature engineering complete: %d samples, horizon %d", len(df), horizon)
    return df


def add_holiday_status(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary holiday_status column (French public holidays)."""
    index = df.index
    if index.tz is None:
        index = index.tz_localize("UTC")
    ts_paris = index.tz_convert("Europe/Paris")
    french_holidays = holidays.France()
    df["holiday_status"] = [1 if dt.date() in french_holidays else 0 for dt in ts_paris]
    return df


def add_weekend_status(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary weekend_status column (Sat=1, Sun=1, else 0)."""
    index = df.index
    if index.tz is None:
        index = index.tz_localize("UTC")
    ts_paris = index.tz_convert("Europe/Paris")
    df["weekend_status"] = [1 if dt.weekday() in (5, 6) else 0 for dt in ts_paris]
    return df


def add_lag_features(df: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """Add lag features shifted by (lag_days + horizon - 1) * 24 hours."""
    for lag_days in LAG_DAYS:
        shift_hours = HOURS_PER_DAY * (lag_days + horizon - 1)
        df[f"{target}_lag_{lag_days}_h{horizon}"] = df[target].shift(shift_hours)
    return df


def add_rolling_features(df: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """Add rolling statistics (mean/min/max over 7 days), shifted to avoid leakage."""
    shift_hours = HOURS_PER_DAY * horizon
    for col_name, (stat, window_days) in ROLLING_FEATURES.items():
        window_hours = HOURS_PER_DAY * window_days
        rolled = getattr(df[target].rolling(window=window_hours), stat)()
        df[col_name] = rolled.shift(shift_hours)
    return df


def forward_fill_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill exogenous forecast features per hour-of-day."""
    df = df.copy()
    present = [f for f in FORECAST_FEATURES if f in df.columns]
    hour_groups = df.index.hour
    for feature in present:
        if df[feature].isna().any():
            df[feature] = df.groupby(hour_groups)[feature].ffill()
    return df


def select_features(df: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """Return only the columns needed for training/prediction."""
    lag_cols = [f"{target}_lag_{d}_h{horizon}" for d in LAG_DAYS]
    rolling_cols = list(ROLLING_FEATURES.keys())
    forecast_cols = [f for f in FORECAST_FEATURES if f in df.columns]

    required = [target] + lag_cols + rolling_cols + forecast_cols + ["holiday_status", "weekend_status"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after feature engineering: {missing}")

    return df[required]
