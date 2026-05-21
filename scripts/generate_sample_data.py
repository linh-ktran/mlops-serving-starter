"""Generate synthetic aFRR-like hourly time-series data for local development.

Usage: python scripts/generate_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_data(
    start: str = "2025-07-01",
    end: str = "2026-04-30",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic-looking hourly energy market data."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h", tz="UTC", name="timestamp_utc")
    n = len(idx)

    hour = idx.hour
    dow = idx.dayofweek

    # Base daily/hourly pattern for aFRR prices (EUR/MW)
    daily_pattern = 20 + 10 * np.sin(2 * np.pi * hour / 24)
    weekly_pattern = np.where(dow >= 5, -5, 0)

    afrr_up = daily_pattern + weekly_pattern + rng.normal(0, 8, n)
    afrr_up = np.clip(afrr_up, 0, None)

    afrr_down = 0.6 * daily_pattern + weekly_pattern + rng.normal(0, 5, n)
    afrr_down = np.clip(afrr_down, 0, None)

    # Exogenous features
    fcr = 15 + rng.normal(0, 3, n)
    consumption = 50000 + 10000 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 2000, n)
    gas = 30 + 5 * np.sin(2 * np.pi * np.arange(n) / (24 * 30)) + rng.normal(0, 2, n)
    spot = 50 + 20 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 10, n)

    # Solar: only daylight hours
    solar = np.where((hour >= 7) & (hour <= 20), 3000 * np.sin(np.pi * (hour - 7) / 13), 0)
    solar = solar + rng.normal(0, 200, n)
    solar = np.clip(solar, 0, None)

    wind_on = 2000 + rng.normal(0, 800, n)
    wind_on = np.clip(wind_on, 0, None)

    wind_off = 800 + rng.normal(0, 400, n)
    wind_off = np.clip(wind_off, 0, None)

    df = pd.DataFrame(
        {
            "afrr_capacity_price_up": afrr_up,
            "afrr_capacity_price_down": afrr_down,
            "fcr_price_symmetric": fcr,
            "consumption_forecast": consumption,
            "gas_price_forecast": gas,
            "spot_price_forecast": spot,
            "solar_forecast": solar,
            "wind_onshore_forecast": wind_on,
            "wind_offshore_forecast": wind_off,
        },
        index=idx,
    )
    return df


def main() -> None:
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    df = generate_sample_data()
    out_path = out_dir / "sample.csv"
    df.to_csv(out_path)
    print(f"Generated {len(df)} rows → {out_path}")


if __name__ == "__main__":
    main()
