"""Feature engineering. Every feature uses only data available before the current bar."""
import numpy as np
import pandas as pd

from . import config as C
from .targets import TARGET_COLS

RAW_COLS = [
    'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
    'num_trades', 'taker_buy_base_volume', 'taker_buy_quote_volume',
    'log_return', 'buy_pressure_ratio', 'hour_of_day', 'day_of_week',
]


def _rv(x):
    return np.sqrt(np.sum(x ** 2))


def add_return_lags(df):
    for lag in C.LAG_DEPTHS:
        df[f'ret_lag_{lag}h'] = df['log_return'].shift(lag)

    # Past realized vol, shifted 1 to avoid current-bar leakage
    for name, bars in [('1d', C.BARS_PER_DAY), ('1w', C.BARS_PER_WEEK), ('1m', C.BARS_PER_MONTH)]:
        df[f'past_rv_{name}'] = df['log_return'].rolling(bars).apply(_rv, raw=True).shift(1)
    return df


def add_rolling_stats(df):
    r = df['log_return']
    for w in C.ROLLING_WINDOWS:
        df[f'roll_mean_ret_{w}h'] = r.rolling(w).mean().shift(1)
        df[f'roll_std_ret_{w}h']  = r.rolling(w).std().shift(1)
        df[f'roll_skew_ret_{w}h'] = r.rolling(w).skew().shift(1)
        df[f'roll_kurt_ret_{w}h'] = r.rolling(w).kurt().shift(1)
    return df


def add_volume_features(df):
    for lag in [1, 6, 24, 168]:
        df[f'vol_lag_{lag}h']    = df['volume'].shift(lag)
        df[f'trades_lag_{lag}h'] = df['num_trades'].shift(lag)

    df['vol_change_1h']  = (df['volume'] / df['volume'].shift(1).replace(0, np.nan)).shift(1)
    df['vol_change_24h'] = (df['volume'] / df['volume'].shift(24).replace(0, np.nan)).shift(1)

    for w in [24, 168, 720]:
        df[f'roll_avg_vol_{w}h'] = df['volume'].rolling(w).mean().shift(1)
    return df


def add_market_pressure(df):
    vol = df['volume'].replace(0, np.nan)
    buy = df['taker_buy_base_volume']

    df['buy_share'] = (buy / vol).shift(1)
    # positive = buy heavy, negative = sell heavy
    df['buy_sell_imbalance'] = ((buy - (df['volume'] - buy)) / vol).shift(1)

    df['roll_buy_pressure_24h']  = df['buy_share'].shift(1).rolling(24).mean()
    df['roll_buy_pressure_168h'] = df['buy_share'].shift(1).rolling(168).mean()

    for lag in [1, 24]:
        df[f'taker_buy_usd_lag_{lag}h'] = df['taker_buy_quote_volume'].shift(lag)
    return df


def add_price_and_time(df):
    # High-low range as a fraction of close (intrabar volatility proxy)
    df['hl_range'] = ((df['high'] - df['low']) / df['close'].replace(0, np.nan)).shift(1)
    for w in [24, 168]:
        df[f'roll_hl_range_{w}h'] = df['hl_range'].shift(1).rolling(w).mean()

    # Cyclical time encoding
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['dow_sin']  = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']  = np.cos(2 * np.pi * df['day_of_week'] / 7)
    return df


def categorize(col):
    if 'ret_lag' in col: return 'return_lag'
    if 'past_rv' in col: return 'volatility_lag'
    if 'roll_' in col and 'ret' in col: return 'rolling_return_stat'
    if 'roll_avg_vol' in col or 'vol_lag' in col or 'vol_change' in col or 'trades' in col:
        return 'volume_activity'
    if 'buy' in col or 'taker' in col or 'imbalance' in col: return 'market_pressure'
    if 'hl_range' in col: return 'price_range'
    if 'hour' in col or 'dow' in col: return 'time_cyclical'
    return 'other'


def build_features(df):
    """Add every feature group and return (df, feature_cols)."""
    for step in (add_return_lags, add_rolling_stats, add_volume_features,
                 add_market_pressure, add_price_and_time):
        df = step(df)

    feature_cols = [c for c in df.columns if c not in TARGET_COLS + RAW_COLS]
    manifest = pd.DataFrame({'feature': feature_cols,
                             'category': [categorize(c) for c in feature_cols]})
    manifest.to_csv(f'{C.DIR_NOTES}/feature_manifest.csv', index=False)
    print(f'Features: {len(feature_cols)}')
    print(manifest['category'].value_counts().to_string())
    return df, feature_cols


def freeze_dataset(df, feature_cols):
    """Drop rows missing any feature or target and save the modeling dataset."""
    before = len(df)
    df_model = df.dropna(subset=feature_cols + TARGET_COLS).copy()
    print(f'Rows: {before} -> {len(df_model)} (dropped {before - len(df_model)})')
    df_model.to_csv(f'{C.DIR_FEATURES}/modeling_dataset_1h.csv')
    return df_model
