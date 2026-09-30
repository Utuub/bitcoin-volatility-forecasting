"""Forward-looking realized volatility targets."""
import numpy as np

from . import config as C

TARGET_COLS = ['future_1d_vol', 'future_1w_vol', 'future_1m_vol']


def realized_vol_forward(returns, bars):
    """
    Realized volatility over the NEXT `bars` rows: sqrt(sum of squared returns).
    Reverse-rolling the squared returns and shifting aligns each future window
    sum to the current row.
    """
    squared = returns ** 2
    rv = squared[::-1].rolling(window=bars, min_periods=bars).sum()[::-1].shift(-(bars - 1))
    return np.sqrt(rv)


def add_targets(df):
    for h, bars in C.HORIZON_BARS.items():
        df[f'future_{h}_vol'] = realized_vol_forward(df['log_return'], bars)

    for col in TARGET_COLS:
        print(f'{col}: {df[col].notna().sum()} valid rows')
        assert (df[col].dropna() >= 0).all(), f'Negative values in {col}'
    return df
