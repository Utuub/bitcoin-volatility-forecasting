"""Data ingestion, cleaning, and log returns."""
import numpy as np
import pandas as pd

from . import config as C

KEEP_COLS = [
    'Open time', 'Open', 'High', 'Low', 'Close',
    'Volume', 'Quote asset volume', 'Number of trades',
    'Taker buy base asset volume', 'Taker buy quote asset volume',
]
SNAKE_COLS = [
    'timestamp', 'open', 'high', 'low', 'close',
    'volume', 'quote_asset_volume', 'num_trades',
    'taker_buy_base_volume', 'taker_buy_quote_volume',
]


def load_and_clean(path=C.RAW_DATA_PATH):
    """Load the raw hourly CSV and return a clean frame indexed by timestamp."""
    raw = pd.read_csv(path)
    print(f'Raw shape: {raw.shape}')

    df = raw[KEEP_COLS].copy()
    df.columns = SNAKE_COLS

    # Rows are consecutive hourly bars, so timestamps are generated as a continuous
    # hourly range from TRAIN_START rather than parsed from 'Open time'. This also
    # handles copies of the CSV where spreadsheet software mangled that column.
    df['timestamp'] = pd.date_range(start=pd.to_datetime(C.TRAIN_START), periods=len(df), freq='1h')
    df = df.sort_values('timestamp').reset_index(drop=True)
    assert df['timestamp'].is_monotonic_increasing, 'Timestamps are not monotonic'

    print(f'Duplicate timestamps : {df["timestamp"].duplicated().sum()}')
    df = df.drop_duplicates(subset=['timestamp'])

    print(f'High < Low violations: {(df["high"] < df["low"]).sum()}')
    print(f'Negative volume rows : {(df["volume"] < 0).sum()}')
    print(f'Negative trade rows  : {(df["num_trades"] < 0).sum()}')

    time_diffs = df['timestamp'].diff().dropna()
    gaps = time_diffs[time_diffs > pd.Timedelta('1h') * 1.5]
    print(f'Time gaps detected   : {len(gaps)}')

    df = df.set_index('timestamp')
    df.to_csv(f'{C.DIR_CLEAN}/btc_1h_clean.csv')
    print(f'Clean shape: {df.shape}, {df.index.min()} -> {df.index.max()}')
    return df


def add_log_returns(df):
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df = df.iloc[1:].copy()                       # first return is undefined
    r = df['log_return']
    print(f'Log returns: n={len(df)}, mean={r.mean():.6f}, std={r.std():.6f}')
    return df


def add_buy_pressure_ratio(df):
    """Unshifted taker-buy share, used only for EDA plots (excluded from features)."""
    df['buy_pressure_ratio'] = df['taker_buy_base_volume'] / df['volume'].replace(0, np.nan)
    return df


def save_data_dictionary():
    data_dict = pd.DataFrame([
        ['timestamp',              'Bar open timestamp (UTC)',                  'datetime', 'raw'],
        ['open',                   'Opening price of the bar',                  'USD',      'raw'],
        ['high',                   'Highest price during the bar',              'USD',      'raw'],
        ['low',                    'Lowest price during the bar',               'USD',      'raw'],
        ['close',                  'Closing price of the bar',                  'USD',      'raw'],
        ['volume',                 'BTC traded volume',                         'BTC',      'raw'],
        ['quote_asset_volume',     'USD equivalent of volume traded',           'USD',      'raw'],
        ['num_trades',             'Number of individual trades in the bar',    'count',    'raw'],
        ['taker_buy_base_volume',  'BTC volume from market buy (taker) orders', 'BTC',      'raw'],
        ['taker_buy_quote_volume', 'USD volume from market buy (taker) orders', 'USD',      'raw'],
        ['log_return',             'Log return from bar to bar close',          'ratio',    'engineered'],
        ['future_1d_vol',          'Realized vol over next 24 hours (target)',  'ratio',    'target'],
        ['future_1w_vol',          'Realized vol over next 168 hours (target)', 'ratio',    'target'],
        ['future_1m_vol',          'Realized vol over next 720 hours (target)', 'ratio',    'target'],
    ], columns=['variable', 'description', 'units', 'type'])
    data_dict.to_csv(f'{C.DIR_NOTES}/data_dictionary.csv', index=False)
    return data_dict
