"""All pipeline settings in one place."""
import os

# ── File paths ────────────────────────────────────────────────────────────────
RAW_DATA_PATH = 'data/btc_1h_data_2018_to_2025.csv'

OUTPUT_ROOT  = 'outputs'
DIR_CLEAN    = f'{OUTPUT_ROOT}/clean'
DIR_FEATURES = f'{OUTPUT_ROOT}/features'
DIR_MODELS   = f'{OUTPUT_ROOT}/models'
DIR_PREDS    = f'{OUTPUT_ROOT}/predictions'
DIR_FIGURES  = f'{OUTPUT_ROOT}/figures'
DIR_METRICS  = f'{OUTPUT_ROOT}/metrics'
DIR_NOTES    = f'{OUTPUT_ROOT}/notes'
ALL_DIRS = [DIR_CLEAN, DIR_FEATURES, DIR_MODELS, DIR_PREDS, DIR_FIGURES, DIR_METRICS, DIR_NOTES]

# ── Frequency ─────────────────────────────────────────────────────────────────
FREQ           = '1h'
BARS_PER_DAY   = 24
BARS_PER_WEEK  = 168   # 24 * 7
BARS_PER_MONTH = 720   # 24 * 30

# ── Forecast horizons (in days) ───────────────────────────────────────────────
HORIZONS     = {'1d': 1, '1w': 7, '1m': 30}
HORIZON_BARS = {'1d': BARS_PER_DAY, '1w': BARS_PER_WEEK, '1m': BARS_PER_MONTH}

# ── Chronological split dates ─────────────────────────────────────────────────
TRAIN_START = '2018-01-01'
VAL_START   = '2023-01-01'   # ~5 years training
TEST_START  = '2024-01-01'   # ~1 year validation
TEST_END    = '2026-02-01'

# ── Walk-forward validation ───────────────────────────────────────────────────
WF_STEP_DAYS       = 30      # re-evaluate every 30 days
WF_MAX_WINDOW_DAYS = 365     # switch from expanding to fixed at 1 year

# ── GARCH ─────────────────────────────────────────────────────────────────────
GARCH_REFIT_DAYS = 30        # refit every 30 days during the test period
RETURN_SCALE     = 100       # scale returns for numerical stability

# ── Feature engineering ───────────────────────────────────────────────────────
LAG_DEPTHS      = [1, 2, 3, 6, 12, 24, 48, 168]   # hourly lags
ROLLING_WINDOWS = [24, 48, 168, 336, 720]          # hourly rolling windows

# ── XGBoost (tuned on validation) ─────────────────────────────────────────────
XGB_PARAMS = {
    'n_estimators'         : 100,
    'max_depth'            : 5,
    'learning_rate'        : 0.08,
    'subsample'            : 0.8,
    'colsample_bytree'     : 0.8,
    'random_state'         : 42,
    'n_jobs'               : -1,
    'early_stopping_rounds': 30,
}

# ── LSTM ──────────────────────────────────────────────────────────────────────
LSTM_SEQ_LEN = 72    # three days of hourly bars
LSTM_EPOCHS  = 50
LSTM_BATCH   = 128
LSTM_LR      = 0.0005

RANDOM_SEED = 42


def make_output_dirs():
    for d in ALL_DIRS:
        os.makedirs(d, exist_ok=True)
