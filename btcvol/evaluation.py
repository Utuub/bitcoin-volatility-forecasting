"""Shared evaluation: RMSE/MAE, Mincer-Zarnowitz, and Diebold-Mariano."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config as C


def _valid(*arrays):
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    mask = ~np.any([np.isnan(a) for a in arrays], axis=0)
    return [a[mask] for a in arrays]


def evaluate_predictions(actual, predicted, model_name, horizon):
    a, p = _valid(actual, predicted)
    return {
        'model'  : model_name,
        'horizon': horizon,
        'n_obs'  : len(a),
        'rmse'   : round(np.sqrt(mean_squared_error(a, p)), 8),
        'mae'    : round(mean_absolute_error(a, p), 8),
    }


def mincer_zarnowitz(actual, predicted):
    """Regress actual on predicted. An unbiased forecast has alpha=0, beta=1."""
    a, p = _valid(actual, predicted)
    slope, intercept, r, _, _ = stats.linregress(p, a)
    return round(intercept, 6), round(slope, 6), round(r ** 2, 6)


def diebold_mariano(actual, pred1, pred2):
    """
    DM test on squared-error loss. Negative statistic = pred1 is more accurate.
    Returns (dm_stat, p_value).
    """
    a, p1, p2 = _valid(actual, pred1, pred2)
    d = (a - p1) ** 2 - (a - p2) ** 2
    dm_stat = np.mean(d) / np.sqrt(np.var(d, ddof=1) / len(d))
    p_val = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return round(dm_stat, 4), round(p_val, 4)


def _load_aligned(h):
    """Load naive/garch/xgb predictions for a horizon, restricted to common timestamps."""
    frames = {name: pd.read_csv(f'{C.DIR_PREDS}/{prefix}_{h}_predictions.csv')
              for name, prefix in [('naive', 'naive'), ('garch', 'garch'), ('xgboost', 'xgb')]}
    common = set.intersection(*(set(f['timestamp']) for f in frames.values()))
    return {name: f[f['timestamp'].isin(common)].sort_values('timestamp').reset_index(drop=True)
            for name, f in frames.items()}


def run_evaluation():
    metrics, mz, dm = [], [], []

    for h in C.HORIZONS:
        frames = _load_aligned(h)
        actual = frames['xgboost']['actual'].values

        for name, f in frames.items():
            metrics.append(evaluate_predictions(actual, f['predicted'].values, name, h))
            alpha, beta, r2 = mincer_zarnowitz(actual, f['predicted'].values)
            mz.append({'model': name, 'horizon': h, 'mz_alpha': alpha, 'mz_beta': beta, 'mz_r2': r2})

        xgb = frames['xgboost']['predicted'].values
        for other, label in [('garch', 'XGB vs GARCH'), ('naive', 'XGB vs Naive')]:
            stat, p = diebold_mariano(actual, xgb, frames[other]['predicted'].values)
            dm.append({'horizon': h, 'comparison': label, 'dm_stat': stat, 'p_value': p})

    metrics_df, mz_df, dm_df = pd.DataFrame(metrics), pd.DataFrame(mz), pd.DataFrame(dm)
    metrics_df.to_csv(f'{C.DIR_METRICS}/metrics_summary.csv', index=False)
    mz_df.to_csv(f'{C.DIR_METRICS}/mincer_zarnowitz.csv', index=False)
    dm_df.to_csv(f'{C.DIR_METRICS}/diebold_mariano.csv', index=False)

    print(metrics_df.pivot_table(index='model', columns='horizon', values=['rmse', 'mae']).round(6))
    print('\nMincer-Zarnowitz (ideal: alpha=0, beta=1)')
    print(mz_df.to_string(index=False))
    print('\nDiebold-Mariano (p < 0.05 = significant difference)')
    print(dm_df.to_string(index=False))
    return metrics_df, mz_df, dm_df


def add_lstm_results(metrics_df, mz_df, dm_df):
    """Extend all three tables with the LSTM, reading its saved predictions."""
    metric_rows, mz_rows, dm_rows = [], [], []

    for h in C.HORIZONS:
        lstm = pd.read_csv(f'{C.DIR_PREDS}/lstm_{h}_predictions.csv')
        metric_rows.append(evaluate_predictions(lstm['actual'].values, lstm['predicted'].values, 'lstm', h))
        alpha, beta, r2 = mincer_zarnowitz(lstm['actual'].values, lstm['predicted'].values)
        mz_rows.append({'model': 'lstm', 'horizon': h, 'mz_alpha': alpha, 'mz_beta': beta, 'mz_r2': r2})

        # DM comparisons on the timestamps all four models share
        frames = {name: pd.read_csv(f'{C.DIR_PREDS}/{prefix}_{h}_predictions.csv')
                  for name, prefix in [('lstm', 'lstm'), ('xgboost', 'xgb'), ('naive', 'naive'), ('garch', 'garch')]}
        common = set.intersection(*(set(f['timestamp']) for f in frames.values()))
        aligned = {name: f[f['timestamp'].isin(common)].sort_values('timestamp') for name, f in frames.items()}
        actual = aligned['xgboost']['actual'].values
        lstm_p = aligned['lstm']['predicted'].values
        for other, label in [('garch', 'LSTM vs GARCH'), ('naive', 'LSTM vs Naive'), ('xgboost', 'LSTM vs XGBoost')]:
            stat, p = diebold_mariano(actual, lstm_p, aligned[other]['predicted'].values)
            dm_rows.append({'horizon': h, 'comparison': label, 'dm_stat': stat, 'p_value': p})

    metrics_full = pd.concat([metrics_df, pd.DataFrame(metric_rows)], ignore_index=True)
    mz_full = pd.concat([mz_df, pd.DataFrame(mz_rows)], ignore_index=True)
    dm_full = pd.concat([dm_df, pd.DataFrame(dm_rows)], ignore_index=True)
    metrics_full.to_csv(f'{C.DIR_METRICS}/metrics_summary_full.csv', index=False)
    mz_full.to_csv(f'{C.DIR_METRICS}/mincer_zarnowitz_full.csv', index=False)
    dm_full.to_csv(f'{C.DIR_METRICS}/diebold_mariano_full.csv', index=False)

    print(metrics_full.pivot_table(index='model', columns='horizon', values=['rmse', 'mae']).round(6))
    print(mz_full.to_string(index=False))
    print(dm_full.to_string(index=False))
    return metrics_full, mz_full, dm_full
