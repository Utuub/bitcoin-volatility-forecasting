"""All figures. Each function saves a PNG to outputs/figures and closes it."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config as C

sns.set_theme(style='darkgrid')
YEAR_FMT = mdates.DateFormatter('%Y')
MODEL_COLORS = {'naive': 'orange', 'garch': 'crimson', 'xgboost': 'steelblue', 'lstm': '#1D9E75'}
MODEL_ORDER = ['naive', 'garch', 'xgboost', 'lstm']


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(f'{C.DIR_FIGURES}/{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def lstm_available():
    return all(os.path.exists(f'{C.DIR_PREDS}/lstm_{h}_predictions.csv') for h in C.HORIZONS)


def load_all_preds(h, include_lstm=False):
    """Every model's predictions for one horizon, aligned on shared timestamps."""
    prefixes = {'naive': 'naive', 'garch': 'garch', 'xgboost': 'xgb'}
    if include_lstm:
        prefixes['lstm'] = 'lstm'
    dfs = {name: pd.read_csv(f'{C.DIR_PREDS}/{p}_{h}_predictions.csv', parse_dates=['timestamp'])
           for name, p in prefixes.items()}
    common = set.intersection(*(set(d['timestamp']) for d in dfs.values()))
    return {name: d[d['timestamp'].isin(common)].sort_values('timestamp').reset_index(drop=True)
            for name, d in dfs.items()}


# ── EDA ───────────────────────────────────────────────────────────────────────

def eda_price_returns_vol(df):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    axes[0].plot(df.index, df['close'], color='steelblue', linewidth=0.6)
    axes[0].set(title='Bitcoin Close Price (USD)', ylabel='Price (USD)')

    axes[1].plot(df.index, df['log_return'], color='darkorange', linewidth=0.4, alpha=0.8)
    axes[1].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[1].set(title='Hourly Log Returns', ylabel='Log Return')

    roll_vol = df['log_return'].rolling(C.BARS_PER_MONTH).std() * np.sqrt(C.BARS_PER_MONTH)
    axes[2].plot(df.index, roll_vol, color='crimson', linewidth=0.6)
    axes[2].set(title='Rolling 30-Day Realized Volatility', ylabel='Volatility')
    for ax in axes:
        ax.xaxis.set_major_formatter(YEAR_FMT)
    _save(fig, 'eda_price_returns_vol')


def eda_return_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df['log_return'].dropna(), bins=200, color='steelblue', edgecolor='none', alpha=0.8)
    axes[0].set(title='Return Distribution', xlabel='Log Return', ylabel='Frequency')

    sc = axes[1].scatter(df.index, df['log_return'], c=df['log_return'].abs(),
                         cmap='RdYlGn_r', s=0.5, alpha=0.6)
    axes[1].set(title='Returns Colored by Magnitude', xlabel='Date', ylabel='Log Return')
    fig.colorbar(sc, ax=axes[1], label='|Return|')
    _save(fig, 'eda_return_distribution')


def eda_volume_pressure(df):
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    axes[0].plot(df.index, df['volume'].rolling(C.BARS_PER_WEEK).mean(), color='purple', linewidth=0.7)
    axes[0].set(title='7-Day Rolling Average Volume (BTC)', ylabel='Volume (BTC)')

    axes[1].plot(df.index, df['buy_pressure_ratio'].rolling(C.BARS_PER_WEEK).mean(),
                 color='teal', linewidth=0.7)
    axes[1].axhline(0.5, color='black', linewidth=0.5, linestyle='--', label='Neutral (0.5)')
    axes[1].set(title='7-Day Rolling Buy Pressure Ratio (Taker Buy / Total Volume)',
                ylabel='Buy Pressure Ratio')
    axes[1].legend()
    for ax in axes:
        ax.xaxis.set_major_formatter(YEAR_FMT)
    _save(fig, 'eda_volume_pressure')


def targets_all_horizons(df):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    specs = [('future_1d_vol', '1-Day Ahead RV', 'steelblue'),
             ('future_1w_vol', '1-Week Ahead RV', 'darkorange'),
             ('future_1m_vol', '1-Month Ahead RV', 'crimson')]
    for ax, (col, label, color) in zip(axes, specs):
        ax.plot(df.index, df[col], color=color, linewidth=0.5, alpha=0.8)
        ax.set(title=f'Target: {label}', ylabel='Realized Volatility')
        ax.xaxis.set_major_formatter(YEAR_FMT)
    _save(fig, 'targets_all_horizons')


# ── Models ────────────────────────────────────────────────────────────────────

def xgb_feature_importance(models, feature_cols):
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    fig.suptitle('XGBoost Feature Importance: Top 20 by Horizon', fontsize=15)
    for ax, h in zip(axes, C.HORIZONS):
        imp = pd.Series(models[h].feature_importances_, index=feature_cols)
        imp.nlargest(20).sort_values().plot(kind='barh', ax=ax, color='steelblue')
        ax.set(title=f'Horizon: {h}', xlabel='Importance Score')
    _save(fig, 'xgb_feature_importance')


def actual_vs_predicted(h, include_lstm):
    preds = load_all_preds(h, include_lstm)
    styles = {'naive': (':', 0.6, 0.6), 'garch': ('--', 0.7, 0.7),
              'xgboost': ('-', 0.7, 0.8), 'lstm': ('-', 0.7, 0.85)}
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(preds['naive']['timestamp'], preds['naive']['actual'], color='black', linewidth=0.8,
            label='Actual', alpha=0.9, zorder=5)
    for name, (ls, lw, alpha) in styles.items():
        if name in preds:
            ax.plot(preds[name]['timestamp'], preds[name]['predicted'], color=MODEL_COLORS[name],
                    linewidth=lw, linestyle=ls, label=name.capitalize(), alpha=alpha)
    ax.set(title=f'Actual vs Predicted Realized Volatility: Horizon {h}',
           ylabel='Realized Volatility', xlabel='Date')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    _save(fig, f'actual_vs_predicted_{h}')


def residuals(h, include_lstm):
    preds = load_all_preds(h, include_lstm)
    names = ['xgboost', 'garch'] + (['lstm'] if include_lstm else [])
    fig, axes = plt.subplots(2, len(names), figsize=(7 * len(names), 10))
    fig.suptitle(f'Residual Analysis: Horizon {h}', fontsize=14)
    for col, name in enumerate(names):
        p = preds[name]
        resid = p['actual'] - p['predicted']
        axes[0, col].plot(p['timestamp'], resid, color=MODEL_COLORS[name], linewidth=0.5, alpha=0.7)
        axes[0, col].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axes[0, col].set(title=f'{name.capitalize()} Residuals Over Time', ylabel='Residual')
        axes[1, col].hist(resid.dropna(), bins=100, color=MODEL_COLORS[name], alpha=0.8, edgecolor='none')
        axes[1, col].set(title=f'{name.capitalize()} Residual Distribution', xlabel='Residual')
    _save(fig, f'residuals_{h}')


def error_comparison(metrics_df):
    order = [m for m in MODEL_ORDER if m in metrics_df['model'].values]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('RMSE and MAE: All Models and Horizons', fontsize=15)
    for ax, metric in zip(axes, ['rmse', 'mae']):
        pivot = metrics_df.pivot_table(index='model', columns='horizon', values=metric).loc[order]
        pivot.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='none')
        ax.set(title=f'{metric.upper()} by Model and Horizon', ylabel=metric.upper(), xlabel='')
        ax.tick_params(axis='x', rotation=15)
        ax.legend(title='Horizon')
    _save(fig, 'error_comparison_by_model')


def mincer_zarnowitz_chart(mz_df):
    order = [m for m in MODEL_ORDER if m in mz_df['model'].values]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Mincer-Zarnowitz Results: All Models (perfect: alpha=0, beta=1, R²=1)', fontsize=13)
    for ax, metric, ideal, title in zip(axes, ['mz_alpha', 'mz_beta', 'mz_r2'], [0, 1, 1],
                                        ['Alpha (ideal = 0)', 'Beta (ideal = 1)', 'R-squared (ideal = 1)']):
        mz_df.pivot(index='model', columns='horizon', values=metric).loc[order] \
            .plot(kind='bar', ax=ax, colormap='Set2', edgecolor='none')
        ax.axhline(ideal, color='black', linewidth=1, linestyle='--', label=f'Ideal ({ideal})')
        ax.set(title=title, xlabel='')
        ax.tick_params(axis='x', rotation=15)
        ax.legend(title='Horizon', fontsize=8)
    _save(fig, 'mincer_zarnowitz')


def diebold_mariano_chart(dm_df):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Diebold-Mariano Statistics by Horizon\n'
                 '(negative = left model wins, positive = right model wins)', fontsize=13)
    for ax, h in zip(axes, C.HORIZONS):
        subset = dm_df[dm_df['horizon'] == h].set_index('comparison')
        colors = ['#1D9E75' if v < 0 else '#D85A30' for v in subset['dm_stat']]
        ax.barh(subset.index, subset['dm_stat'], color=colors, edgecolor='none', alpha=0.85)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set(title=f'Horizon: {h}', xlabel='DM Statistic')
        for i, (_, row) in enumerate(subset.iterrows()):
            ax.text(row['dm_stat'], i, f"  p={row['p_value']:.4f}", va='center', fontsize=9,
                    color='white' if abs(row['dm_stat']) > 5 else 'black')
    _save(fig, 'diebold_mariano')


# ── LSTM-only ─────────────────────────────────────────────────────────────────

def lstm_vs_xgb_vs_garch():
    labels = {'xgboost': 'XGBoost', 'garch': 'GARCH', 'lstm': 'LSTM'}
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle('LSTM vs XGBoost vs GARCH: Actual vs Predicted per Horizon', fontsize=14)
    for ax, h in zip(axes, C.HORIZONS):
        preds = load_all_preds(h, include_lstm=True)
        ax.plot(preds['xgboost']['timestamp'], preds['xgboost']['actual'], color='black',
                linewidth=0.8, label='Actual', alpha=0.9)
        for name, ls, alpha in [('xgboost', '-', 0.8), ('garch', '--', 0.7), ('lstm', '-', 0.85)]:
            ax.plot(preds[name]['timestamp'], preds[name]['predicted'], color=MODEL_COLORS[name],
                    linewidth=0.7, linestyle=ls, label=labels[name], alpha=alpha)
        ax.set(title=f'Horizon: {h}', ylabel='Realized Volatility')
        ax.legend(loc='upper right', fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=30)
    _save(fig, 'lstm_vs_xgb_vs_garch')


def lstm_vs_xgb_residuals():
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('LSTM vs XGBoost Residuals by Horizon', fontsize=14)
    for row, h in enumerate(C.HORIZONS):
        preds = load_all_preds(h, include_lstm=True)
        for name, label in [('xgboost', 'XGBoost'), ('lstm', 'LSTM')]:
            p = preds[name]
            resid = p['actual'] - p['predicted']
            axes[row, 0].plot(p['timestamp'], resid, color=MODEL_COLORS[name], linewidth=0.5, alpha=0.7, label=label)
            axes[row, 1].hist(resid.dropna(), bins=80, color=MODEL_COLORS[name], alpha=0.6,
                              edgecolor='none', label=label)
        axes[row, 0].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axes[row, 0].set(title=f'Residuals Over Time: {h}', ylabel='Residual')
        axes[row, 1].set(title=f'Residual Distribution: {h}', xlabel='Residual')
        axes[row, 0].legend()
        axes[row, 1].legend()
    _save(fig, 'lstm_vs_xgb_residuals')


def lstm_training_curves(histories):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('LSTM Training and Validation Loss by Horizon', fontsize=14)
    for ax, h in zip(axes, C.HORIZONS):
        ax.plot(histories[h]['loss'], color='steelblue', label='Train loss')
        ax.plot(histories[h]['val_loss'], color='crimson', label='Val loss', linestyle='--')
        ax.set(title=f'Horizon: {h}', xlabel='Epoch', ylabel='MSE Loss')
        ax.legend()
    _save(fig, 'lstm_training_curves')
