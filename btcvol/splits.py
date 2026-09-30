"""Chronological train / validation / test splits."""
import pandas as pd

from . import config as C


def chronological_split(df_model, feature_cols):
    train = df_model[df_model.index < C.VAL_START]
    val   = df_model[(df_model.index >= C.VAL_START) & (df_model.index < C.TEST_START)]
    test  = df_model[df_model.index >= C.TEST_START]

    for name, part in [('Train', train), ('Val', val), ('Test', test)]:
        print(f'{name:5}: {part.index.min()} -> {part.index.max()} ({len(part):,} rows)')

    split = {}
    for name, part in [('train', train), ('val', val), ('test', test)]:
        split[name] = {
            'df': part.copy(),
            'X' : part[feature_cols],
            'y' : {h: part[f'future_{h}_vol'] for h in C.HORIZONS},
        }
    return split


def get_walkforward_windows(df_full, train_start, val_start, test_start,
                            step_days=30, max_window_days=365):
    """
    Walk-forward (train_idx, val_idx) windows. The training window expands from
    train_start until it exceeds max_window_days, then rolls at a fixed length.
    """
    windows = []
    cur_val_start = pd.Timestamp(val_start)
    cur_val_end   = cur_val_start + pd.Timedelta(days=step_days)
    abs_train_start = pd.Timestamp(train_start)
    test_boundary   = pd.Timestamp(test_start)

    while cur_val_end <= test_boundary:
        train_window_days = (cur_val_start - abs_train_start).days
        expanding = train_window_days <= max_window_days
        cur_train_start = abs_train_start if expanding else cur_val_start - pd.Timedelta(days=max_window_days)

        train_mask = (df_full.index >= cur_train_start) & (df_full.index < cur_val_start)
        val_mask   = (df_full.index >= cur_val_start) & (df_full.index < cur_val_end)

        if train_mask.sum() > 0 and val_mask.sum() > 0:
            windows.append({
                'train_start': cur_train_start,
                'train_end'  : cur_val_start,
                'val_start'  : cur_val_start,
                'val_end'    : cur_val_end,
                'mode'       : 'expanding' if expanding else 'fixed',
                'train_idx'  : df_full.index[train_mask],
                'val_idx'    : df_full.index[val_mask],
            })

        cur_val_start = cur_val_end
        cur_val_end   = cur_val_start + pd.Timedelta(days=step_days)

    return windows
