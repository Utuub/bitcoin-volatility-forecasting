"""Persistence baseline: next period's volatility equals the most recent past realized vol."""
import pandas as pd

from .. import config as C


def predict(test_df):
    preds = {}
    for h in C.HORIZONS:
        preds[h] = pd.DataFrame({
            'timestamp': test_df.index,
            'actual'   : test_df[f'future_{h}_vol'].values,
            'predicted': test_df[f'past_rv_{h}'].values,
            'model'    : 'naive_persistence',
            'horizon'  : h,
        }).dropna()
        preds[h].to_csv(f'{C.DIR_PREDS}/naive_{h}_predictions.csv', index=False)
        print(f'Naive {h}: {len(preds[h])} predictions')
    return preds
