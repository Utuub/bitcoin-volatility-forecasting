"""GARCH(1,1) with Student-t errors, refit every GARCH_REFIT_DAYS during the test period."""
import numpy as np
import pandas as pd
from arch import arch_model

from .. import config as C


def fit_garch(returns):
    model = arch_model(returns * C.RETURN_SCALE, vol='GARCH', p=1, q=1,
                       dist='StudentsT', rescale=False)
    return model.fit(disp='off', show_warning=False)


def forecast_vol(result, horizon_bars):
    """Realized vol estimate for the horizon: sqrt of summed multi-step variances."""
    fc = result.forecast(horizon=horizon_bars, reindex=False)
    return np.sqrt(fc.variance.iloc[-1].sum()) / C.RETURN_SCALE


def predict(df_model, test_df):
    train_returns = df_model.loc[df_model.index < C.VAL_START, 'log_return']
    current = fit_garch(train_returns)
    print(f'Initial GARCH fit on {len(train_returns):,} obs | AIC {current.aic:.2f} | BIC {current.bic:.2f}')
    print(current.summary().tables[1])

    rows = {h: [] for h in C.HORIZONS}
    last_refit = pd.Timestamp(C.TEST_START)

    for i, ts in enumerate(test_df.index):
        if (ts - last_refit).days >= C.GARCH_REFIT_DAYS:
            try:
                current = fit_garch(df_model.loc[df_model.index < ts, 'log_return'])
                last_refit = ts
            except Exception as e:
                print(f'GARCH refit failed at {ts}: {e}; keeping prior fit')

        for h in C.HORIZONS:
            try:
                vol = forecast_vol(current, C.HORIZON_BARS[h])
            except Exception:
                vol = np.nan
            rows[h].append({
                'timestamp': ts,
                'actual'   : test_df.loc[ts, f'future_{h}_vol'],
                'predicted': vol,
                'model'    : 'garch',
                'horizon'  : h,
            })

        if i % 1000 == 0:
            print(f'GARCH progress: {i}/{len(test_df)}')

    preds = {}
    for h in C.HORIZONS:
        preds[h] = pd.DataFrame(rows[h]).dropna()
        preds[h].to_csv(f'{C.DIR_PREDS}/garch_{h}_predictions.csv', index=False)
        print(f'GARCH {h}: {len(preds[h])} predictions')
    return preds
