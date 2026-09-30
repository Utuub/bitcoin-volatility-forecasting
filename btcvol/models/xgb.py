"""XGBoost regressor per horizon, early-stopped on the validation period."""
import joblib
import pandas as pd
from xgboost import XGBRegressor

from .. import config as C


def train_and_predict(split, feature_cols):
    models, preds = {}, {}
    X_train, X_val, X_test = split['train']['X'], split['val']['X'], split['test']['X']

    for h in C.HORIZONS:
        model = XGBRegressor(**C.XGB_PARAMS)
        model.fit(X_train, split['train']['y'][h],
                  eval_set=[(X_val, split['val']['y'][h])], verbose=50)

        preds[h] = pd.DataFrame({
            'timestamp': split['test']['df'].index,
            'actual'   : split['test']['y'][h].values,
            'predicted': model.predict(X_test),
            'model'    : 'xgboost',
            'horizon'  : h,
        })
        preds[h].to_csv(f'{C.DIR_PREDS}/xgb_{h}_predictions.csv', index=False)
        joblib.dump(model, f'{C.DIR_MODELS}/xgb_{h}.pkl')

        importance = pd.Series(model.feature_importances_, index=feature_cols)
        importance.sort_values(ascending=False).to_csv(
            f'{C.DIR_METRICS}/xgb_feature_importance_{h}.csv', header=['importance'])

        models[h] = model
        print(f'XGBoost {h}: best iteration {model.best_iteration}')
    return models, preds
