"""
End-to-end run: clean data -> targets -> features -> split -> naive / GARCH / XGBoost / LSTM
-> evaluation -> figures. Everything lands in outputs/.

    python run_pipeline.py --data data/btc_1h_data_2018_to_2025.csv
    python run_pipeline.py --no-lstm     # skip the LSTM (no tensorflow needed)
"""
import argparse
import warnings

import numpy as np

from btcvol import config as C
from btcvol import data, evaluation, features, plots, splits, targets
from btcvol.models import garch, naive, xgb

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=C.RAW_DATA_PATH, help='path to the hourly BTC CSV')
    parser.add_argument('--no-lstm', action='store_true', help='skip training the LSTM')
    parser.add_argument('--no-plots', action='store_true', help='skip figure generation')
    args = parser.parse_args()
    run_lstm = not args.no_lstm

    np.random.seed(C.RANDOM_SEED)
    C.make_output_dirs()

    print('\n== 1. Ingest & clean')
    df = data.load_and_clean(args.data)
    data.save_data_dictionary()

    print('\n== 2. Returns & EDA')
    df = data.add_log_returns(df)
    df = data.add_buy_pressure_ratio(df)
    if not args.no_plots:
        plots.eda_price_returns_vol(df)
        plots.eda_return_distribution(df)
        plots.eda_volume_pressure(df)
    df[['close', 'log_return', 'volume', 'num_trades']].describe().round(6) \
        .to_csv(f'{C.DIR_NOTES}/descriptive_statistics.csv')

    print('\n== 3. Targets')
    df = targets.add_targets(df)
    if not args.no_plots:
        plots.targets_all_horizons(df)

    print('\n== 4. Features')
    df, feature_cols = features.build_features(df)
    df_model = features.freeze_dataset(df, feature_cols)

    print('\n== 5. Split')
    split = splits.chronological_split(df_model, feature_cols)
    wf = splits.get_walkforward_windows(df_model, C.TRAIN_START, C.VAL_START, C.TEST_START,
                                        C.WF_STEP_DAYS, C.WF_MAX_WINDOW_DAYS)
    print(f'Walk-forward windows: {len(wf)}')
    test_df = split['test']['df']

    print('\n== 6. Naive baseline')
    naive.predict(test_df)

    print('\n== 7. GARCH')
    garch.predict(df_model, test_df)

    print('\n== 8. XGBoost')
    xgb_models, _ = xgb.train_and_predict(split, feature_cols)

    print('\n== 9. Evaluation')
    metrics_df, mz_df, dm_df = evaluation.run_evaluation()

    histories = None
    if run_lstm:
        print('\n== 10. LSTM')
        from btcvol.models import lstm
        _, _, histories = lstm.train_and_predict(split)
        metrics_df, mz_df, dm_df = evaluation.add_lstm_results(metrics_df, mz_df, dm_df)

    if not args.no_plots:
        print('\n== 11. Figures')
        plots.xgb_feature_importance(xgb_models, feature_cols)
        for h in C.HORIZONS:
            plots.actual_vs_predicted(h, run_lstm)
            plots.residuals(h, run_lstm)
        plots.error_comparison(metrics_df)
        plots.mincer_zarnowitz_chart(mz_df)
        plots.diebold_mariano_chart(dm_df)
        if run_lstm:
            plots.lstm_vs_xgb_vs_garch()
            plots.lstm_vs_xgb_residuals()
            plots.lstm_training_curves(histories)

    print(f'\nDone. Outputs are in {C.OUTPUT_ROOT}/')


if __name__ == '__main__':
    main()
