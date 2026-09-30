"""Two-layer LSTM over 72-hour feature sequences (requires TensorFlow)."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .. import config as C


def make_sequences(X, y, seq_len):
    """Each sample is the seq_len rows before position i; its target is y[i]."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def build_model(seq_len, n_features):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.LSTM(32, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(16, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=C.LSTM_LR), loss='mse')
    return model


def train_and_predict(split):
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(C.RANDOM_SEED)

    # Scale on training data only
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(split['train']['X'])
    X_vl = scaler.transform(split['val']['X'])
    X_te = scaler.transform(split['test']['X'])

    models, preds, histories = {}, {}, {}
    for h in C.HORIZONS:
        X_tr_seq, y_tr_seq = make_sequences(X_tr, split['train']['y'][h].values, C.LSTM_SEQ_LEN)
        X_vl_seq, y_vl_seq = make_sequences(X_vl, split['val']['y'][h].values, C.LSTM_SEQ_LEN)
        X_te_seq, y_te_seq = make_sequences(X_te, split['test']['y'][h].values, C.LSTM_SEQ_LEN)

        model = build_model(C.LSTM_SEQ_LEN, X_tr_seq.shape[2])
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                   restore_best_weights=True, verbose=1)
        history = model.fit(X_tr_seq, y_tr_seq, validation_data=(X_vl_seq, y_vl_seq),
                            epochs=C.LSTM_EPOCHS, batch_size=C.LSTM_BATCH,
                            callbacks=[early_stop], verbose=1)

        p = model.predict(X_te_seq).flatten()
        # Sequences consume the first LSTM_SEQ_LEN test rows
        ts = split['test']['df'].index[C.LSTM_SEQ_LEN:]
        preds[h] = pd.DataFrame({
            'timestamp': ts[:len(p)],
            'actual'   : y_te_seq[:len(p)],
            'predicted': p,
            'model'    : 'lstm',
            'horizon'  : h,
        })
        preds[h].to_csv(f'{C.DIR_PREDS}/lstm_{h}_predictions.csv', index=False)
        model.save(f'{C.DIR_MODELS}/lstm_{h}.keras')
        models[h], histories[h] = model, history.history
    return models, preds, histories
