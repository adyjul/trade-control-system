import os
import pandas as pd
import joblib

MODEL_PATH = '/root/trade-control-system/strategy/ml/models'

def get_model_path(pair: str, timeframe: str):
    return os.path.join(MODEL_PATH, f'model_{pair}_{timeframe}.pkl')

def load_ml_model(pair: str, timeframe: str):
    path = get_model_path(pair, timeframe)
    if not os.path.exists(path):
        print(f"⚠️ Model untuk {pair} {timeframe} tidak ditemukan.")
        return None
    return joblib.load(path)

def predict_ml_signal(model, row: pd.Series) -> bool:
    # Ambil fitur-fitur yang dipakai di model
    features = [
        'rsi', 'atr', 'boll_width', 'volume', 'close',
        'upper_band', 'lower_band', 'bb_percentile',
        'support', 'resistance', 'atr_multiple',
        'is_potential_breakout', 'entry_signal'
    ]
    X = row[features].values.reshape(1, -1)
    pred = model.predict(X)
    return pred[0] == 1  # 1 artinya valid (bukan fake breakout)