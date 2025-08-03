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
    row = row.copy()
    if row['signal'] == 'LONG':
        row['atr_multiple'] = (row['close'] - row['resistance']) / row['atr']
    else:
        row['atr_multiple'] = (row['support'] - row['close']) / row['atr']
    # Ambil fitur-fitur yang dipakai di model
    features = [
        'rsi', 'atr', 'boll_width', 'volume', 'close',
        'upper_band', 'lower_band', 'bb_percentile',
        'support', 'resistance', 'atr_multiple',
        'is_potential_breakout', 'entry_signal',
        'macd', 'macd_signal', 'macd_hist', 'signal_numeric'
        
    ]

    # X = row[features].values.reshape(1, -1)
    X = pd.DataFrame([row[features]], columns=features)
    pred = model.predict(X)
    return pred[0] == 1  # 1 artinya valid (bukan fake breakout)

def predict_signal_success(model, latest_features: pd.DataFrame):
    if model is None or latest_features is None:
        return True  # Tetap lanjut jika model tidak ada

    try:
        # Asumsi latest_features sudah dalam bentuk DataFrame baris tunggal
        pred = model.predict(latest_features)
        return pred[0] == 1  # 1 berarti sinyal layak (potensi TP HIT)
    except Exception as e:
        print(f"[ML FILTER ERROR] Gagal prediksi model: {e}")
        return True