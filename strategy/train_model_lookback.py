# strategy/train_model_lookback.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

RAW_DIR = "/root/trade-control-system/data_backtest"
BACKTEST_DIR = "/root/trade-control-system/backtest_result"
MODEL_DIR = "/root/trade-control-system/strategy/ml/models"

os.makedirs(MODEL_DIR, exist_ok=True)

def prepare_dataset(raw_file, backtest_file):
    # load raw data
    raw_df = pd.read_excel(raw_file, index_col=0, parse_dates=True)
    # load hasil backtest
    bt_df = pd.read_excel(backtest_file, index_col=0, parse_dates=True)

    # pastikan kolom signal ada
    if 'signal' not in bt_df.columns:
        raise ValueError(f"'signal' column not found in {backtest_file}")

    # gabung raw + hasil backtest untuk fitur tambahan
    df = raw_df.join(bt_df[['signal','false_reversal','label']], how='left')
    df['signal'] = df['signal'].fillna('HOLD')
    df['false_reversal'] = df['false_reversal'].fillna(False)
    df = df[df['label'] != -1]  # drop NO HIT

    # Fitur teknikal untuk ML
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['signal_numeric'] = df['signal'].map({'LONG': 1, 'SHORT': -1, 'HOLD': 0})

    feature_cols = [
        'rsi', 'atr', 'boll_width', 'volume', 'close',
        'upper_band', 'lower_band', 'bb_percentile',
        'support', 'resistance', 'macd', 'macd_signal', 'macd_hist',
        'signal_numeric', 'false_reversal'
    ]
    X = df[feature_cols]
    y = df['label']

    return X, y

def train_model(pair, raw_file, backtest_file):
    X, y = prepare_dataset(raw_file, backtest_file)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    model_path = os.path.join(MODEL_DIR, f'breakout_rf_model_{pair}.pkl')
    joblib.dump(model, model_path)
    print(f"✅ Model trained and saved: {model_path}")

if __name__ == "__main__":
    # contoh run untuk satu pair
    pairs = ['AVAXUSDT']  # bisa ditambah
    for pair in pairs:
        raw_file = os.path.join(RAW_DIR, f"{pair}_1h_all_signals.xlsx")  # sesuaikan timeframe
        backtest_file = os.path.join(BACKTEST_DIR, f"hasil_backtest_avaxusdt_1h.xlsx")
        if os.path.exists(raw_file) and os.path.exists(backtest_file):
            train_model(pair, raw_file, backtest_file)
        else:
            print(f"⚠️ File missing for {pair}: raw or backtest not found")
