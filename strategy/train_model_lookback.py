import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- Folder paths ---
RAW_DATA_DIR = '/root/trade-control-system/backtest_result/'
BACKTEST_DIR = '/root/trade-control-system/backtest_result/'
MODEL_DIR = '/root/trade-control-system/models'
os.makedirs(MODEL_DIR, exist_ok=True)

LOOKBACK = 2  # jumlah candle sebelumnya

def prepare_dataset(raw_path, backtest_path):
    raw_df = pd.read_excel(raw_path, index_col='timestamp', parse_dates=True)
    bt_df = pd.read_excel(backtest_path, index_col='timestamp', parse_dates=True)

    # Merge signal & false_reversal ke raw data
    raw_df = raw_df.merge(bt_df[['signal', 'false_reversal', 'label']], 
                          left_index=True, right_index=True, how='left')

    # Fill NA untuk signal bar yang bukan TP/SL
    raw_df['signal'] = raw_df['signal'].fillna('HOLD')
    raw_df['false_reversal'] = raw_df['false_reversal'].fillna(False)
    raw_df['label'] = raw_df['label'].fillna(-1)

    # Buat fitur lookback
    for i in range(1, LOOKBACK + 1):
        raw_df[f'prev_close_{i}'] = raw_df['close'].shift(i)
        raw_df[f'prev_signal_{i}'] = raw_df['signal'].shift(i).map({'LONG':1,'SHORT':-1,'HOLD':0})
    
    # Pilih bar valid untuk training (label != -1)
    df_train = raw_df[raw_df['label'] != -1].copy()

    feature_columns = [
        'rsi', 'atr', 'boll_width', 'volume', 'close',
        'upper_band', 'lower_band', 'bb_percentile',
        'support', 'resistance',
        'macd', 'macd_signal', 'macd_hist',
        'false_reversal'
    ]
    # Tambah lookback fitur
    for i in range(1, LOOKBACK + 1):
        feature_columns += [f'prev_close_{i}', f'prev_signal_{i}']

    X = df_train[feature_columns]
    y = df_train['label']
    return X, y

def train_model(pair, raw_file, backtest_file):
    X, y = prepare_dataset(raw_file, backtest_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    model_path = os.path.join(MODEL_DIR, f'breakout_rf_model_{pair}.pkl')
    joblib.dump(model, model_path)
    print(f"✅ Model saved: {model_path}")

# --- Train semua pair ---
for file in os.listdir(BACKTEST_DIR):
    if file.endswith('.xlsx') and file.startswith('hasil_backtest_'):
        # pair = file.replace('hasil_backtest_', '').replace('.xlsx', '')
        raw_file = os.path.join(RAW_DATA_DIR, f'AVAXUSDT_1h_all_signals.xlsx')
        backtest_file = os.path.join(BACKTEST_DIR, file)
        if os.path.exists(raw_file):
            train_model('AVAXUSDT', raw_file, backtest_file)
        else:
            print(f"⚠️ Raw data not found for , skipping.")
