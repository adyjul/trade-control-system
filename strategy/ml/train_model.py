import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

def train_model_for_pair(pair: str, timeframe: str):
    import os
    filename = f"{pair}_{timeframe}_full.xlsx"
    path = f"/root/trade-control-system/data_predict/{filename}"
    df = pd.read_excel(path)

    df['label'] = df['exit_status'].map({'TP HIT': 1, 'SL HIT': 0, 'NO HIT': -1})
    df['entry_signal'] = ((df['is_potential_breakout'] == 1) & (df['signal'].notna())).astype(int)
    df = df[df['label'] != -1]

    df['macd'] = df['macd'].astype(float)
    df['macd_signal'] = df['macd_signal'].astype(float)
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['signal_numeric'] = df['signal'].map({'LONG': 1, 'SHORT': -1})
    df['atr_multiple'] = np.where(
        df['signal'] == 'LONG',
        (df['close'] - df['resistance']) / df['atr'],
        (df['support'] - df['close']) / df['atr']
    )

    features = [
        'rsi', 'atr', 'boll_width', 'volume', 'close',
        'upper_band', 'lower_band', 'bb_percentile',
        'support', 'resistance', 'atr_multiple',
        'is_potential_breakout', 'entry_signal',
        'macd', 'macd_signal', 'macd_hist', 'signal_numeric'
    ]

    X = df[features]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    save_path = f"/root/trade-control-system/strategy/ml/models/model_{pair}_{timeframe}.pkl"
    joblib.dump(model, save_path)
    print(f"✅ Model untuk {pair} - {timeframe} disimpan di {save_path}")
