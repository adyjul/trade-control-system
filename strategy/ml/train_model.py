import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import re  # Untuk validasi pola nama file

def train_all_models():
    folder_path = '/home/trade-control-system/backtest_result'
    model_output_dir = '/home/trade-control-system/strategy/ml/models'
    os.makedirs(model_output_dir, exist_ok=True)

    pattern = r'^hasil_backtest_[a-z0-9]+_[0-9a-z]+\.xlsx$'

    for filename in os.listdir(folder_path):
        if re.match(pattern, filename):
            if filename.endswith('.xlsx'):
                filepath = os.path.join(folder_path, filename)
                print(filepath)
                df = pd.read_excel(filepath)

                # Proses label
                if 'label' not in df.columns:
                    df['label'] = df['exit_status'].map({'TP HIT': 1, 'SL HIT': 0, 'NO HIT': -1})

                df['entry_signal'] = ((df['is_potential_breakout'] == 1) & (df['signal'].notna())).astype(int)
                df = df[df['label'] != -1]

                # Fitur teknikal
                df['macd'] = df['macd'].astype(float)
                df['macd_signal'] = df['macd_signal'].astype(float)
                df['macd_hist'] = df['macd'] - df['macd_signal']
                df['signal_numeric'] = df['signal'].map({'LONG': 1, 'SHORT': -1})
                df['atr_multiple'] = np.where(
                    df['signal'] == 'LONG',
                    (df['close'] - df['resistance']) / df['atr'],
                    (df['support'] - df['close']) / df['atr']
                )

                feature_columns = [
                    'rsi', 'atr', 'boll_width', 'volume', 'close',
                    'upper_band', 'lower_band', 'bb_percentile',
                    'support', 'resistance', 'atr_multiple',
                    'is_potential_breakout', 'entry_signal','macd', 'macd_signal', 'macd_hist', 'signal_numeric'
                ]

                X = df[feature_columns]
                y = df['label']

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                model.fit(X_train, y_train)

                # Simpan model dengan nama sesuai coin dan timeframe
                coin_tf = filename.replace("hasil_backtest_", "").replace(".xlsx", "")
                model_path = os.path.join(model_output_dir, f'breakout_rf_model_{coin_tf}.pkl')
                joblib.dump(model, model_path)
                print(f"✅ Model disimpan: {model_path}")
