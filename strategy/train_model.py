import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np

# === STEP 1: Baca file Excel hasil backtest ===
df = pd.read_excel('/root/trade-control-system/backtest_result/hasil_backtest_avaxusdt_1h.xlsx')

# === STEP 2: Buat kolom label (0/1) jika belum ada ===
if 'label' not in df.columns:
    df['label'] = df['exit_status'].map({'TP HIT': 1, 'SL HIT': 0, 'NO HIT': -1})

df['entry_signal'] = ((df['is_potential_breakout'] == 1) & (df['signal'].notna())).astype(int)

# === STEP 3: Drop data yang labelnya -1 (NO HIT, tidak jelas hasilnya) ===
df = df[df['label'] != -1]


# === Tambahan Fitur ===
df['macd'] = df['macd'].astype(float)
df['macd_signal'] = df['macd_signal'].astype(float)
df['macd_hist'] = df['macd'] - df['macd_signal']
df['signal_numeric'] = df['signal'].map({'LONG': 1, 'SHORT': -1})
df['atr_multiple'] = np.where(
    df['signal'] == 'LONG',
    (df['close'] - df['resistance']) / df['atr'],
    (df['support'] - df['close']) / df['atr']
)

# === STEP 4: (Opsional) Simpan ke CSV untuk cek manual / pelatihan lanjutan ===
df.to_csv('/root/trade-control-system/backtest_result/ml_dataset.csv', index=False)

# === STEP 5: Tentukan fitur yang akan digunakan ===
feature_columns = [
    'rsi', 'atr', 'boll_width', 'volume', 'close',
    'upper_band', 'lower_band', 'bb_percentile',
    'support', 'resistance', 'atr_multiple',
    'is_potential_breakout', 'entry_signal',
    'macd', 'macd_signal', 'macd_hist', 'signal_numeric'
]

X = df[feature_columns]
y = df['label']

# === STEP 6: Bagi dataset menjadi train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === STEP 7: Latih model Random Forest ===
# model = RandomForestClassifier(n_estimators=100, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# === STEP 8: Evaluasi model ===
y_pred = model.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("=== Cross Validation Score (5-fold) ===")
print("Mean Accuracy:", cv_scores.mean())
print("All scores:", cv_scores)

# === STEP 9: Simpan model (opsional, kalau mau dipakai buat prediksi nanti) ===
import joblib
joblib.dump(model, '/root/trade-control-system/models/breakout_rf_model.pkl')
print("Model disimpan sebagai '/root/trade-control-system/models/breakout_rf_model.pkl'")

