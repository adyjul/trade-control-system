import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# === STEP 1: Baca file Excel hasil backtest ===
df = pd.read_excel('/root/trade-control-system/backtest_result/hasil_backtest_enausdt_1h.xlsx')

# === STEP 2: Buat kolom label (0/1) jika belum ada ===
if 'label' not in df.columns:
    df['label'] = df['exit_status'].map({'TP HIT': 1, 'SL HIT': 0, 'NO HIT': -1})

# Sinyal entry
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
df['entry_signal'] = df['entry_signal'].shift(1)
df['vol_3_candle'] = df['volume'].rolling(window=3).sum()
df['rsi_diff'] = df['rsi'] - df['rsi'].shift(1)
df['prev_volume'] = df['volume'].shift(1)
df['prev_close'] = df['close'].shift(1)
df['prev_return'] = df['close'].pct_change().shift(1)

# Drop NaN
df.dropna(inplace=True)

# === Drop kolom yang tidak dipakai (jika ada) ===
df = df.drop(columns=["timestamp", "pair", "timeframe"], errors="ignore")
df.drop(columns=['pair', 'timeframe', 'tp', 'sl', 'result'], inplace=True, errors='ignore')

# === Simpan untuk referensi ===
df.to_csv('/root/trade-control-system/backtest_result/ml_dataset_cleaned.csv', index=False)

# === Siapkan fitur dan label ===
y = df['label']
X = df.drop(columns=['label'])

# === Balancing ===
if X.empty or y.empty:
    raise ValueError("X atau y kosong setelah preprocessing!")

smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

# === Scaling ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_bal)

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)

# === Models ===
xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                    use_label_encoder=False, eval_metric='logloss')
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
lr = LogisticRegression(max_iter=500)

voting_model = VotingClassifier(
    estimators=[('xgb', xgb), ('rf', rf), ('lr', lr)],
    voting='soft'
)

# === Training ===
voting_model.fit(X_train, y_train)

# === Evaluation ===
y_pred = voting_model.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# === Save model ===
model_path = '/root/trade-control-system/backtest_result/breakout_model.pkl'
joblib.dump(voting_model, model_path)
print(f"Model disimpan sebagai '{model_path}'")
