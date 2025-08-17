import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load dataset hasil backtest
df = pd.read_excel("/root/trade-control-system/backtest_result/hasil_backtest_avaxusdt_1h.xlsx")

# 2. Filter hanya baris dengan false_reversal = True
df = df[df["false_reversal"] == True]

# 3. Buat label: 1 untuk TP HIT, 0 untuk SL HIT
df = df[df["exit_status"].isin(["TP HIT", "SL HIT"])]
df["label"] = (df["exit_status"] == "TP HIT").astype(int)

# 4. Pilih fitur teknikal (contoh: RSI, ATR, MACD, volume, dsb)
# Pastikan kolom2 ini memang ada di datasetmu
features = [
    "RSI", "ATR", "macd", "macd_signal", "macd_hist",
    "upper_band", "lower_band", "volume", "support", "resistance"
]
X = df[features]
y = df["label"]

# 5. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train RandomForest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluasi
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 8. Feature importance
importances = pd.Series(model.feature_importances_, index=features)
print(importances.sort_values(ascending=False))
