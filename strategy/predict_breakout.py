# strategy/predict_breakout.py
import pandas as pd
import joblib
import glob
import os
import sys
import numpy as np

# ------- CONFIG -------
# MODEL_PATH = "/root/trade-control-system/strategy/ml/models/breakout_rf_model_avaxusdt.pkl"
MODEL_BREAKOUT = "/root/trade-control-system/strategy/ml/models/breakout_rf_model_avaxusdt_1h.pkl"
MODEL_REVERSAL = "/root/trade-control-system/models/false_reversal_rf.pkl"
BACKTEST_GLOB = "/root/trade-control-system/backtest_result/hasil_backtest_avaxusdt_1h.xlsx"  # sesuaikan timeframe/pair jika perlu
# BACKTEST_GLOB = "/root/trade-control-system/data_predict/AVAXUSDT_1h_full.xlsx"
OUTPUT_PRED_PATH = "/root/trade-control-system/backtest_result/predicted_result_single_pair.xlsx"
ENTRY_OUTPUT = "/root/trade-control-system/backtest_result/entry_layak.xlsx"
PROB_THRESHOLD = 0.6  # hanya ambil prediksi TP dengan confidence >= threshold
# -----------------------

def safe_load_backtest(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Backtest file tidak ditemukan: {path}")
    df = pd.read_excel(path)
    return df

def ensure_features(df):
    # Fitur tambahan yang diperlukan untuk prediksi
    if 'macd' not in df.columns or 'macd_signal' not in df.columns:
        raise ValueError("Kolom 'macd' atau 'macd_signal' tidak ada di backtest. Pastikan backtest menghasilkan keduanya.")
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['signal_numeric'] = df['signal'].map({'LONG': 1, 'SHORT': -1})
    # Jika ada NaN di signal_numeric akibat label lain, isi 0 supaya model tidak error
    df['signal_numeric'] = df['signal_numeric'].fillna(0)
    df['atr_multiple'] = np.where(
        df['signal'] == 'LONG',
        (df['close'] - df['resistance']) / df['atr'],
        (df['support'] - df['close']) / df['atr']
    )
    return df

def main():
    # 1. Load model
    # if not os.path.isfile(MODEL_PATH):
    #     print(f"Model tidak ditemukan di {MODEL_PATH}")
    #     sys.exit(1)
    # model = joblib.load(MODEL_PATH)
    breakout_model = joblib.load(MODEL_BREAKOUT)
    reversal_model = joblib.load(MODEL_REVERSAL)
    print(f"[+] Model dimuat dari {MODEL_BREAKOUT}")
    print(f"[+] Model dimuat dari {MODEL_REVERSAL}")

    # 2. Temukan backtest file (ambil pertama yang match)
    matches = glob.glob(BACKTEST_GLOB)
    if not matches:
        print(f"Tidak ada file backtest yang cocok dengan pattern: {BACKTEST_GLOB}")
        sys.exit(1)
    backtest_path = matches[0]
    print(f"[+] Menggunakan backtest file: {backtest_path}")

    # 3. Load backtest
    df = safe_load_backtest(backtest_path)
    print(f"[+] Backtest rows sebelum filter: {len(df)}")

    # 4. Drop sinyal NO HIT kalau memang kamu ingin hanya sinyal jelas
    # (kalau ingin prediksi juga terhadap NO HIT bisa di-comment)
    if 'exit_status' in df.columns:
        df = df[df['exit_status'] != 'NO HIT'].copy()
        print(f"[+] Setelah drop NO HIT: {len(df)}")

    # 5. Tambah fitur turunan
    df = ensure_features(df)

    # 6. Siapkan fitur yang dipakai model — harus sinkron dengan training
    # feature_columns = [
    #     'rsi', 'atr', 'boll_width', 'volume', 'close',
    #     'upper_band', 'lower_band', 'bb_percentile',
    #     'support', 'resistance',
    #     'false_reversal',
    #     'macd', 'macd_signal', 'macd_hist', 'signal_numeric'
    # ]

    # feature_columns = [
    #     'rsi', 'atr', 'boll_width', 'volume', 'close',
    #     'upper_band', 'lower_band', 'bb_percentile',
    #     'support', 'resistance', 'macd', 'macd_signal', 'macd_hist',
    #     'signal_numeric', 'false_reversal'
    # ]

    breakout_features = [
        'rsi', 'atr', 'boll_width', 'volume', 'close',
        'upper_band', 'lower_band', 'bb_percentile',
        'support', 'resistance', 'atr_multiple',
        'is_potential_breakout', 'entry_signal','macd', 'macd_signal', 'macd_hist', 'signal_numeric'
    ]

    reversal_features = [
        "rsi", "atr", "macd", "macd_signal", "macd_hist",
        "upper_band", "lower_band", "volume", "support", "resistance"
    ]
    

    breakout_missing = [f for f in breakout_features if f not in df.columns]
    if breakout_missing:
        raise ValueError(f"Kolom fitur hilang untuk breakout model: {breakout_missing}")
    X_breakout = df[breakout_features]

    df['predicted_breakout'] = breakout_model.predict(X_breakout)

    if hasattr(breakout_model, "predict_proba"):
        probs = breakout_model.predict_proba(X_breakout)
        try:
            idx_tp = list(breakout_model.classes_).index(1)
            df['prob_breakout'] = probs[:, idx_tp]
        except ValueError:
            df['prob_breakout'] = probs.max(axis=1)
    else:
        df['prob_breakout'] = None

    # --- 7. Kandidat hanya yg lolos breakout ---
    df_candidate = df[(df['predicted_breakout'] == 1) & (df['prob_breakout'] >= PROB_THRESHOLD)].copy()

    # --- 8. Prediksi tahap 2: Reversal model ---
    if not df_candidate.empty:
        reversal_missing = [f for f in reversal_features if f not in df_candidate.columns]
        if reversal_missing:
            raise ValueError(f"Kolom fitur hilang untuk reversal model: {reversal_missing}")
        X_rev = df_candidate[reversal_features]
        df_candidate['predicted_reversal'] = reversal_model.predict(X_rev)
    else:
        df_candidate['predicted_reversal'] = []

    # --- 9. Gabung hasil ---
    df = df.merge(df_candidate[['predicted_reversal']], left_index=True, right_index=True, how='left')
    df['predicted_reversal'] = df['predicted_reversal'].fillna(0).astype(int)

    df['final_label'] = df.apply(
        lambda row: 1 if (row['predicted_breakout']==1 and row['predicted_reversal']==1) else 0,
        axis=1
    )

    df['predicted_result'] = df['final_label'].map({1: 'ENTRY_VALID', 0: 'DITOLAK'})

    # 8. Filter entry layak berdasarkan threshold
    # if df['prob_tp'].notna().all():
    #     df_layak = df[(df['predicted_label'] == 1) & (df['prob_tp'] >= PROB_THRESHOLD)].copy()
    # else:
    #     df_layak = df[df['predicted_label'] == 1].copy()
    df_layak = df[(df['final_label'] == 1) & (df['prob_breakout'] >= PROB_THRESHOLD)].copy()

    # 9. Simpan hasil
    df.to_excel(OUTPUT_PRED_PATH, index=False)
    print(f"[+] Semua prediksi disimpan ke {OUTPUT_PRED_PATH}")
    if not df_layak.empty:
        df_layak.to_excel(ENTRY_OUTPUT, index=False)
        print(f"[+] Entry layak (prediksi TP dengan confidence tinggi) disimpan ke {ENTRY_OUTPUT}")
    else:
        print("[!] Tidak ada entry layak di threshold saat ini.")

if __name__ == "__main__":
    main()
