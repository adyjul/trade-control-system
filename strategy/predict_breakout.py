import pandas as pd
import joblib
import glob
import os
import sys
import numpy as np

# ------- CONFIG -------
MODEL_PATH = "/root/trade-control-system/backtest_result/breakout_rf_model.pkl"
BACKTEST_GLOB = "/root/trade-control-system/backtest_result/hasil_backtest_enausdt_1h.xlsx"  # sesuaikan timeframe/pair jika perlu
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
    if not os.path.isfile(MODEL_PATH):
        print(f"Model tidak ditemukan di {MODEL_PATH}")
        sys.exit(1)
    model = joblib.load(MODEL_PATH)
    print(f"[+] Model dimuat dari {MODEL_PATH}")

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
    feature_columns = [
        'rsi', 'atr', 'boll_width', 'volume', 'close',
        'upper_band', 'lower_band', 'bb_percentile',
        'support', 'resistance', 'atr_multiple',
        'is_potential_breakout',
        'macd', 'macd_signal', 'macd_hist', 'signal_numeric',
        'entry_signal', 'vol_3_candle', 'rsi_diff',
        'prev_close', 'prev_volume', 'prev_return'
    ]
    missing = [f for f in feature_columns if f not in df.columns]
    if missing:
        raise ValueError(f"Kolom fitur hilang dari backtest: {missing}")

    X = df[feature_columns]

    # 7. Prediksi label + probabilitas
    df['predicted_label'] = model.predict(X)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        # RandomForestClassifier classes_ should align; ambil probabilitas untuk kelas 1 (TP)
        try:
            idx_tp = list(model.classes_).index(1)
            df['prob_tp'] = probs[:, idx_tp]
        except ValueError:
            # kalau kelas 1 tidak ada (skew extreme), fallback: ambil max-prob
            df['prob_tp'] = probs.max(axis=1)
    else:
        df['prob_tp'] = None

    df['predicted_result'] = df['predicted_label'].map({1: 'POTENSIAL_TP', 0: 'BERPOTENSI_GAGAL'})

    # 8. Filter entry layak berdasarkan threshold
    if df['prob_tp'].notna().all():
        df_layak = df[(df['predicted_label'] == 1) & (df['prob_tp'] >= PROB_THRESHOLD)].copy()
    else:
        df_layak = df[df['predicted_label'] == 1].copy()

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
