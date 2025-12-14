import pandas as pd

def prepare_ml_dataset(input_file='/home/trade-control-system/backtest_result/summary_backtes.xlsx', output_file='/home/trade-control-system/backtest_result/ml_dataset.csv'):
    # Baca data hasil backtest
    df = pd.read_excel(input_file)

    # Hapus baris kosong atau tidak ada hasil
    df = df.dropna(subset=['Result'])

    # Labeling: 1 jika TP HIT, 0 jika SL HIT atau NO HIT
    df['label'] = df['Result'].apply(lambda x: 1 if x == 'TP HIT' else 0)

    # Fitur yang akan dipakai untuk training
    feature_cols = [
        'rsi', 'atr', 'boll_width', 'volume', 'close',
        'upper_band', 'lower_band', 'bb_percentile',
        'support', 'resistance', 'atr_multiple',
        'is_potential_breakout', 'entry_signal',
        'macd', 'macd_signal', 'macd_hist', 'signal_numeric'
    ]

    # Pastikan semua kolom fitur ada di data
    feature_cols = [col for col in feature_cols if col in df.columns]

    # Ambil hanya fitur + label
    df_ml = df[feature_cols + ['label']].copy()

    # Simpan ke CSV
    df_ml.to_csv(output_file, index=False)
    print(f"✅ Dataset berhasil disiapkan: {output_file} ({len(df_ml)} baris)")

if __name__ == '__main__':
    prepare_ml_dataset()
