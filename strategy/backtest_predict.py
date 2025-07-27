import os
import pandas as pd
from utils.backtest import evaluate_signals_in_file  # Kita buat fungsi ini modular
from datetime import datetime

DATA_DIR = './data_predict'
OUTPUT_PATH = './data_predict/filter_result.xlsx'

def run_backtest_filter():
    results = []

    for file in os.listdir(DATA_DIR):
        if file.endswith('_full.xlsx'):
            path = os.path.join(DATA_DIR, file)
            pair_tf = file.replace('_full.xlsx', '')
            try:
                result = evaluate_signals_in_file(path)
                result['pair_tf'] = pair_tf
                results.append(result)
                print(f"✅ Evaluated: {pair_tf} → {result['tp_rate']:.2f}%")
            except Exception as e:
                print(f"❌ Error evaluating {pair_tf}: {e}")

    if results:
        df_result = pd.DataFrame(results)
        df_result = df_result.sort_values(by='tp_rate', ascending=False)
        df_result.to_excel(OUTPUT_PATH, index=False)
        print(f"\n📊 Saved result: {OUTPUT_PATH}")
    else:
        print("⚠️ No result generated")

if __name__ == '__main__':
    run_backtest_filter()
