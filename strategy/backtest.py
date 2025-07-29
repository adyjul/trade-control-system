# strategy/backtest.py
import os
from datetime import datetime, timezone
import pandas as pd
import ta
from binance.client import Client
import glob

from utils.binance_client import get_client
from utils.timeframes import BINANCE_INTERVAL_MAP

# ---- Optional: set default folder kalau belum ada di config/env ----
DEFAULT_DATA_DIR = "./data_backtest"
DEFAULT_RESULT_DIR = "./backtest_result"

def evaluate_tp_sl(df: pd.DataFrame, look_ahead=7) -> pd.DataFrame:
    """
    Evaluasi TP / SL dengan melihat N (look_ahead) candle setelah sinyal keluar.
    """
    df['exit_status'] = 'NO HIT'
    idxs = list(df.index)
    for i, ts in enumerate(idxs):
        row = df.loc[ts]
        signal = row['signal']
        tp = row['tp_price']
        sl = row['sl_price']
        idx_start = df.index.get_loc(row.name)

        # ambil 6 candle ke depan
        # future_slice = df.loc[idxs[i+1:i+1+look_ahead]]
        # future_slice = df.iloc[idx_start+1:idx_start+7]
        future_slice = df.iloc[idx_start+1:idx_start+1+look_ahead]
        for _, f in future_slice.iterrows():
            if signal == 'LONG':
                if f['high'] >= tp:
                    df.at[ts, 'exit_status'] = 'TP HIT'
                    break
                elif f['low'] <= sl:
                    df.at[ts, 'exit_status'] = 'SL HIT'
                    break
            elif signal == 'SHORT':
                if f['low'] <= tp:
                    df.at[ts, 'exit_status'] = 'TP HIT'
                    break
                elif f['high'] >= sl:
                    df.at[ts, 'exit_status'] = 'SL HIT'
                    break
    return df


def detect_signal(row):

    # v1
    if pd.isna(row['macd']) or pd.isna(row['macd_signal']) or pd.isna(row['rsi']) or pd.isna(row['volume_sma20']):
        return 'HOLD'

    if row['atr'] < 0.005 * row['close']:
        return 'HOLD'

    # ========== LONG Condition ==========
    # if row['macd'] > row['macd_signal'] and row['rsi'] > 50:
    #     if row['rsi'] > 75:  # Overbought → hindari entry LONG
    #         return 'HOLD'
    #     if row['volume'] < row['volume_sma20']:  # Volume rendah → hindari breakout
    #         return 'HOLD'
    #     return 'LONG'

    # ========== SHORT Condition ==========
    # if row['macd'] < row['macd_signal'] and row['rsi'] < 50:
    #     if row['rsi'] < 35:  # Oversold → hindari entry SHORT
    #         return 'HOLD'
    #     if row['volume'] < row['volume_sma20']:  # Volume rendah → hindari breakdown
    #         return 'HOLD'
    #     return 'SHORT'

     # --- Validasi candlestick: hindari doji/spinning top ---
    # candle_body = abs(row['close'] - row['open'])
    # candle_range = row['high'] - row['low']
    # if candle_range == 0 or candle_body < 0.3 * candle_range:
    #     return 'HOLD'

    if row['macd'] > row['macd_signal'] and row['rsi'] > 50:
        return 'LONG' if row['volume'] > row['volume_sma20'] else 'LONG_WEAK'

    if row['macd'] < row['macd_signal'] and row['rsi'] < 50:
        if row['rsi'] < 35:
            return 'HOLD'
        return 'SHORT'

    return 'HOLD'

    # v2 mungkin untuk weekend
    # if pd.isna(row['macd']) or pd.isna(row['macd_signal']) or pd.isna(row['rsi']) or pd.isna(row['volume_sma20']):
    #     return 'HOLD'

    # Filter ATR kecil → tidak volatile
    # if row['atr'] < 0.005 * row['close']:
    #     return 'HOLD'

    # ========== LONG Condition ==========
    # if row['macd'] > row['macd_signal'] and row['rsi'] > 50:
    #     if row['rsi'] > 75:  # Overbought → hindari entry LONG
    #         return 'HOLD'
    #     if row['volume'] < row['volume_sma20']:  # Volume rendah → hindari breakout
    #         return 'HOLD'
    #     return 'LONG'

    # ========== SHORT Condition ==========
    # if row['macd'] < row['macd_signal'] and row['rsi'] < 50:
    #     if row['rsi'] < 35:  # Oversold → hindari entry SHORT
    #         return 'HOLD'
    #     if row['volume'] < row['volume_sma20']:  # Volume rendah → hindari breakdown
    #         return 'HOLD'
    #     return 'SHORT'

    # return 'HOLD'

def clear_folder(folder_path):
    for file_path in glob.glob(os.path.join(folder_path, '*')):
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")


def run_full_backtest(
    pairs,
    timeframe: str,
    limit: int,
    look_ahead: int = 6,
    tp_atr_mult: float = 1.0,
   
    sl_atr_mult: float = 1.0,
    data_dir: str = DEFAULT_DATA_DIR,
    result_dir: str = DEFAULT_RESULT_DIR,
    save_summary: bool = True
):
    
    """
    1) Scrape OHLCV dari Binance Futures
    2) Hitung indikator & sinyal
    3) Set TP/SL berbasis ATR
    4) Evaluasi TP/SL (TP HIT / SL HIT / NO HIT)
    5) Simpan hasil per-pair (xlsx) + summary keseluruhan (xlsx)
    """
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    client: Client = get_client()
    interval = BINANCE_INTERVAL_MAP[timeframe]

    # clear folder
    clear_folder(data_dir)
    clear_folder(result_dir)

    for pair in pairs:
        # --- scrape data ---
        klines = client.futures_klines(symbol=pair, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
        df.set_index('timestamp', inplace=True)

        # simpan raw
        raw_path = os.path.join(data_dir, f"{pair}_{timeframe}.csv")
        df.to_csv(raw_path)

        # --- indikator ---
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['volume_sma20'] = df['volume'].rolling(window=20).mean()

        # --- sinyal ---
        df['signal'] = df.apply(detect_signal, axis=1)
        df = df[df['signal'].isin(['LONG', 'SHORT'])].copy()
        if df.empty:
            # tidak ada sinyal sama sekali
            out_path = os.path.join(result_dir, f"hasil_backtest_{pair.lower()}_{timeframe}.xlsx")
            pd.DataFrame(columns=[
                'signal','entry_price','tp_price','sl_price','exit_status'
            ]).to_excel(out_path, index=False)
            return {
                "pair": pair,
                "timeframe": timeframe,
                "total": 0,
                "tp": 0,
                "sl": 0,
                "no_hit": 0,
                "tp_rate": 0.0,
                "result_path": out_path
            }

        # --- TP/SL ATR ---
        df['entry_price'] = df['close']
        df['tp_price'] = df['entry_price'] + df['atr'] * tp_atr_mult
        df['sl_price'] = df['entry_price'] - df['atr'] * sl_atr_mult
        df.loc[df['signal'] == 'SHORT', 'tp_price'] = df['entry_price'] - df['atr'] * tp_atr_mult
        df.loc[df['signal'] == 'SHORT', 'sl_price'] = df['entry_price'] + df['atr'] * sl_atr_mult

        # --- evaluasi ---
        df = evaluate_tp_sl(df, look_ahead=look_ahead)

        # --- simpan hasil pair ---
        out_path = os.path.join(result_dir, f"hasil_backtest_{pair.lower()}_{timeframe}.xlsx")
        df.to_excel(out_path)

        # --- summary single pair ---
        total = len(df)
        tp = (df['exit_status'] == 'TP HIT').sum()
        sl = (df['exit_status'] == 'SL HIT').sum()
        no_hit = (df['exit_status'] == 'NO HIT').sum()
        tp_rate = round(tp / total * 100, 2) if total > 0 else 0.0

        single_summary = {
            "pair": pair,
            "timeframe": timeframe,
            "total": total,
            "tp": tp,
            "sl": sl,
            "no_hit": no_hit,
            "tp_rate": tp_rate,
            "result_path": out_path
        }

        # --- (opsional) regenerate summary keseluruhan ---
        if save_summary:
            summaries = []
            for f in os.listdir(result_dir):
                if f.startswith("hasil_backtest_") and f.endswith(".xlsx"):
                    _pair_tf = f.replace("hasil_backtest_", "").replace(".xlsx", "")
                    parts = _pair_tf.split("_")
                    if len(parts) >= 2:
                        _pair = "_".join(parts[:-1]).upper()
                        _tf = parts[-1]
                    else:
                        _pair = _pair_tf.upper()
                        _tf = timeframe

                    _df = pd.read_excel(os.path.join(result_dir, f))
                    _total = len(_df)
                    _tp = (_df['exit_status'] == 'TP HIT').sum() if _total else 0
                    _sl = (_df['exit_status'] == 'SL HIT').sum() if _total else 0
                    _no = (_df['exit_status'] == 'NO HIT').sum() if _total else 0
                    _rate = round(_tp / _total * 100, 2) if _total else 0.0

                    summaries.append({
                        "Pair": _pair,
                        "Timeframe": _tf,
                        "Total Sinyal": _total,
                        "TP": _tp,
                        "SL": _sl,
                        "NO HIT": _no,
                        "TP Rate (%)": _rate
                    })

           
            if summaries:
                summary_df = pd.DataFrame(summaries)

                # Hitung persentase per kategori
                summary_df["TP (%)"] = (summary_df["TP"] / summary_df["Total Sinyal"] * 100).round(2)
                summary_df["SL (%)"] = (summary_df["SL"] / summary_df["Total Sinyal"] * 100).round(2)
                summary_df["NO HIT (%)"] = (summary_df["NO HIT"] / summary_df["Total Sinyal"] * 100).round(2)

                summary_df = summary_df.sort_values(by="TP Rate (%)", ascending=False)

                summary_df.to_excel(os.path.join(result_dir, "summary_backtest.xlsx"), index=False)

    # return single_summary
    # print(summaries)
    return summaries
