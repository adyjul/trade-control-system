# strategy/backtest_trend_rider.py
import os
from datetime import datetime, timedelta
import pandas as pd
import ta
from binance.client import Client
import glob
from strategy.utils import calculate_support_resistance
from utils.binance_client import get_client
from utils.timeframes import BINANCE_INTERVAL_MAP
import numpy as np

DEFAULT_DATA_DIR = "./data_backtest"
DEFAULT_RESULT_DIR = "./backtest_result"

# ---------------- Trend-Rider Functions ----------------

def get_trend(df, long_window=50, short_window=200):
    """Hitung tren: SMA cross. >0 = uptrend, <0 = downtrend"""
    ma_short = df['close'].rolling(window=short_window).mean()
    ma_long = df['close'].rolling(window=long_window).mean()
    trend = ma_short - ma_long
    return trend

def detect_signal_trend_rider(row, trend_tf):
    """Entry hanya kalau tren besar mendukung"""
    if pd.isna(row['macd']) or pd.isna(row['macd_signal']) or pd.isna(row['rsi']):
        return 'HOLD'

    if row['atr'] < 0.005 * row['close']:
        return 'HOLD'

    if row['macd'] > row['macd_signal'] and row['rsi'] > 50:
        if trend_tf < 0:
            return 'HOLD'
        return 'LONG'

    if row['macd'] < row['macd_signal'] and row['rsi'] < 50:
        if trend_tf > 0:
            return 'HOLD'
        return 'SHORT'

    return 'HOLD'

# ---------------- Existing Functions ----------------

def evaluate_tp_sl(df: pd.DataFrame, look_ahead=7) -> pd.DataFrame:
    df['exit_status'] = 'NO HIT'
    idxs = list(df.index)
    for i, ts in enumerate(idxs):
        row = df.loc[ts]
        signal = row['signal']
        tp = row['tp_price']
        sl = row['sl_price']
        idx_start = df.index.get_loc(row.name)
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

def is_false_reversal(row, df, atr_window=14, ma_fast=50, ma_slow=100):
    idx = df.index.get_loc(row.name)
    close = row['close']
    if idx < max(atr_window, ma_slow):
        return False
    ma50 = df['close'].rolling(ma_fast).mean().iloc[idx]
    ma100 = df['close'].rolling(ma_slow).mean().iloc[idx]
    high = df['high'].iloc[idx - atr_window:idx]
    low = df['low'].iloc[idx - atr_window:idx]
    close_prev = df['close'].iloc[idx - atr_window:idx]
    tr1 = high - low
    tr2 = abs(high - close_prev.shift(1))
    tr3 = abs(low - close_prev.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_window).mean().iloc[-1]
    last2 = df['close'].iloc[idx-2:idx].reset_index(drop=True)
    bullish_confirm = all(last2[i] > df['open'].iloc[idx-2+i] for i in range(2))
    bearish_confirm = all(last2[i] < df['open'].iloc[idx-2+i] for i in range(2))
    if row['signal'] == 'LONG':
        if close < ma50 or close < ma100: return True
        if not bullish_confirm: return True
        if (df['open'].iloc[idx-1] - df['close'].iloc[idx-1]) > atr: return True
    if row['signal'] == 'SHORT':
        if close > ma50 or close > ma100: return True
        if not bearish_confirm: return True
        if (df['close'].iloc[idx-1] - df['open'].iloc[idx-1]) > atr: return True
    return False

def apply_filters(df):
    df['false_reversal'] = df.apply(lambda row: is_false_reversal(row, df), axis=1)
    df.loc[df['false_reversal'], 'signal'] = 'HOLD'
    return df

def clear_folder(folder_path):
    for file_path in glob.glob(os.path.join(folder_path, '*')):
        try: os.remove(file_path)
        except: pass

def run_full_backtest(
    pairs, timeframe, period=None, start_date=None, end_date=None,
    look_ahead=6, tp_atr_mult=1.2, sl_atr_mult=0.9,
    data_dir=DEFAULT_DATA_DIR, result_dir=DEFAULT_RESULT_DIR
):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    client: Client = get_client()
    interval = BINANCE_INTERVAL_MAP[timeframe]
    clear_folder(data_dir)
    clear_folder(result_dir)

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000) if start_date else int((datetime.utcnow() - timedelta(days=30*6)).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000) if end_date else int(datetime.utcnow().timestamp() * 1000)

    summaries = []

    for pair in pairs:
        # --- Scrape ---
        klines = []
        limit = 1500
        start = start_ts
        while start < end_ts:
            batch = client.futures_klines(symbol=pair, interval=interval, limit=limit, startTime=start, endTime=end_ts)
            if not batch: break
            klines.extend(batch)
            last_time = batch[-1][0]
            start = last_time + 1
        df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume','ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
        df.set_index('timestamp', inplace=True)

        # --- Indicators ---
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['support'], df['resistance'] = calculate_support_resistance(df)
        df['volume_sma20'] = df['volume'].rolling(window=20).mean()
        df['trend_4H'] = get_trend(df, long_window=50, short_window=200)

        # --- Signals ---
        df['signal'] = df.apply(lambda row: detect_signal_trend_rider(row, row['trend_4H']), axis=1)
        df = apply_filters(df)

        # --- TP/SL ---
        df = df[df['signal'].isin(['LONG','SHORT'])]
        if df.empty: continue
        df['entry_price'] = df['close']
        df['tp_price'] = df['entry_price'] + df['atr'] * tp_atr_mult
        df['sl_price'] = df['entry_price'] - df['atr'] * sl_atr_mult
        df.loc[df['signal'] == 'SHORT', 'tp_price'] = df['entry_price'] - df['atr'] * tp_atr_mult
        df.loc[df['signal'] == 'SHORT', 'sl_price'] = df['entry_price'] + df['atr'] * sl_atr_mult

        df = evaluate_tp_sl(df, look_ahead=look_ahead)
        df['label'] = df['exit_status'].map({'TP HIT':1,'SL HIT':0,'NO HIT':-1})

        # --- Save ---
        out_path = os.path.join(result_dir, f"hasil_backtest_{pair.lower()}_{timeframe}.xlsx")
        df.to_excel(out_path)
        total = len(df)
        tp = (df['exit_status'] == 'TP HIT').sum()
        sl = (df['exit_status'] == 'SL HIT').sum()
        no_hit = (df['exit_status'] == 'NO HIT').sum()
        tp_rate = round(tp/total*100,2) if total else 0.0
        summaries.append({
            "Pair": pair,
            "Timeframe": timeframe,
            "Total Sinyal": total,
            "TP": tp,
            "SL": sl,
            "NO HIT": no_hit,
            "TP Rate (%)": tp_rate
        })

    # --- Summary ---
    if summaries:
        summary_df = pd.DataFrame(summaries).sort_values(by="TP Rate (%)", ascending=False)
        summary_df.to_excel(os.path.join(result_dir, "summary_backtest.xlsx"), index=False)
        print("📊 Summary saved!")

    return summaries
