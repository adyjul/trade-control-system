# strategy/backtest_trend_rider_aggressive.py
import os
from datetime import datetime, timedelta
import pandas as pd
import ta
from binance.client import Client
from utils.binance_client import get_client
from utils.timeframes import BINANCE_INTERVAL_MAP
import numpy as np
import glob

DEFAULT_DATA_DIR = "./data_backtest_trend_rider"
DEFAULT_RESULT_DIR = "./backtest_result_trend_rider"

def clear_folder(folder_path):
    for file_path in glob.glob(os.path.join(folder_path, '*')):
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except:
            pass

def period_to_start_timestamp(period: str) -> int:
    now = datetime.utcnow()
    if period == "6m":
        start = now - timedelta(days=30*6)
    elif period == "1y":
        start = now - timedelta(days=365)
    elif period == "2y":
        start = now - timedelta(days=365*2)
    else:
        raise ValueError("Period harus 6m, 1y, atau 2y")
    return int(start.timestamp() * 1000)

def calculate_indicators(df):
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['ma50'] = df['close'].rolling(50).mean()
    df['ma100'] = df['close'].rolling(100).mean()
    return df

def detect_signal_1h(row):
    if pd.isna(row['macd']) or pd.isna(row['rsi']):
        return 'HOLD'
    if row['macd'] > row['macd_signal'] and row['rsi'] > 50:
        return 'LONG'
    elif row['macd'] < row['macd_signal'] and row['rsi'] < 50:
        return 'SHORT'
    return 'HOLD'

def detect_trend_4h(row):
    if pd.isna(row['ma50']) or pd.isna(row['ma100']):
        return 'NEUTRAL'
    return 'UP' if row['ma50'] > row['ma100'] else 'DOWN'

def evaluate_tp_sl(df, look_ahead=6, tp_mult=1.0, sl_mult=0.8):
    df['exit_status'] = 'NO HIT'
    for i in range(len(df)):
        row = df.iloc[i]
        signal = row['signal']
        tp = row['entry_price'] + row['atr'] * tp_mult if signal == 'LONG' else row['entry_price'] - row['atr'] * tp_mult
        sl = row['entry_price'] - row['atr'] * sl_mult if signal == 'LONG' else row['entry_price'] + row['atr'] * sl_mult

        future = df.iloc[i+1:i+1+look_ahead]
        for _, f in future.iterrows():
            if signal == 'LONG':
                if f['high'] >= tp:
                    df.at[df.index[i], 'exit_status'] = 'TP HIT'
                    break
                elif f['low'] <= sl:
                    df.at[df.index[i], 'exit_status'] = 'SL HIT'
                    break
            elif signal == 'SHORT':
                if f['low'] <= tp:
                    df.at[df.index[i], 'exit_status'] = 'TP HIT'
                    break
                elif f['high'] >= sl:
                    df.at[df.index[i], 'exit_status'] = 'SL HIT'
                    break
    return df

def run_trend_rider_backtest(pairs, start_date, end_date, look_ahead=6, tp_mult=1.0, sl_mult=0.8):
    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
    os.makedirs(DEFAULT_RESULT_DIR, exist_ok=True)
    clear_folder(DEFAULT_DATA_DIR)
    clear_folder(DEFAULT_RESULT_DIR)

    client = get_client()

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    results_summary = []

    for pair in pairs:
        # Scrape 1H data
        klines_1h = client.futures_klines(symbol=pair, interval=BINANCE_INTERVAL_MAP['1h'], limit=1500, startTime=start_ts, endTime=end_ts)
        df_1h = pd.DataFrame(klines_1h, columns=['timestamp','open','high','low','close','volume','close_time','qav','nt','tbbv','tbqv','ignore'])
        df_1h[['open','high','low','close','volume']] = df_1h[['open','high','low','close','volume']].astype(float)
        df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], unit='ms')
        df_1h.set_index('timestamp', inplace=True)
        df_1h = calculate_indicators(df_1h)

        # Scrape 4H data
        klines_4h = client.futures_klines(symbol=pair, interval=BINANCE_INTERVAL_MAP['4h'], limit=1500, startTime=start_ts, endTime=end_ts)
        df_4h = pd.DataFrame(klines_4h, columns=['timestamp','open','high','low','close','volume','close_time','qav','nt','tbbv','tbqv','ignore'])
        df_4h[['open','high','low','close','volume']] = df_4h[['open','high','low','close','volume']].astype(float)
        df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], unit='ms')
        df_4h.set_index('timestamp', inplace=True)
        df_4h = calculate_indicators(df_4h)
        df_4h['trend_4h'] = df_4h.apply(detect_trend_4h, axis=1)

        # Gabungkan tren 4H ke 1H
        df_1h['trend_4h'] = df_1h.index.map(lambda x: df_4h['trend_4h'].asof(x))

        # Tentukan sinyal
        df_1h['signal'] = df_1h.apply(detect_signal_1h, axis=1)
        # Hanya ambil jika searah tren 4H
        df_1h.loc[((df_1h['signal']=='LONG') & (df_1h['trend_4h']!='UP')) |
                  ((df_1h['signal']=='SHORT') & (df_1h['trend_4h']!='DOWN')), 'signal'] = 'HOLD'

        df_1h = df_1h[df_1h['signal'].isin(['LONG','SHORT'])]
        if df_1h.empty:
            continue

        df_1h['entry_price'] = df_1h['close']
        df_1h = evaluate_tp_sl(df_1h, look_ahead=look_ahead, tp_mult=tp_mult, sl_mult=sl_mult)
        df_1h['label'] = df_1h['exit_status'].map({'TP HIT':1,'SL HIT':0,'NO HIT':-1})

        out_path = os.path.join(DEFAULT_RESULT_DIR, f"{pair}_trend_rider_aggressive.xlsx")
        df_1h.to_excel(out_path)
        print(f"📄 Backtest saved: {pair}")

        # Summary
        total = len(df_1h)
        tp = (df_1h['exit_status']=='TP HIT').sum()
        sl = (df_1h['exit_status']=='SL HIT').sum()
        tp_rate = round(tp/total*100,2) if total else 0
        results_summary.append({
            'Pair': pair,
            'Total Sinyal': total,
            'TP': tp,
            'SL': sl,
            'TP Rate (%)': tp_rate,
            'Result': out_path
        })

    return pd.DataFrame(results_summary)
