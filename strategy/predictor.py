import pandas as pd
import ta
from datetime import datetime, timedelta, timezone
from binance.client import Client
import os

from utils.timeframes import BINANCE_INTERVAL_MAP
from config import DATA_DIR, PREDICT_DIR, TIMEZONE_OFFSET_HOURS

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PREDICT_DIR, exist_ok=True)

# --- Logika sinyal yang kamu punya ---
def detect_signal(row):
    if pd.isna(row['macd']) or pd.isna(row['macd_signal']) or pd.isna(row['rsi']) or pd.isna(row['volume_sma20']):
        return 'HOLD'
    if row['macd'] > row['macd_signal'] and row['rsi'] > 50:
        return 'LONG' if row['volume'] > row['volume_sma20'] else 'LONG_WEAK'
    elif row['macd'] < row['macd_signal'] and row['rsi'] < 50:
        return 'SHORT'
    else:
        return 'HOLD'


def fetch_klines(client: Client, symbol: str, tf: str, limit: int = 100):
    interval = BINANCE_INTERVAL_MAP[tf]

    now = datetime.now(timezone.utc)
    rounded = now.replace(second=0, microsecond=0)
    if tf == '1h':
        rounded = rounded.replace(minute=0)
    elif tf == '4h':
        m = rounded.hour % 4
        rounded = rounded.replace(hour=rounded.hour - m, minute=0)
    elif tf == '15m':
        m = rounded.minute % 15
        rounded = rounded.replace(minute=rounded.minute - m)

    end_time_dt = rounded - timedelta(seconds=1)
    end_time = int(end_time_dt.timestamp() * 1000)

    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit, endTime=end_time)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df.set_index('timestamp', inplace=True)
    return df


def generate_signal(client: Client, symbol: str, tf: str, tp_percent: float, sl_percent: float, atr_multiplier: float = 1.0, save_excel: bool = False):
    df = fetch_klines(client, symbol, tf, limit=100)
    data_path = os.path.join(DATA_DIR, f"{symbol}_{tf}.csv")
    df.to_csv(data_path)

    macd_ind = ta.trend.MACD(df['close'])
    df['macd'] = macd_ind.macd()
    df['macd_signal'] = macd_ind.macd_signal()
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['volume_sma20'] = df['volume'].rolling(window=20).mean()

    df['signal'] = df.apply(detect_signal, axis=1)
    df = df[df['signal'].isin(['LONG', 'SHORT'])].copy()
    if df.empty:
        return None

    last = df.iloc[-1].copy()
    entry_price = last['close']

    # TP/SL from percent (not ATR)
    if last['signal'] == 'LONG':
        tp_price = entry_price * (1 + tp_percent / 100.0)
        sl_price = entry_price * (1 - sl_percent / 100.0)
    else:  # SHORT
        tp_price = entry_price * (1 - tp_percent / 100.0)
        sl_price = entry_price * (1 + sl_percent / 100.0)

    last['entry_price'] = entry_price
    last['tp_price'] = tp_price
    last['sl_price'] = sl_price

    last['timestamp_utc'] = last.name
    last['timestamp_wib'] = last.name + pd.Timedelta(hours=TIMEZONE_OFFSET_HOURS)

    result = {
        'symbol': symbol,
        'timeframe': tf,
        'signal': last['signal'],
        'entry_price': entry_price,
        'tp_price': tp_price,
        'sl_price': sl_price,
        'atr': last['atr'],
        'close': last['close'],
        'timestamp_utc': last['timestamp_utc'],
        'timestamp_wib': last['timestamp_wib']
    }

    if save_excel:
        out_path = os.path.join(PREDICT_DIR, f"prediksi_{symbol}_{tf}.xlsx")
        pd.DataFrame([result]).to_excel(out_path, index=False)

    return result