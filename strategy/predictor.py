# strategy/predictor.py
import os
import pandas as pd
import ta
from binance.client import Client
from datetime import datetime, timezone
from utils.db import get_active_bots
from utils.binance_client import get_client
from utils.timeframes import BINANCE_INTERVAL_MAP, is_time_to_run
from strategy.utils import calculate_support_resistance
import glob
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DATA_DIR = "./data_predict"
os.makedirs(DATA_DIR, exist_ok=True)

def detect_signal(row):
    # v1
    # if pd.isna(row['macd']) or pd.isna(row['macd_signal']) or pd.isna(row['rsi']) or pd.isna(row['volume_sma20']):
    #     return 'HOLD'

    # if row['atr'] < 0.005 * row['close']:
    #     return 'HOLD'

    # if row['macd'] > row['macd_signal'] and row['rsi'] > 50:
    #     return 'LONG' if row['volume'] > row['volume_sma20'] else 'LONG_WEAK'

    # if row['macd'] < row['macd_signal'] and row['rsi'] < 50:
    #     if row['rsi'] < 35:
    #         return 'HOLD'
    #     return 'SHORT'

    # return 'HOLD'

    # v2 dihidupkan kalau sudah urgent
    # if pd.isna(row['macd']) or pd.isna(row['macd_signal']) or pd.isna(row['rsi']) or pd.isna(row['volume_sma20']):
    #     return 'HOLD'

    # if row['atr'] < 0.005 * row['close']:
    #     return 'HOLD'
    
    # if row['macd'] > row['macd_signal'] and row['rsi'] > 50:
    #     if row['rsi'] > 70:
    #         return 'HOLD'
    #     if row['volume'] < row['volume_sma20']:
    #         return 'HOLD'
    #     return 'LONG'
    
    # if row['macd'] < row['macd_signal'] and row['rsi'] < 50:
    #     if row['rsi'] < 30:
    #         return 'HOLD'
    #     if row['volume'] < row['volume_sma20']:
    #         return 'HOLD'
    #     return 'SHORT'

    # if row['macd'] > row['macd_signal'] and row['rsi'] > 50:
    #     return 'LONG' if row['volume'] > row['volume_sma20'] else 'LONG_WEAK'

    # if row['macd'] < row['macd_signal'] and row['rsi'] < 50:
    #     if row['rsi'] < 35:
    #         return 'HOLD'
    #     return 'SHORT'

    # return 'HOLD'

    # v4
    if pd.isna(row['macd']) or pd.isna(row['macd_signal']) or pd.isna(row['rsi']) or pd.isna(row['volume_sma20']) \
       or pd.isna(row['ema_fast']) or pd.isna(row['ema_slow']):
        return 'HOLD'

    if row['atr'] < 0.005 * row['close']:
        return 'HOLD'

    # LONG → butuh MACD bullish, RSI > 50, dan EMA fast > EMA slow
    if row['macd'] > row['macd_signal'] and row['rsi'] > 50 and row['ema_fast'] > row['ema_slow']:
        return 'LONG' if row['volume'] > row['volume_sma20'] else 'LONG_WEAK'

    # SHORT → butuh MACD bearish, RSI < 50, dan EMA fast < EMA slow
    if row['macd'] < row['macd_signal'] and row['rsi'] < 50 and row['ema_fast'] < row['ema_slow']:
        if row['rsi'] < 35:
            return 'HOLD'
        return 'SHORT'

    return 'HOLD'

def clear_folder(folder_path):
    for file_path in glob.glob(os.path.join(folder_path, '*')):
        try:
            os.remove(file_path)
            print(f"🗑️ Deleted: {file_path}")
        except Exception as e:
            print(f"⚠️ Failed to delete {file_path}: {e}")

def is_false_reversal(row, df, atr_window=14, ma_fast=50, ma_slow=100):
    """
    Deteksi apakah sinyal reversal rawan false.
    Return True jika dianggap false reversal (harus di-filter)
    """
    idx = df.index.get_loc(row.name)
    close = row['close']
    
    # Ambil data historis sebelum current bar
    if idx < max(atr_window, ma_slow):
        return False  # data belum cukup
    
    # Moving Average filter
    ma50 = df['close'].rolling(ma_fast).mean().iloc[idx]
    ma100 = df['close'].rolling(ma_slow).mean().iloc[idx]

    # ATR filter
    high = df['high'].iloc[idx - atr_window:idx]
    low = df['low'].iloc[idx - atr_window:idx]
    close_prev = df['close'].iloc[idx - atr_window:idx]
    tr1 = high - low
    tr2 = abs(high - close_prev.shift(1))
    tr3 = abs(low - close_prev.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_window).mean().iloc[-1]

    # Konfirmasi multi-candle (2 candle searah)
    last2 = df['close'].iloc[idx-2:idx].reset_index(drop=True)
    bullish_confirm = all(last2[i] > df['open'].iloc[idx-2+i] for i in range(2))
    bearish_confirm = all(last2[i] < df['open'].iloc[idx-2+i] for i in range(2))

    # Deteksi false reversal untuk LONG
    if row['signal'] == 'LONG':
        if close < ma50 or close < ma100:  # masih di bawah tren
            return True
        if not bullish_confirm:  # belum ada 2 candle confirm
            return True
        if (df['open'].iloc[idx-1] - df['close'].iloc[idx-1]) > atr:  # candle merah terakhir besar
            return True

    # Deteksi false reversal untuk SHORT
    if row['signal'] == 'SHORT':
        if close > ma50 or close > ma100:  # masih di atas tren
            return True
        if not bearish_confirm:  # belum ada 2 candle confirm
            return True
        if (df['close'].iloc[idx-1] - df['open'].iloc[idx-1]) > atr:  # candle hijau terakhir besar
            return True

    return False

def apply_filters(df):
    df['false_reversal'] = df.apply(lambda row: is_false_reversal(row, df), axis=1)
    # Filter sinyal → hapus kalau false_reversal = True
    df.loc[df['false_reversal'], 'signal'] = 'HOLD'
    return df

def detect_potential_breakout(df, atr_mult=0.2, vol_mult=1.2):

    # Kondisi LONG breakout
    long_cond = (
        (df['close'] >= df['resistance'] - df['atr'] * atr_mult) &  # harga dekat/tembus resistance
        (df['close'] > df['resistance']) &  # sudah melewati resistance
        (df['volume'] >= df['volume'].rolling(20).mean() * vol_mult)  # volume di atas rata-rata
    )

    # Kondisi SHORT breakout
    short_cond = (
        (df['close'] <= df['support'] + df['atr'] * atr_mult) &  # harga dekat/tembus support
        (df['close'] < df['support']) &  # sudah melewati support
        (df['volume'] >= df['volume'].rolling(20).mean() * vol_mult)  # volume di atas rata-rata
    )

    df['is_potential_breakout'] = np.where(long_cond | short_cond, 1, 0)

    return df

def run_predict():
    client: Client = get_client()
    active_bots = get_active_bots()
    now = datetime.now(timezone.utc)

    for bot in active_bots:
        tf = bot['timeframe']

        # if not is_time_to_run(tf, now):
        #     print(f"⏱️ Skipping {bot['coin']} ({bot['timeframe']})")
        #     continue
        
        pair_text = bot['coin']
        pairs = pair_text.split(',')
        timeframe = bot['timeframe']
        interval = BINANCE_INTERVAL_MAP[timeframe]

        for pair in pairs:
            try:
                full_path = os.path.join(DATA_DIR, f"{pair}_{timeframe}_full.xlsx")
                if os.path.exists(full_path):
                    os.remove(full_path)

                klines = client.futures_klines(symbol=pair, interval=interval, limit=170)
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
                ])

                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

                # === INDICATOR
                macd = ta.trend.MACD(df['close'])
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()
                # df['macd_hist'] = macd.macd_diff()
                df['macd_hist'] = df['macd'] - df['macd_signal']

                df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
                df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
                df['volume_sma20'] = df['volume'].rolling(window=20).mean()

                bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
                df['upper_band'] = bb.bollinger_hband()
                df['lower_band'] = bb.bollinger_lband()
                # df['boll_width'] = bb.bollinger_wband()
                df['boll_width'] = df['upper_band'] - df['lower_band']
                df['bb_percentile'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])

                df['support'], df['resistance'] = calculate_support_resistance(df)
                # df['atr_multiple'] = df['atr'] / df['close']

                # df['is_potential_breakout'] = df.apply(is_potential_breakout, axis=1)
                # df['is_potential_breakout'] = (
                #     (df['high'] > df['resistance']) |
                #     (df['low'] < df['support'])
                # )

                # EMA untuk crossing line filter
                df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()
                df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()

                df['signal'] = df.apply(detect_signal, axis=1)
                df = apply_filters(df)
                df = detect_potential_breakout(df)

                df_signal = df[df['signal'].isin(['LONG', 'SHORT'])].copy()

               # Hitung breakout setelah ada signal, support, resistance
               # df_signal['is_potential_breakout'] = (
               #     (df_signal['high'] > df_signal['resistance']) |
               #     (df_signal['low'] < df_signal['support'])
               # )

                # Hitung atr_multiple untuk df_signal saja
                df_signal['atr_multiple'] = np.where(
                    df_signal['signal'] == 'LONG',
                    (df_signal['close'] - df_signal['resistance']) / df_signal['atr'],
                    (df_signal['support'] - df_signal['close']) / df_signal['atr']
                )

                df_signal['entry_signal'] = ((df_signal['is_potential_breakout'] == 1) & (df_signal['signal'].notna())).astype(int)

                signal_map = {'HOLD': 0, 'LONG': 1, 'SHORT': -1, 'LONG_WEAK': 0}
                df_signal['signal_numeric'] = df_signal['signal'].map(signal_map)

                # Set index & timezone untuk df_signal
                df_signal.set_index('timestamp', inplace=True)
                df_signal.index = df_signal.index.tz_localize('UTC')

                now = pd.Timestamp.now(tz='UTC').replace(minute=0, second=0, microsecond=0)
                df_signal = df_signal[df_signal.index < now]
                df_signal.index = df_signal.index.tz_convert(None)

                # Simpan ke Excel
                df_signal.to_excel(full_path)

                # last_row = df.iloc[-1]
                # if last_row['signal'] in ['LONG', 'SHORT']:
                #     last_row_df = pd.DataFrame([last_row])
                #     signal_out = {
                #         "pair": pair,
                #         "timeframe": timeframe,
                #         "signal": last_row['signal'],
                #         "entry_price": last_row['close'],
                #         "atr": last_row['atr'],
                #         "timestamp_utc": last_row.name.replace(tzinfo=None).isoformat(),
                #         "timestamp_wib": (last_row.name + pd.Timedelta(hours=7)).replace(tzinfo=None).isoformat(),
                #     }

                #     signal_path = os.path.join(DATA_DIR, f"prediksi_entry_logic_{pair}.xlsx")
                #     pd.DataFrame([signal_out]).to_excel(signal_path, index=False)
                #     print(f"✅ Signal saved: {pair} {timeframe} → {last_row['signal']}")
                # else:
                #     print(f"⏭️ No signal for {pair} {timeframe}")

                last_row = df_signal.iloc[-1]

                ts = last_row.name

                last_row = last_row.copy()
                last_row['timestamp'] = ts
                if last_row['signal'] in ['LONG', 'SHORT']:
                   last_row_df = pd.DataFrame([last_row])

                   last_row_df['entry_price'] = last_row['close']
                   last_row_df['timestamp_utc'] = ts
                   last_row_df['timestamp_wib'] = ts + pd.Timedelta(hours=7)

                   signal_path = os.path.join(DATA_DIR, f"prediksi_entry_logic_{pair}.xlsx")
                   last_row_df.to_excel(signal_path, index=False)
                   print(f"✅ Signal saved: {pair} {timeframe} → {last_row['signal']}")

                else:
                    print(f"⏭️ No signal for {pair} {timeframe}")

            except Exception as e:
                print(f"❌ Error for {pair}: {e}")

if __name__ == "__main__":
    run_predict()
