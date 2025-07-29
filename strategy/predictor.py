import os
import pandas as pd
import ta
from binance.client import Client
from datetime import datetime, timezone
from utils.db import get_active_bots  # Ambil list pair+tf aktif dari DB
from utils.binance_client import get_client
from utils.timeframes import BINANCE_INTERVAL_MAP, is_time_to_run
import glob
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DATA_DIR = "./data_predict"
os.makedirs(DATA_DIR, exist_ok=True)

# === Logika Sinyal ===
def detect_signal(row):
    if pd.isna(row['macd']) or pd.isna(row['macd_signal']) or pd.isna(row['rsi']) or pd.isna(row['volume_sma20']):
        return 'HOLD'

    if row['atr'] < 0.005 * row['close']:
        return 'HOLD'
    
    # ========== LONG Condition ==========
    if row['macd'] > row['macd_signal'] and row['rsi'] > 50:
        if row['rsi'] > 70:  # Overbought → hindari entry LONG
            return 'HOLD'
        if row['volume'] < row['volume_sma20']:  # Volume rendah → hindari breakout
            return 'HOLD'
        return 'LONG'
    
     # ========== SHORT Condition ==========
    if row['macd'] < row['macd_signal'] and row['rsi'] < 50:
        if row['rsi'] < 30:  # Oversold → hindari entry SHORT
            return 'HOLD'
        if row['volume'] < row['volume_sma20']:  # Volume rendah → hindari breakdown
            return 'HOLD'
        return 'SHORT'

    if row['macd'] > row['macd_signal'] and row['rsi'] > 50:
        return 'LONG' if row['volume'] > row['volume_sma20'] else 'LONG_WEAK'

    if row['macd'] < row['macd_signal'] and row['rsi'] < 50:
        if row['rsi'] < 35:
            return 'HOLD'
        return 'SHORT'

    return 'HOLD'

# === Bersihkan folder data predict (opsional)
def clear_folder(folder_path):
    for file_path in glob.glob(os.path.join(folder_path, '*')):
        try:
            os.remove(file_path)
            print(f"🗑️ Deleted: {file_path}")
        except Exception as e:
            print(f"⚠️ Failed to delete {file_path}: {e}")

def run_predict():
    client: Client = get_client()
    active_bots = get_active_bots()
    now = datetime.now(timezone.utc)

    # clear_folder(DATA_DIR)  # Kosongkan folder sinyal & data full sebelum scrape

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

                # Hapus file lama milik pair ini
                full_path = os.path.join(DATA_DIR, f"{pair}_{timeframe}_full.xlsx")
                # pred_path = os.path.join(DATA_DIR, f"prediksi_entry_logic_{pair}.xlsx")
                for path in [full_path]:
                    if os.path.exists(path):
                        os.remove(path)

                # lanjut scrape + prediksi...

                klines = client.futures_klines(symbol=pair, interval=interval, limit=100)
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
                ])

                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                df.set_index('timestamp', inplace=True)

                # Indikator
                macd = ta.trend.MACD(df['close'])
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()
                df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
                df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
                df['volume_sma20'] = df['volume'].rolling(window=20).mean()

                # Deteksi sinyal
                df['signal'] = df.apply(detect_signal, axis=1)

                # Simpan data full OHLCV + indikator
                full_out_path = os.path.join(DATA_DIR, f"{pair}_{timeframe}_full.xlsx")
                df.to_excel(full_out_path)

                # Ambil sinyal terbaru (bar terakhir)
                now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
                df = df[df.index < now]

                if df.empty:
                    print(f"⚠️ Tidak ada candle yang fix untuk {pair}")
                    continue

                last_row = df.iloc[-1]

                # Simpan sinyal

                if last_row['signal'] in ['LONG', 'SHORT']:
                    signal_out = {
                        "pair": pair,
                        "timeframe": timeframe,
                        "signal": last_row['signal'],
                        "entry_price": last_row['close'],
                        "atr": last_row['atr'],
                        "timestamp_utc": last_row.name.tz_localize("UTC").isoformat(),
                        "timestamp_wib": (last_row.name + pd.Timedelta(hours=7)).isoformat()
                    }

                    signal_path = os.path.join(DATA_DIR, f"prediksi_entry_logic_{pair}.xlsx")
                    pd.DataFrame([signal_out]).to_excel(signal_path, index=False)
                    print(f"✅ Signal saved: {pair} {timeframe} → {last_row['signal']}")
                else:
                    print(f"⏭️ No signal for {pair} {timeframe}")

            except Exception as e:
                print(f"❌ Error for {pair}: {e}")

if __name__ == "__main__":
    run_predict()
