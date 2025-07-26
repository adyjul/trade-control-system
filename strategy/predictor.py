import os
import pandas as pd
import ta
from binance.client import Client
from datetime import datetime
from utils.db import get_active_bots  # Ambil list pair+tf aktif dari DB
from utils.binance_client import get_client
from utils.timeframes import BINANCE_INTERVAL_MAP

DATA_DIR = "./data_predict"
os.makedirs(DATA_DIR, exist_ok=True)

# === Logika Sinyal ===
def detect_signal(row):
    if pd.isna(row['macd']) or pd.isna(row['macd_signal']) or pd.isna(row['rsi']) or pd.isna(row['volume_sma20']):
        return 'HOLD'

    if row['atr'] < 0.005 * row['close']:
        return 'HOLD'

    if row['macd'] > row['macd_signal'] and row['rsi'] > 50:
        return 'LONG' if row['volume'] > row['volume_sma20'] else 'LONG_WEAK'

    if row['macd'] < row['macd_signal'] and row['rsi'] < 50:
        if row['rsi'] < 35:
            return 'HOLD'
        return 'SHORT'

    return 'HOLD'


def run_predict():
    client: Client = get_client()
    active_bots = get_active_bots()

    for bot in active_bots:
        pair_text = bot['pair']
        pairs = pair_text.split(',')
        timeframe = bot['timeframe']
        interval = BINANCE_INTERVAL_MAP[timeframe]

        for pair in pairs:
            try:
                klines = client.futures_klines(symbol=pair, interval=interval, limit=100)
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
                ])

                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
                df.set_index('timestamp', inplace=True)

                # Indikator
                macd = ta.trend.MACD(df['close'])
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()
                df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
                df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
                df['volume_sma20'] = df['volume'].rolling(window=20).mean()

                df['signal'] = df.apply(detect_signal, axis=1)

                last_row = df.iloc[-1]
                if last_row['signal'] in ['LONG', 'SHORT']:
                    signal_out = {
                        "pair": pair,
                        "timeframe": timeframe,
                        "signal": last_row['signal'],
                        "entry_price": last_row['close'],
                        "timestamp": last_row.name.isoformat()
                    }
                    out_path = os.path.join(DATA_DIR, f"{pair}_{timeframe}.xlsx")
                    pd.DataFrame([signal_out]).to_excel(out_path, index=False)
                    print(f"✅ Signal saved: {pair} {timeframe} → {signal_out['signal']}")
                else:
                    print(f"⏭️ No signal for {pair} {timeframe}")

            except Exception as e:
                print(f"❌ Error for {pair}: {e}")


if __name__ == "__main__":
    run_predict()
