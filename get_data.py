import ccxt
import pandas as pd
import talib

# 🔧 SETTINGS — EDIT THESE DIRECTLY
SYMBOLS = [
    'ZKP/USDT',
]
TIMEFRAME = '15m'      # e.g., '15m', '1h', '4h', '1d'
ROWS = 1000
OUTPUT_FILE = 'futures_data.xlsx'

# ----------------------------
# Use Binance USDⓈ-M Futures (perpetual contracts)
exchange = ccxt.binanceusdm({
    'options': {'defaultType': 'future'}
})

print(f"🚀 Fetching {ROWS} {TIMEFRAME} futures candles for {len(SYMBOLS)} symbols...")

with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
    for symbol in SYMBOLS:
        try:
            print(f"  → {symbol}")
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=ROWS)
            
            if not ohlcv:
                print(f"    ⚠️ No data for {symbol}")
                continue

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Excel sheet names can't have / and must be <=31 chars
            sheet_name = symbol.replace("/", "")[:31]
            df.to_excel(writer, sheet_name=sheet_name)

        except Exception as e:
            print(f"    ❌ Failed {symbol}: {str(e)}")

print(f"\n✅ All done! Futures data saved to → {OUTPUT_FILE}")