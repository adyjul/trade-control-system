import ccxt
import pandas as pd
import argparse
from datetime import datetime

def fetch_data(symbol, timeframe, limit):
    print(f"📥 Fetching {limit} {timeframe} candles for {symbol} from Binance...")
    exchange = ccxt.binance()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def main():
    parser = argparse.ArgumentParser(description="Fetch crypto price data and save to Excel (.xlsx)")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair (e.g., PEPE/USDT, HYPE/USDT)")
    parser.add_argument("--timeframe", type=str, default="4h", help="Timeframe (e.g., 15m, 1h, 4h, 1d)")
    parser.add_argument("--rows", type=int, default=100, help="Number of candles to fetch")
    parser.add_argument("--output", type=str, default="", help="Output Excel filename (e.g., data.xlsx)")

    args = parser.parse_args()

    df = fetch_data(args.symbol, args.timeframe, args.rows)
    if df is None:
        return

    # Auto-generate filename if not provided
    if not args.output:
        safe_symbol = args.symbol.replace("/", "_")
        args.output = f"{safe_symbol}_{args.timeframe}_{args.rows}rows.xlsx"

    # Save to Excel
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='OHLCV')

    print(f"✅ Data saved to Excel: {args.output}")
    print(f"📅 Range: {df.index[0]} → {df.index[-1]}")
    print(f"📊 Rows: {len(df)} | Columns: {list(df.columns)}")

    # Show last 3 close prices
    print("\nLast 3 close prices:")
    print(df['close'].tail(3))

if __name__ == "__main__":
    main()