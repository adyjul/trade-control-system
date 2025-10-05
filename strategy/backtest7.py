"""
forward_test_regime_bot_tick.py
Forward-test untuk deteksi regime + TP/SL dengan tick mode
Regime: ADX + ATR + Bollinger Band Width
Hanya simulasikan trade (tidak eksekusi real order)
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from binance.client import Client
from binance import BinanceSocketManager
import ta   # pip install ta
import openpyxl
import os
from dotenv import load_dotenv
load_dotenv()

# ================= CONFIG =================
API_KEY     = os.getenv('BINANCE_API_KEY')
API_SECRET  = os.getenv('BINANCE_API_KEY')
SYMBOL      = "AVAXUSDT"
TIMEFRAME   = "5m"          # timeframe untuk indikator
TP_PCT      = 0.005         # TP 0.5%
SL_PCT      = 0.003         # SL 0.3%
LOG_FILE    = "forward_test_tick.xlsx"
USE_TESTNET = False

ADX_TREND_THRESHOLD = 25
ADX_SIDEWAYS_THRESHOLD = 20
BB_WIDTH_LOW  = 0.06
BB_WIDTH_HIGH = 0.10

client = Client(API_KEY, API_SECRET)


# ================= UTIL ===================
def get_candles(limit=200):
    """Ambil candle historis untuk indikator"""
    kl = client.get_klines(symbol=SYMBOL, interval=TIMEFRAME, limit=limit)
    df = pd.DataFrame(kl, columns=[
        'timestamp','open','high','low','close','volume','close_time',
        'qav','num_trades','taker_base','taker_quote','ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp','open','high','low','close','volume']].copy()
    cols = ['open','high','low','close','volume']
    df[cols] = df[cols].astype(float)
    return df

def detect_regime(df):
    """Deteksi trend / sideways"""
    close = df['close']
    high  = df['high']
    low   = df['low']

    adx = ta.trend.adx(high, low, close, window=14).iloc[-1]
    atr = ta.volatility.average_true_range(high, low, close, window=14).iloc[-1]

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_width = ((bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()).iloc[-1]

    if adx >= ADX_TREND_THRESHOLD or bb_width >= BB_WIDTH_HIGH:
        regime = "TREND"
    elif adx < ADX_SIDEWAYS_THRESHOLD and bb_width < BB_WIDTH_LOW:
        regime = "SIDEWAYS"
    else:
        regime = "TRANSITION"

    return regime, adx, atr, bb_width

def log_trade(entry_time, regime, order_type, entry_price, exit_price, result, pnl):
    data = {
        "timestamp":   [entry_time],
        "regime":      [regime],
        "order_type":  [order_type],
        "entry_price": [entry_price],
        "exit_price":  [exit_price],
        "result":      [result],     # TP / SL
        "pnl_pct":     [pnl]
    }
    df = pd.DataFrame(data)
    try:
        old = pd.read_excel(LOG_FILE)
        df  = pd.concat([old, df], ignore_index=True)
    except FileNotFoundError:
        pass
    df.to_excel(LOG_FILE, index=False)
    print(f"LOGGED: {result} | PnL: {pnl*100:.2f}%")

# ================= CORE ====================
async def forward_test():
    print("🚀 Forward-test Tick Mode started …")
    bm = BinanceSocketManager(client)
    ts = bm.trade_socket(SYMBOL.lower())

    current_trade = None

    # Regime update awal
    candles = get_candles()
    regime, adx, atr, bbw = detect_regime(candles)
    print(f"[INIT] Regime: {regime}, ADX:{adx:.2f}, BBW:{bbw:.4f}")

    async with ts as stream:
        async for msg in stream:
            price = float(msg['p'])  # tick price
            now   = datetime.utcnow()

            # update regime tiap close candle baru (setiap 5m)
            if now.minute % 5 == 0 and now.second < 3:
                candles = get_candles()
                regime, adx, atr, bbw = detect_regime(candles)
                print(f"[{now}] Regime: {regime} ADX:{adx:.1f} BBW:{bbw:.3f}")

            # jika tidak ada trade aktif → buka simulasi posisi
            if current_trade is None and regime in ["TREND","SIDEWAYS"]:
                order_type = "MARKET" if regime == "TREND" else "LIMIT"
                current_trade = {
                    "entry_time": now,
                    "entry_price": price,
                    "regime": regime,
                    "order_type": order_type,
                    "tp": price * (1 + TP_PCT),
                    "sl": price * (1 - SL_PCT)
                }
                print(f"ENTRY [{regime}] ({order_type}) @ {price:.4f}")

            # jika ada trade → cek TP/SL
            if current_trade:
                if price >= current_trade["tp"]:
                    log_trade(current_trade["entry_time"], current_trade["regime"],
                              current_trade["order_type"],
                              current_trade["entry_price"], price, "TP", TP_PCT)
                    current_trade = None

                elif price <= current_trade["sl"]:
                    log_trade(current_trade["entry_time"], current_trade["regime"],
                              current_trade["order_type"],
                              current_trade["entry_price"], price, "SL", -SL_PCT)
                    current_trade = None

            await asyncio.sleep(0.1)  # small delay to avoid busy loop

# ===========================================
if __name__ == "__main__":
    asyncio.run(forward_test())
