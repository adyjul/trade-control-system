"""
forward_test_regime_bot_tick.py
Forward-test deteksi regime + TP/SL tick mode
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from binance import AsyncClient,BinanceSocketManager
import ta                   # pip install ta
import openpyxl
import os
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIG =================
API_KEY     = os.getenv('BINANCE_API_KEY')
API_SECRET  = os.getenv('BINANCE_API_SECRET')
SYMBOL      = "AVAXUSDT"
TIMEFRAME   = "5m"
TP_PCT      = 0.005         # 0.5% TP
SL_PCT      = 0.003         # 0.3% SL
LOG_FILE    = "forward_test_tick.xlsx"
USE_TESTNET = False

ADX_TREND_THRESHOLD = 25
ADX_SIDEWAYS_THRESHOLD = 20
BB_WIDTH_LOW  = 0.06
BB_WIDTH_HIGH = 0.10

# ================= UTIL ===================
async def get_candles(client, limit=200):
    """Ambil candle historis untuk indikator"""
    kl = await client.get_klines(symbol=SYMBOL, interval=TIMEFRAME, limit=limit)
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
    close = df['close']
    high  = df['high']
    low   = df['low']

    adx = ta.trend.adx(high, low, close, window=14).iloc[-1]
    atr = ta.volatility.average_true_range(high, low, close, window=14).iloc[-1]
    bb  = ta.volatility.BollingerBands(close, window=20, window_dev=2)
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
    print(f"LOGGED: {result} | {regime} | PnL: {pnl*100:.2f}%")

# ================= CORE ====================
async def forward_test():
    print("🚀 Forward-test Tick Mode started …")
    client = await AsyncClient.create(API_KEY, API_SECRET, testnet=USE_TESTNET)
    bm = BinanceSocketManager(client)

    # ambil regime awal
    candles = await get_candles(client)
    regime, adx, atr, bbw = detect_regime(candles)
    print(f"[INIT] Regime: {regime}, ADX:{adx:.2f}, BBW:{bbw:.4f}")

    current_trade = None
    # simpan menit candle terakhir (mis. menit ke-5, 10, 15, dst.)
    last_candle_minute = (datetime.utcnow().minute // 5) * 5

    async with bm.trade_socket(SYMBOL.lower()) as stream:
        while True:
            msg = await stream.recv()
            price = float(msg['p'])
            now   = datetime.utcnow()

            # hitung menit candle sekarang
            current_candle_minute = (now.minute // 5) * 5

            # jika candle baru close → update regime
            if current_candle_minute != last_candle_minute and now.second < 3:
                candles = await get_candles(client)
                regime, adx, atr, bbw = detect_regime(candles)
                last_candle_minute = current_candle_minute
                print(f"[{now}] Regime update: {regime} | ADX:{adx:.1f} | BBW:{bbw:.3f}")

                # HANYA entry saat candle BARU terbentuk
                if current_trade is None and regime in ["TREND", "SIDEWAYS"]:
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

            # cek TP / SL
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

            await asyncio.sleep(0.05)


# ===========================================
if __name__ == "__main__":
    asyncio.run(forward_test())
