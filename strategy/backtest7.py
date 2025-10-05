"""
forward_test_regime_bot.py
Bot forward-test untuk deteksi regime (TREND / SIDEWAYS)
Menggunakan ADX + ATR + Bollinger Band Width
Mencatat hasil ke Excel untuk analisis performa
"""

import asyncio
import pandas as pd
import ta   # pip install ta
import numpy as np
from datetime import datetime
from binance.client import Client
from binance.enums import *
from dotenv import load_dotenv
import os

load_dotenv()

# ========== CONFIG ==========
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
SYMBOL = "AVAXUSDT"
TIMEFRAME = "5m"            # timeframe forward-test
TRADE_QTY = 0.05            # contoh: 0.05 AVAX
USE_TESTNET = True          # set True untuk testnet
FORWARD_LOGFILE = "forward_test_log.xlsx"

# Threshold regime detection
ADX_TREND_THRESHOLD = 25
ADX_SIDEWAYS_THRESHOLD = 20
BB_WIDTH_LOW = 0.06         # band sempit → sideways
BB_WIDTH_HIGH = 0.10        # band lebar → trend
ATR_PERIOD = 14

client = Client(API_KEY, API_SECRET, testnet=USE_TESTNET)

# ========== UTIL ==========

def get_candles(symbol=SYMBOL, interval=TIMEFRAME, limit=200):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp','open','high','low','close','volume','close_time',
        'qav','num_trades','taker_base','taker_quote','ignore'])
    
    # konversi timestamp ke datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # pilih kolom yang diperlukan
    df = df[['timestamp','open','high','low','close','volume']].copy()
    
    # ubah hanya kolom angka ke float
    cols = ['open','high','low','close','volume']
    df[cols] = df[cols].astype(float)
    
    return df


def detect_regime(df):
    # Hitung indikator
    close = df['close']
    high = df['high']
    low = df['low']

    # ADX
    adx = ta.trend.adx(high, low, close, window=14).iloc[-1]

    # ATR
    atr = ta.volatility.average_true_range(high, low, close, window=ATR_PERIOD).iloc[-1]

    # Bollinger Band Width
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_width = ((bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()).iloc[-1]

    regime = "SIDEWAYS"
    if adx >= ADX_TREND_THRESHOLD or bb_width >= BB_WIDTH_HIGH:
        regime = "TREND"
    elif adx < ADX_SIDEWAYS_THRESHOLD and bb_width < BB_WIDTH_LOW:
        regime = "SIDEWAYS"
    else:
        regime = "TRANSITION"

    return regime, adx, atr, bb_width

def log_trade(regime, order_type, entry_price, exit_price, pnl):
    data = {
        "timestamp": [datetime.utcnow()],
        "regime": [regime],
        "order_type": [order_type],
        "entry_price": [entry_price],
        "exit_price": [exit_price],
        "pnl_pct": [pnl]
    }

    df = pd.DataFrame(data)
    try:
        old = pd.read_excel(FORWARD_LOGFILE)
        df = pd.concat([old, df], ignore_index=True)
    except FileNotFoundError:
        pass

    df.to_excel(FORWARD_LOGFILE, index=False)

# ========== SIMULASI FORWARD TEST (DUMMY ORDER) ==========
# Bot ini hanya untuk forward test regime dan log
# Belum mengirim order ke exchange (aman untuk testing)

async def run_forward_test():
    print("🚀 Forward-test regime bot started.")
    last_regime = None
    while True:
        try:
            df = get_candles()
            regime, adx, atr, bb_width = detect_regime(df)

            if regime != last_regime:
                print(f"[{datetime.utcnow()}] Regime changed: {regime} | ADX:{adx:.2f} | BBW:{bb_width:.4f}")
                last_regime = regime

            # === Simulasi sinyal ===
            close_price = df['close'].iloc[-1]
            # Dummy P/L
            pnl = np.random.uniform(-0.003, 0.005)  # -0.3% s/d +0.5% untuk testing log
            order_type = "LIMIT" if regime == "SIDEWAYS" else "MARKET"

            # Simpan log setiap loop
            log_trade(regime, order_type, close_price, close_price*(1+pnl), pnl*100)

            await asyncio.sleep(300)  # tunggu 5 menit sesuai timeframe
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(run_forward_test())
