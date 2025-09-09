# live_bot_m1.py
import pandas as pd
import numpy as np
import asyncio
import json
from datetime import datetime, timezone
from binance import AsyncClient, BinanceSocketManager
from dataclasses import dataclass

# ------------------- Config -------------------
@dataclass
class BotConfig:
    pair: str = "AVAXUSDT"
    interval: str = "1m"
    initial_balance: float = 100.0
    risk_per_trade: float = 0.01
    tp_pct: float = 0.001   # 0.1%
    sl_pct: float = 0.002   # 0.2%
    leverage: float = 3
    slippage: float = 0.0005
    fee_taker: float = 0.0004
    use_taker: bool = True

cfg = BotConfig()

# ------------------- Helper -------------------
def signal_mean_reversion(df, bb_period=20, bb_dev=2.0, rsi_period=14):
    """Simple mean-reversion dual entry signals: 1=long, -1=short, 0=none"""
    sma = df['close'].rolling(bb_period).mean()
    std = df['close'].rolling(bb_period).std()
    upper = sma + bb_dev*std
    lower = sma - bb_dev*std

    delta = df['close'].diff()
    up = delta.clip(lower=0).rolling(rsi_period).mean()
    down = -delta.clip(upper=0).rolling(rsi_period).mean()
    rsi = 100 - (100 / (1 + up/down.replace(0, np.nan)))

    sig = pd.Series(0, index=df.index)
    long_cond = (df['close'] < lower) & (rsi < 40)
    short_cond = (df['close'] > upper) & (rsi > 60)
    sig[long_cond] = 1
    sig[short_cond] = -1
    return sig.fillna(0)

# ------------------- Live Bot -------------------
class LiveBot:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.balance = cfg.initial_balance
        self.position = 0
        self.entry_price = None
        self.position_size = 0.0
        self.trades = []

    async def start(self):
        client = await AsyncClient.create()
        bm = BinanceSocketManager(client)
        ts = bm.kline_socket(self.cfg.pair, interval=self.cfg.interval)
        print(f"[INFO] Starting LiveBot for {self.cfg.pair} on {self.cfg.interval} timeframe...")
        async with ts as tscm:
            while True:
                res = await tscm.recv()
                k = res['k']
                ohlc = {
                    'open': float(k['o']),
                    'high': float(k['h']),
                    'low': float(k['l']),
                    'close': float(k['c']),
                    'volume': float(k['v']),
                    'time': pd.to_datetime(k['t'], unit='ms', utc=True)
                }
                df = pd.DataFrame([ohlc]).set_index('time')
                await self.check_signal(df)

    async def check_signal(self, df):
        sig = signal_mean_reversion(df).iloc[-1]
        price = df['close'].iloc[-1]
        fee_rate = self.cfg.fee_taker if self.cfg.use_taker else 0

        # Entry logic
        if self.position == 0 and sig != 0:
            size = self.balance * self.cfg.risk_per_trade * self.cfg.leverage
            self.position = sig
            self.entry_price = price * (1 + self.cfg.slippage if sig==1 else 1 - self.cfg.slippage)
            self.position_size = size
            fee = size * fee_rate
            self.balance -= fee
            self.trades.append({
                'time': df.index[-1],
                'side': 'LONG' if sig==1 else 'SHORT',
                'entry': self.entry_price,
                'size': size,
                'fee': fee
            })
            print(f"[ENTRY] {self.trades[-1]} | Balance: {self.balance:.4f}")

        # Exit logic
        elif self.position != 0:
            tp = self.entry_price * (1 + self.cfg.tp_pct if self.position==1 else 1 - self.cfg.tp_pct)
            sl = self.entry_price * (1 - self.cfg.sl_pct if self.position==1 else 1 + self.cfg.sl_pct)
            exit_reason = None
            exit_price = price
            if self.position == 1:
                if price >= tp:
                    exit_reason = 'TP'
                    exit_price = tp
                elif price <= sl:
                    exit_reason = 'SL'
                    exit_price = sl
            else:
                if price <= tp:
                    exit_reason = 'TP'
                    exit_price = tp
                elif price >= sl:
                    exit_reason = 'SL'
                    exit_price = sl
            if exit_reason:
                pnl = (exit_price - self.entry_price)/self.entry_price * self.position_size * (1 if self.position==1 else -1)
                fee = self.position_size * fee_rate
                self.balance += pnl - fee
                self.trades[-1].update({'exit_time': df.index[-1], 'exit_price': exit_price, 'pnl': pnl, 'exit_reason': exit_reason, 'balance_after': self.balance})
                print(f"[EXIT] {self.trades[-1]} | Balance: {self.balance:.4f}")
                self.position = 0
                self.entry_price = None
                self.position_size = 0.0

# ------------------- Main -------------------
if __name__ == "__main__":
    bot = LiveBot(cfg)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(bot.start())
