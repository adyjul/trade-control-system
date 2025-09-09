# backtest_m1_dual.py
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class BacktestConfig:
    initial_balance: float = 100.0
    fee_taker: float = 0.0004
    slippage: float = 0.0005
    risk_per_trade: float = 0.01
    leverage: float = 3.0
    min_size: float = 1e-6

def load_ohlcv(path):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp')
    return df[['open','high','low','close','volume']].astype(float)

def compute_atr(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def run_backtest_dual(df, cfg, atr_period=14, tp_mult=0.3, sl_mult=0.3):
    balance = cfg.initial_balance
    trades = []
    equity_curve = []

    atr = compute_atr(df, atr_period)

    for i in range(len(df)-1):
        price = df['close'].iat[i]
        atr_now = atr.iat[i] if not np.isnan(atr.iat[i]) else 0.0001

        # levels
        long_entry = price + atr_now*0.1
        short_entry = price - atr_now*0.1
        tp_long = long_entry + atr_now*tp_mult
        sl_long = long_entry - atr_now*sl_mult
        tp_short = short_entry - atr_now*tp_mult
        sl_short = short_entry + atr_now*sl_mult

        # monitor next 3 candles
        for j in range(1,4):
            idx = i+j
            if idx >= len(df):
                break
            high = df['high'].iat[idx]
            low = df['low'].iat[idx]

            # LONG
            long_done = False
            if high >= long_entry:
                exit_price = tp_long if high >= tp_long else sl_long
                pnl = (exit_price - long_entry)/long_entry * balance
                balance += pnl - balance*cfg.fee_taker
                trades.append({'time': df.index[idx], 'side':'LONG', 'entry':long_entry, 'exit':exit_price, 'pnl':pnl})
                long_done = True

            # SHORT
            short_done = False
            if low <= short_entry:
                exit_price = tp_short if low <= tp_short else sl_short
                pnl = (short_entry - exit_price)/short_entry * balance
                balance += pnl - balance*cfg.fee_taker
                trades.append({'time': df.index[idx], 'side':'SHORT', 'entry':short_entry, 'exit':exit_price, 'pnl':pnl})
                short_done = True

            if long_done and short_done:
                break

        equity_curve.append({'time': df.index[i], 'balance': balance})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve).set_index('time')

    summary = {
        'initial_balance': cfg.initial_balance,
        'final_balance': balance,
        'net_profit': balance - cfg.initial_balance,
        'total_trades': len(trades_df),
        'winrate': (trades_df['pnl']>0).mean() if len(trades_df)>0 else np.nan,
        'max_drawdown': compute_max_drawdown(equity_df['balance'])
    }

    return trades_df, equity_df, summary

def compute_max_drawdown(equity_series):
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    return drawdown.min()

if __name__ == "__main__":
    df = load_ohlcv("/root/trade-control-system/backtest_by_data/TIAUSDT_1m.csv")
    cfg = BacktestConfig()
    trades_df, equity_df, summary = run_backtest_dual(df, cfg)
    print("Summary:", summary)
    trades_df.to_csv("trades_dual.csv", index=False)
    equity_df.to_csv("equity_dual.csv")
