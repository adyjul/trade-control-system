import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class BacktestConfig:
    initial_balance: float = 100.0
    risk_per_trade: float = 0.01
    leverage: float = 3.0
    fee_rate: float = 0.0004
    min_atr: float = 0.0005  # filter market flat
    tp_atr_mult: float = 0.8
    sl_atr_mult: float = 0.8

def load_ohlcv(path):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp')
    df = df[['open','high','low','close','volume']].astype(float)
    return df

def compute_atr(df, period=14):
    tr = pd.concat([df['high'] - df['low'],
                    (df['high'] - df['close'].shift(1)).abs(),
                    (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def dual_entry_backtest(df, cfg: BacktestConfig):
    balance = cfg.initial_balance
    trades = []
    equity_curve = []
    atr = compute_atr(df)

    for i in range(len(df)-1):
        atr_now = atr.iat[i]
        if np.isnan(atr_now) or atr_now < cfg.min_atr:
            equity_curve.append({'time': df.index[i], 'balance': balance})
            continue

        long_level = df['close'].iat[i] + atr_now * 0.2
        short_level = df['close'].iat[i] - atr_now * 0.2

        # cek candle berikut untuk kena TP/SL
        for j in range(1,4):
            if i+j >= len(df):
                break
            high = df['high'].iat[i+j]
            low = df['low'].iat[i+j]
            # LONG
            if high >= long_level:
                entry_price = df['close'].iat[i]
                tp = entry_price + atr_now*cfg.tp_atr_mult
                sl = entry_price - atr_now*cfg.sl_atr_mult
                exit_price = min(tp, high) if low > sl else sl
                pnl = (exit_price - entry_price)/entry_price * balance * cfg.leverage - cfg.fee_rate*balance
                balance += pnl
                trades.append({'time': df.index[i+j], 'side':'LONG', 'pnl':pnl})
                break
            # SHORT
            if low <= short_level:
                entry_price = df['close'].iat[i]
                tp = entry_price - atr_now*cfg.tp_atr_mult
                sl = entry_price + atr_now*cfg.sl_atr_mult
                exit_price = max(tp, low) if high < sl else sl
                pnl = (entry_price - exit_price)/entry_price * balance * cfg.leverage - cfg.fee_rate*balance
                balance += pnl
                trades.append({'time': df.index[i+j], 'side':'SHORT', 'pnl':pnl})
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
    drawdown = (equity_series - roll_max)/roll_max
    return drawdown.min()

if __name__ == "__main__":
    cfg = BacktestConfig()
    df = load_ohlcv("/root/trade-control-system/backtest_by_data/AVAXUSDT_1m.csv")
    trades_df, equity_df, summary = dual_entry_backtest(df, cfg)
    print("Summary:", summary)
    trades_df.to_csv("trades_dual_entry.csv", index=False)
    equity_df.to_csv("equity_dual_entry.csv")
