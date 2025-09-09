# backtest_dual_entry_m1.py
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class BacktestConfig:
    initial_balance: float = 1000.0
    fee_taker: float = 0.0004
    slippage: float = 0.0005
    risk_per_trade: float = 0.01
    leverage: float = 3.0
    monitor_candles: int = 3

def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp')
    df = df[['open','high','low','close','volume']].astype(float)
    return df

def signal_dual_entry(df, lookback=10, range_threshold=0.0008, atr_period=14):
    """
    Dual-entry straddle signal:
    - enter = True if range sempit
    - long_level = breakout atas
    - short_level = breakout bawah
    """
    high_lb = df['high'].rolling(lookback).max()
    low_lb = df['low'].rolling(lookback).min()
    rng = (high_lb - low_lb) / df['close']
    enter = rng < range_threshold

    # ATR
    tr = pd.concat([df['high'] - df['low'],
                    (df['high'] - df['close'].shift(1)).abs(),
                    (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    long_level = high_lb + atr
    short_level = low_lb - atr

    return pd.DataFrame({'enter': enter, 'long_level': long_level, 'short_level': short_level, 'atr': atr}, index=df.index)

def run_backtest_dual(df, signals, cfg, tp_mult=1.0, sl_mult=1.5):
    balance = cfg.initial_balance
    trades = []
    equity_curve = []

    for i in range(len(df)-cfg.monitor_candles):
        if not signals['enter'].iat[i]:
            equity_curve.append({'time': df.index[i], 'balance': balance})
            continue

        long_level = signals['long_level'].iat[i]
        short_level = signals['short_level'].iat[i]
        atr_now = signals['atr'].iat[i]

        # monitor next N candles
        for j in range(1, cfg.monitor_candles+1):
            idx = i+j
            high = df['high'].iat[idx]
            low = df['low'].iat[idx]

            # LONG triggered
            if high >= long_level:
                entry_price = long_level * (1 + cfg.slippage)
                tp_price = entry_price + atr_now * tp_mult
                sl_price = entry_price - atr_now * sl_mult
                exit_reason, exit_price = None, None

                if low <= sl_price:
                    exit_reason, exit_price = 'SL', sl_price
                elif high >= tp_price:
                    exit_reason, exit_price = 'TP', tp_price

                pnl = (exit_price - entry_price)/entry_price*balance if exit_reason else 0
                fee = balance * cfg.fee_taker
                balance += pnl - fee
                trades.append((df.index[idx], 'LONG', entry_price, exit_price, pnl, exit_reason))
                break  # cancel SHORT

            # SHORT triggered
            elif low <= short_level:
                entry_price = short_level * (1 - cfg.slippage)
                tp_price = entry_price - atr_now * tp_mult
                sl_price = entry_price + atr_now * sl_mult
                exit_reason, exit_price = None, None

                if high >= sl_price:
                    exit_reason, exit_price = 'SL', sl_price
                elif low <= tp_price:
                    exit_reason, exit_price = 'TP', tp_price

                pnl = (entry_price - exit_price)/entry_price*balance if exit_reason else 0
                fee = balance * cfg.fee_taker
                balance += pnl - fee
                trades.append((df.index[idx], 'SHORT', entry_price, exit_price, pnl, exit_reason))
                break  # cancel LONG

        equity_curve.append({'time': df.index[i], 'balance': balance})

    trades_df = pd.DataFrame(trades, columns=['time','side','entry','exit','pnl','exit_reason'])
    equity_df = pd.DataFrame(equity_curve).set_index('time')

    summary = {
        'initial_balance': cfg.initial_balance,
        'final_balance': balance,
        'net_profit': balance - cfg.initial_balance,
        'total_trades': len(trades_df),
        'winrate': (trades_df['pnl']>0).mean() if len(trades_df)>0 else np.nan,
        'max_drawdown': compute_max_drawdown(equity_df['balance']) if not equity_df.empty else 0
    }

    return trades_df, equity_df, summary

def compute_max_drawdown(series):
    roll_max = series.cummax()
    drawdown = (series - roll_max)/roll_max
    return drawdown.min()

# --- Example usage ---
if __name__ == "__main__":
    df = load_ohlcv("/root/trade-control-system/backtest_by_data/TIAUSDT_1m.csv")
    signals = signal_dual_entry(df, lookback=8, range_threshold=0.002, atr_period=14)
    cfg = BacktestConfig(initial_balance=100.0, fee_taker=0.0004, slippage=0.0005, risk_per_trade=0.01, leverage=3.0, monitor_candles=3)
    trades_df, equity_df, summary = run_backtest_dual(df, signals, cfg, tp_mult=1.0, sl_mult=1.5)
    print("Summary:", summary)
    trades_df.to_csv("trades_dual_entry.csv", index=False)
    equity_df.to_csv("equity_dual_entry.csv")
