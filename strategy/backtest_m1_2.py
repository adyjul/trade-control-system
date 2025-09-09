# backtest_m1_dual.py
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class BacktestConfig:
    initial_balance: float = 100.0
    fee_taker: float = 0.0004
    fee_maker: float = 0.0002
    use_taker: bool = True
    slippage: float = 0.0005
    risk_per_trade: float = 0.01
    leverage: float = 1.0
    min_size: float = 1e-6

def load_ohlcv(path: str):
    df = pd.read_csv(path)
    
    # Cek apakah timestamp numeric atau string
    if df['timestamp'].dtype == object:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)  # string → datetime UTC
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)  # ms → datetime UTC

    df = df.set_index('timestamp')
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    return df

def compute_atr(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1)
    atr = tr.max(axis=1).rolling(period).mean()
    return atr

def dual_entry_signals(df, atr_period=14, atr_multiplier=0.5):
    atr = compute_atr(df, atr_period)
    ema = df['close'].ewm(span=30).mean()
    signals = pd.Series(0, index=df.index)

    for i in range(len(df)-1):
        # hanya entry jika ATR cukup besar
        if atr.iat[i] < 0.0005:
            continue
        price = df['close'].iat[i]
        # Dual-entry: long di atas EMA, short di bawah EMA
        if price > ema.iat[i]:
            signals.iat[i] = 1   # long
        elif price < ema.iat[i]:
            signals.iat[i] = -1  # short
    return signals

def run_backtest(df, signals, cfg, tp_atr_mult=0.5, sl_atr_mult=0.5):
    balance = cfg.initial_balance
    trades = []
    position = 0
    entry_price = 0
    fee_rate = cfg.fee_taker if cfg.use_taker else cfg.fee_maker

    for i in range(len(df)-1):
        sig = signals.iat[i]
        if position == 0 and sig != 0:
            position = sig
            entry_price = df['open'].iat[i+1] * (1 + cfg.slippage if sig==1 else 1-cfg.slippage)
            atr_now = compute_atr(df).iat[i]
            tp = entry_price + atr_now*tp_atr_mult if sig==1 else entry_price - atr_now*tp_atr_mult
            sl = entry_price - atr_now*sl_atr_mult if sig==1 else entry_price + atr_now*sl_atr_mult
            size = balance*cfg.risk_per_trade / (abs(entry_price-sl)) * cfg.leverage
            trades.append({'time': df.index[i], 'side': 'LONG' if sig==1 else 'SHORT',
                           'entry': entry_price, 'tp': tp, 'sl': sl, 'size': size})
        elif position != 0:
            high = df['high'].iat[i]
            low = df['low'].iat[i]
            exit_price, exit_reason = None, None
            if position == 1:
                if low <= trades[-1]['sl']:
                    exit_price = trades[-1]['sl']
                    exit_reason = 'SL'
                elif high >= trades[-1]['tp']:
                    exit_price = trades[-1]['tp']
                    exit_reason = 'TP'
            else:
                if high >= trades[-1]['sl']:
                    exit_price = trades[-1]['sl']
                    exit_reason = 'SL'
                elif low <= trades[-1]['tp']:
                    exit_price = trades[-1]['tp']
                    exit_reason = 'TP'

            if exit_price is not None:
                pnl = (exit_price - entry_price)/entry_price * trades[-1]['size'] * (1 if position==1 else -1)
                fee = trades[-1]['size']*fee_rate
                balance += pnl - fee
                trades[-1].update({'exit': exit_price, 'pnl': pnl, 'exit_reason': exit_reason, 'balance_after': balance})
                position = 0

    trades_df = pd.DataFrame(trades)
    final_balance = balance
    summary = {
        'initial_balance': cfg.initial_balance,
        'final_balance': final_balance,
        'net_profit': final_balance-cfg.initial_balance,
        'total_trades': len(trades_df),
        'winrate': (trades_df['pnl']>0).mean() if len(trades_df)>0 else np.nan,
        'max_drawdown': np.min(trades_df['balance_after']/cfg.initial_balance-1) if len(trades_df)>0 else 0
    }
    return trades_df, summary

if __name__=="__main__":
    cfg = BacktestConfig(initial_balance=100.0, risk_per_trade=0.01, leverage=3.0)
    df = load_ohlcv("/root/trade-control-system/backtest_by_data/TIAUSDT_1m.csv")
    signals = dual_entry_signals(df)
    trades_df, summary = run_backtest(df, signals, cfg)
    print("Summary:", summary)
    trades_df.to_csv("trades_dual_entry.csv", index=False)
