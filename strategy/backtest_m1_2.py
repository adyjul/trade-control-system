# strategy/backtest_m1_dual.py
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
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.set_index('timestamp')
    df = df[['open','high','low','close','volume']].astype(float)
    return df

def signal_dual_entry(df, window=20, dev=0.5):
    sma = df['close'].rolling(window).mean()
    upper = sma + dev * df['close'].rolling(window).std()
    lower = sma - dev * df['close'].rolling(window).std()
    sig = pd.Series(0, index=df.index)
    sig[df['close'] < lower] = 1   # long
    sig[df['close'] > upper] = -1  # short
    return sig

def run_backtest(df, signals, cfg: BacktestConfig, tp_pct=0.001, sl_pct=0.001):
    balance = cfg.initial_balance
    equity_curve = []
    trades = []
    position = 0
    entry_price = None
    position_size_notional = 0

    for i in range(len(df)-1):
        sig = signals.iloc[i]
        next_open = df['open'].iat[i+1]

        # entry logic
        if position == 0 and sig != 0:
            sl_price = next_open * (1 - sl_pct) if sig==1 else next_open*(1 + sl_pct)
            size_notional = balance * cfg.risk_per_trade / abs(next_open - sl_price) * cfg.leverage
            if size_notional < cfg.min_size:
                continue
            position = sig
            entry_price = next_open * (1 + cfg.slippage if sig==1 else 1 - cfg.slippage)
            position_size_notional = size_notional
            trades.append({'entry_time': df.index[i+1], 'side':'LONG' if sig==1 else 'SHORT', 'entry_price':entry_price})

        # exit logic
        elif position != 0:
            high = df['high'].iat[i]
            low = df['low'].iat[i]
            exit_price = None
            exit_reason = None
            if position == 1:
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)
                if high >= tp_price:
                    exit_price = tp_price * (1 - cfg.slippage)
                    exit_reason = 'TP'
                elif low <= sl_price:
                    exit_price = sl_price * (1 + cfg.slippage)
                    exit_reason = 'SL'
            else:
                tp_price = entry_price * (1 - tp_pct)
                sl_price = entry_price * (1 + sl_pct)
                if low <= tp_price:
                    exit_price = tp_price * (1 + cfg.slippage)
                    exit_reason = 'TP'
                elif high >= sl_price:
                    exit_price = sl_price * (1 - cfg.slippage)
                    exit_reason = 'SL'

            if exit_reason:
                pnl = (exit_price - entry_price)/entry_price*position_size_notional*(1 if position==1 else -1)
                balance += pnl
                trades[-1].update({'exit_time':df.index[i],'exit_price':exit_price,'pnl':pnl,'exit_reason':exit_reason,'balance_after':balance})
                position = 0
                entry_price = None
                position_size_notional = 0

        equity_curve.append({'time': df.index[i], 'balance': balance})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve).set_index('time')
    summary = {
        'initial_balance': cfg.initial_balance,
        'final_balance': balance,
        'net_profit': balance - cfg.initial_balance,
        'total_trades': len(trades_df),
        'winrate': (trades_df['pnl']>0).mean() if not trades_df.empty else np.nan,
        'avg_win': trades_df[trades_df['pnl']>0]['pnl'].mean() if not trades_df.empty else 0,
        'avg_loss': trades_df[trades_df['pnl']<=0]['pnl'].mean() if not trades_df.empty else 0,
        'max_drawdown': (equity_df['balance']/equity_df['balance'].cummax()-1).min()
    }
    return trades_df, equity_df, summary

if __name__ == "__main__":
    df = load_ohlcv("/root/trade-control-system/backtest_by_data/TIAUSDT_1m.csv")
    signals = signal_dual_entry(df)
    cfg = BacktestConfig()
    trades_df, equity_df, summary = run_backtest(df, signals, cfg)
    print("Summary:", summary)
    trades_df.to_csv("trades_dual.csv", index=False)
    equity_df.to_csv("equity_dual.csv")
