import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class BacktestConfig:
    initial_balance: float = 100.0
    risk_per_trade: float = 0.01  # 1% risk per trade
    leverage: float = 3.0
    fee_rate: float = 0.0004
    min_atr: float = 0.0005
    tp_atr_mult: float = 0.8
    sl_atr_mult: float = 0.8
    max_hold_bars: int = 5

def load_ohlcv(path):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp')
    df = df[['open','high','low','close','volume']].astype(float)
    return df

def compute_atr(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def dual_entry_backtest(df, cfg: BacktestConfig):
    balance = cfg.initial_balance
    trades = []
    equity_curve = []
    atr = compute_atr(df)

    open_position = None  # hanya 1 posisi terbuka

    for i in range(1, len(df)):
        current_time = df.index[i]
        high = df['high'].iat[i]
        low = df['low'].iat[i]
        close = df['close'].iat[i]
        atr_val = atr.iat[i]

        # update posisi terbuka
        if open_position:
            entry_time, entry_price, side, size, sl, tp, hold_bars = open_position
            exit_reason = None
            exit_price = None

            if side == 'LONG':
                if low <= sl:
                    exit_reason, exit_price = 'SL', sl
                elif high >= tp:
                    exit_reason, exit_price = 'TP', tp
            elif side == 'SHORT':
                if high >= sl:
                    exit_reason, exit_price = 'SL', sl
                elif low <= tp:
                    exit_reason, exit_price = 'TP', tp

            # time-based exit
            hold_bars += 1
            if not exit_reason and hold_bars >= cfg.max_hold_bars:
                exit_reason, exit_price = 'TIME', close

            if exit_reason:
                # hitung PnL
                if side == 'LONG':
                    pnl = size * (exit_price - entry_price) / entry_price
                else:
                    pnl = size * (entry_price - exit_price) / entry_price
                pnl -= size * cfg.fee_rate
                balance += pnl

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'side': side,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': exit_reason
                })
                open_position = None
            else:
                # update hold bars
                open_position = (entry_time, entry_price, side, size, sl, tp, hold_bars)

        # entry baru
        if open_position is None and not np.isnan(atr_val) and atr_val >= cfg.min_atr:
            long_level = close + atr_val * 0.2
            short_level = close - atr_val * 0.2
            size = balance * cfg.risk_per_trade * cfg.leverage

            if high >= long_level:
                entry_price = close  # langsung entry saat trigger
                sl = entry_price - atr_val * cfg.sl_atr_mult
                tp = entry_price + atr_val * cfg.tp_atr_mult
                open_position = (current_time, entry_price, 'LONG', size, sl, tp, 0)

            elif low <= short_level:
                entry_price = close
                sl = entry_price + atr_val * cfg.sl_atr_mult
                tp = entry_price - atr_val * cfg.tp_atr_mult
                open_position = (current_time, entry_price, 'SHORT', size, sl, tp, 0)

        equity_curve.append({'time': current_time, 'balance': balance})

    # tutup posisi terakhir
    if open_position:
        entry_time, entry_price, side, size, sl, tp, hold_bars = open_position
        exit_price = df['close'].iat[-1]
        if side == 'LONG':
            pnl = size * (exit_price - entry_price) / entry_price
        else:
            pnl = size * (entry_price - exit_price) / entry_price
        pnl -= size * cfg.fee_rate
        balance += pnl
        trades.append({
            'entry_time': entry_time,
            'exit_time': df.index[-1],
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': 'END'
        })

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve).set_index('time')

    summary = {
        'initial_balance': cfg.initial_balance,
        'final_balance': balance,
        'net_profit': balance - cfg.initial_balance,
        'total_trades': len(trades_df),
        'winrate': (trades_df['pnl'] > 0).mean() if len(trades_df) > 0 else 0,
        'max_drawdown': compute_max_drawdown(equity_df['balance'])
    }
    return trades_df, equity_df, summary

def compute_max_drawdown(equity_series):
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    return drawdown.min()

if __name__ == "__main__":
    cfg = BacktestConfig()
    df = load_ohlcv("/root/trade-control-system/backtest_by_data/INJUSDT_1m.csv")
    trades_df, equity_df, summary = dual_entry_backtest(df, cfg)
    print("Summary:", summary)
    trades_df.to_csv("trades_dual_entry_90.csv", index=False)
    equity_df.to_csv("equity_dual_entry_90.csv")
