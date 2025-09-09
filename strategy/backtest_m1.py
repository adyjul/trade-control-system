# backtest_m1.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class BacktestConfig:
    initial_balance: float = 1000.0
    fee_taker: float = 0.0004   # e.g. 0.04% = 0.0004 (futures taker)
    fee_maker: float = 0.0002
    use_taker: bool = True
    slippage: float = 0.0005     # 0.05% slippage per trade (price impact)
    risk_per_trade: float = 0.01 # fraction of equity risked per trade
    leverage: float = 1.0
    min_size: float = 1e-6       # minimal position notional constraint

def load_ohlcv(csv_path: str) -> pd.DataFrame:
    """
    CSV columns: timestamp (ms or ISO), open, high, low, close, volume
    index must be datetime-like. Returns dataframe with datetime index.
    """
    df = pd.read_excel(csv_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='ignore')
        df = df.set_index('timestamp')
    else:
        df.index = pd.to_datetime(df.iloc[:,0])
        df.index.name = 'timestamp'
        df = df.iloc[:,1:]
    df = df.sort_index()
    return df[['open','high','low','close','volume']].astype(float)

# --- Example signal generators ---
def signal_mean_reversion(df: pd.DataFrame, bb_period=20, bb_dev=2.0, rsi_period=14) -> pd.Series:
    """
    Return signals: 1 = long, -1 = short, 0 = no signal
    Simple: long when close < lower bollinger & rsi < 40; short when close > upper bollinger & rsi > 60
    """
    # SMA
    sma = df['close'].rolling(bb_period).mean()
    std = df['close'].rolling(bb_period).std()
    upper = sma + bb_dev*std
    lower = sma - bb_dev*std

    # simple RSI
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

def signal_straddle_simple(df: pd.DataFrame, lookback=10, range_threshold=0.0008) -> pd.DataFrame:
    """
    Straddle idea: detect narrow range (consolidation). Returns a frame with breakout levels.
    - 'enter' column: 1 if narrow-range detected at that bar (we'll place long and short orders)
    - 'long_level' and 'short_level': price thresholds to consider as breakout
    """
    high_lb = df['high'].rolling(lookback).max()
    low_lb = df['low'].rolling(lookback).min()
    rng = (high_lb - low_lb) / df['close']
    enter = rng < range_threshold
    long_level = high_lb
    short_level = low_lb
    out = pd.DataFrame({'enter': enter, 'long_level': long_level, 'short_level': short_level}, index=df.index)
    return out

# --- Execution engine ---
def run_backtest(df: pd.DataFrame,
                 signals: pd.Series,
                 config: BacktestConfig,
                 tp_pct: float = 0.001,   # 0.1% TP default
                 sl_pct: float = 0.002    # 0.2% SL default
                 ) -> Tuple[pd.DataFrame, dict]:
    """
    signals: series of 1/-1/0 each bar meaning we open a trade at next-open (discrete-time backtest)
    Returns trades dataframe and summary metrics
    """
    balance = config.initial_balance
    equity_curve = []
    trades = []
    position = 0
    entry_price = None
    entry_index = None
    position_size_notional = 0.0

    # compute fee per side function
    fee_rate = config.fee_taker if config.use_taker else config.fee_maker

    for i in range(len(df)-1):  # we'll enter at next bar open to avoid lookahead
        ts = df.index[i]
        next_ts = df.index[i+1]
        price_next_open = df['open'].iat[i+1]
        # If no position, check signal to open
        if position == 0:
            sig = signals.iloc[i]
            if sig != 0:
                # position sizing: risk-based sizing using SL distance
                sl_price = price_next_open * (1 - sl_pct) if sig==1 else price_next_open * (1 + sl_pct)
                # per-trade risk notional = balance * risk_per_trade
                risk_notional = balance * config.risk_per_trade
                sl_distance = abs(price_next_open - sl_price) / price_next_open
                if sl_distance == 0:
                    continue
                size_notional = (risk_notional / sl_distance)  # notional
                # apply leverage
                size_notional = size_notional * config.leverage
                if size_notional < config.min_size:
                    continue
                position = sig
                entry_price = price_next_open * (1 + config.slippage if sig==1 else 1 - config.slippage)
                position_size_notional = size_notional
                entry_index = next_ts
                # pay taker fee on entry
                fee_entry = position_size_notional * fee_rate
                balance -= fee_entry  # fees reduce cash
                trades.append({
                    'entry_time': entry_index,
                    'side': 'LONG' if position==1 else 'SHORT',
                    'entry_price': entry_price,
                    'size_notional': position_size_notional,
                    'fee_entry': fee_entry
                })
        else:
            # have a position: check exit by TP/SL intrabar using high/low of current bar (i)
            # We simulate exits during same bar using that bar's high/low.
            high = df['high'].iat[i]
            low = df['low'].iat[i]
            exit_reason = None
            exit_price = None
            # For LONG: TP when high >= entry*(1+tp), SL when low <= entry*(1-sl)
            if position == 1:
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)
                if low <= sl_price:
                    exit_reason = 'SL'
                    exit_price = sl_price * (1 + config.slippage)  # slippage unfavorable
                elif high >= tp_price:
                    exit_reason = 'TP'
                    exit_price = tp_price * (1 - config.slippage)  # favorable slippage
            else:
                # SHORT: TP when low <= entry*(1-tp), SL when high >= entry*(1+sl)
                tp_price = entry_price * (1 - tp_pct)
                sl_price = entry_price * (1 + sl_pct)
                if high >= sl_price:
                    exit_reason = 'SL'
                    exit_price = sl_price * (1 - config.slippage)
                elif low <= tp_price:
                    exit_reason = 'TP'
                    exit_price = tp_price * (1 + config.slippage)

            if exit_reason is not None:
                # compute P&L: for notional-based: pnl = (exit_price - entry_price)/entry_price * position_size_notional * sign
                price_return = (exit_price - entry_price) / entry_price
                pnl = price_return * position_size_notional * (1 if position==1 else -1)
                fee_exit = position_size_notional * fee_rate
                balance += pnl
                balance -= fee_exit
                # update last trade record
                trades[-1].update({
                    'exit_time': df.index[i],
                    'exit_price': exit_price,
                    'fee_exit': fee_exit,
                    'pnl': pnl,
                    'exit_reason': exit_reason,
                    'balance_after': balance
                })
                # reset pos
                position = 0
                entry_price = None
                position_size_notional = 0.0
                entry_index = None

        equity_curve.append({'time': df.index[i], 'balance': balance})

    equity_df = pd.DataFrame(equity_curve).set_index('time')
    trades_df = pd.DataFrame(trades)
    # summary metrics
    total_trades = len(trades_df)
    wins = trades_df[trades_df['pnl']>0]
    losses = trades_df[trades_df['pnl']<=0]
    net_profit = balance - config.initial_balance
    winrate = len(wins) / total_trades if total_trades>0 else np.nan
    avg_win = wins['pnl'].mean() if len(wins)>0 else 0
    avg_loss = losses['pnl'].mean() if len(losses)>0 else 0
    max_dd = compute_max_drawdown(equity_df['balance'])
    summary = {
        'initial_balance': config.initial_balance,
        'final_balance': balance,
        'net_profit': net_profit,
        'total_trades': total_trades,
        'winrate': winrate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_drawdown': max_dd
    }
    return trades_df, equity_df, summary

def compute_max_drawdown(equity_series: pd.Series):
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    max_dd = drawdown.min()
    return max_dd

# --- Example usage ---
if __name__ == "__main__":
    # 1) load 1m OHLCV CSV
    df = load_ohlcv("/root/trade-control-system/backtest_result_data/TIAUSDT_1m_all_signals.xlsx")  # replace with your file

    # 2) pick signals
    # signals = signal_mean_reversion(df)         # single-side signals
    # OR for straddle approach, we can generate signals when narrow range detected:
    straddle = signal_straddle_simple(df, lookback=8, range_threshold=0.0006)
    # For a simple straddle backtest, we'll convert "enter" to a pseudo-signal: 1 means open both sides -> we'll simulate opening LONG when price breaks above long_level, SHORT when breaks below short_level
    signals = pd.Series(0, index=df.index)

    # Simple straddle logic: when 'enter' == True at i, mark next bar to monitor breakout for next N bars
    monitor_window = 3
    for i in range(len(df)-monitor_window):
        if straddle['enter'].iat[i]:
            high_level = straddle['long_level'].iat[i]
            low_level = straddle['short_level'].iat[i]
            # monitor next few bars
            for j in range(1, monitor_window+1):
                idx = i+j
                if df['high'].iat[idx] >= high_level:
                    signals.iat[idx] = 1
                    break
                if df['low'].iat[idx] <= low_level:
                    signals.iat[idx] = -1
                    break

    # 3) run backtest
    cfg = BacktestConfig(initial_balance=100.0, fee_taker=0.0004, slippage=0.0006, risk_per_trade=0.01, leverage=3.0)
    trades_df, equity_df, summary = run_backtest(df, signals, cfg, tp_pct=0.001, sl_pct=0.0015)

    print("Summary:", summary)
    trades_df.to_csv("trades_result.csv", index=False)
    equity_df.to_csv("equity_curve.csv")
