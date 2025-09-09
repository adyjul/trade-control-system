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

def load_ohlcv(path: str) -> pd.DataFrame:
    """
    Load OHLCV dari CSV/XLSX. Auto-detect timestamp column (ms or ISO) dan konversi ke UTC index.
    Mengembalikan df dengan kolom: open, high, low, close, volume (float) dan datetime index tz-aware UTC.
    """
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".xls") or path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        raise ValueError("File must be .csv or .xls/.xlsx")

    # Standarisasi kolom names -> lowercase keys
    cols_map = {c.lower(): c for c in df.columns}
    # possible timestamp column names
    ts_candidates = ['timestamp', 'open_time', 'time', 'date', 'datetime']
    ts_col = None
    for cand in ts_candidates:
        if cand in cols_map:
            ts_col = cols_map[cand]
            break
    if ts_col is None:
        # fallback: assume first column is timestamp
        ts_col = df.columns[0]

    # try numeric (ms) first
    try:
        # check if can be converted to numeric w/o producing many NaN
        num = pd.to_numeric(df[ts_col], errors='coerce')
        non_na_ratio = num.notna().mean()
        if non_na_ratio > 0.9:
            df[ts_col] = pd.to_datetime(num.astype('Int64'), unit='ms', utc=True)
        else:
            # parse as string datetime
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
    except Exception:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors='coerce')

    df = df.set_index(ts_col)
    # rename columns to expected names if variants exist
    rename_map = {}
    for want in ['open','high','low','close','volume']:
        for c in df.columns:
            if c.lower() == want:
                rename_map[c] = want
                break
    df = df.rename(columns=rename_map)

    # keep only required cols
    for c in ['open','high','low','close','volume']:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in input file")

    df = df.sort_index()
    # ensure timezone-aware UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    # cast types
    df = df[['open','high','low','close','volume']].astype(float)

    # optional: reindex contiguous 1-minute (uncomment if you want fill missing)
    # idx = pd.date_range(start=df.index[0].ceil('T'), end=df.index[-1].floor('T'), freq='1T', tz='UTC')
    # df = df.reindex(idx)
    # df[['open','high','low','close']] = df[['open','high','low','close']].ffill()
    # df['volume'] = df['volume'].fillna(0.0)

    return df

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

def run_backtest_straddle(df, straddle, cfg, tp_pct=0.001, sl_pct=0.0015, monitor_window=3):
    balance = cfg.initial_balance
    equity_curve = []
    trades = []

    for i in range(len(df) - monitor_window):
        if straddle['enter'].iat[i]:
            high_level = straddle['long_level'].iat[i]
            low_level = straddle['short_level'].iat[i]

            # pasang 2 order pending
            for j in range(1, monitor_window+1):
                idx = i + j
                high = df['high'].iat[idx]
                low = df['low'].iat[idx]

                if high >= high_level:
                    # LONG aktif, cancel SHORT
                    entry_price = high_level
                    tp_price = entry_price * (1 + tp_pct)
                    sl_price = entry_price * (1 - sl_pct)
                    exit_reason, exit_price = None, None
                    if df['low'].iat[idx] <= sl_price:
                        exit_reason, exit_price = 'SL', sl_price
                    elif df['high'].iat[idx] >= tp_price:
                        exit_reason, exit_price = 'TP', tp_price
                    pnl = (exit_price - entry_price) / entry_price * balance if exit_reason else 0
                    balance += pnl
                    trades.append((df.index[idx], "LONG", entry_price, exit_price, pnl, exit_reason))
                    break

                elif low <= low_level:
                    # SHORT aktif, cancel LONG
                    entry_price = low_level
                    tp_price = entry_price * (1 - tp_pct)
                    sl_price = entry_price * (1 + sl_pct)
                    exit_reason, exit_price = None, None
                    if df['high'].iat[idx] >= sl_price:
                        exit_reason, exit_price = 'SL', sl_price
                    elif df['low'].iat[idx] <= tp_price:
                        exit_reason, exit_price = 'TP', tp_price
                    pnl = (entry_price - exit_price) / entry_price * balance if exit_reason else 0
                    balance += pnl
                    trades.append((df.index[idx], "SHORT", entry_price, exit_price, pnl, exit_reason))
                    break

        equity_curve.append({'time': df.index[i], 'balance': balance})

    trades_df = pd.DataFrame(trades, columns=["time","side","entry","exit","pnl","exit_reason"])
    equity_df = pd.DataFrame(equity_curve).set_index("time")
    summary = {
        "initial_balance": cfg.initial_balance,
        "final_balance": balance,
        "net_profit": balance - cfg.initial_balance,
        "total_trades": len(trades_df),
        "winrate": (trades_df['pnl']>0).mean() if len(trades_df)>0 else np.nan,
        "max_drawdown": compute_max_drawdown(equity_df['balance']) if not equity_df.empty else 0
    }
    return trades_df, equity_df, summary

# --- Execution engine ---
def run_backtest(df: pd.DataFrame,
                 signals: pd.Series,
                 config: BacktestConfig,
                 tp_pct: float = 0.001,
                 sl_pct: float = 0.002
                 ) -> Tuple[pd.DataFrame, dict]:
    balance = config.initial_balance
    equity_curve = []
    trades = []
    position = 0
    entry_price = None
    entry_index = None
    position_size_notional = 0.0

    fee_rate = config.fee_taker if config.use_taker else config.fee_maker

    # safety: ensure signals aligned with df index and length
    if not isinstance(signals, pd.Series):
        raise ValueError("signals must be a pandas Series")
    if len(signals) != len(df) or not signals.index.equals(df.index):
        # try to reindex signals to df index (fill 0 where missing)
        signals = signals.reindex(df.index, fill_value=0)

    for i in range(len(df)-1):
        ts = df.index[i]
        next_ts = df.index[i+1]
        price_next_open = df['open'].iat[i+1]

        # entry
        if position == 0:
            sig = int(signals.iloc[i]) if not pd.isna(signals.iloc[i]) else 0
            if sig != 0:
                sl_price = price_next_open * (1 - sl_pct) if sig == 1 else price_next_open * (1 + sl_pct)
                risk_notional = balance * config.risk_per_trade
                sl_distance = abs(price_next_open - sl_price) / price_next_open
                if sl_distance == 0:
                    continue
                size_notional = (risk_notional / sl_distance) * config.leverage
                if size_notional < config.min_size:
                    continue
                position = sig
                entry_price = price_next_open * (1 + config.slippage if sig == 1 else 1 - config.slippage)
                position_size_notional = size_notional
                entry_index = next_ts
                fee_entry = position_size_notional * fee_rate
                balance -= fee_entry
                trades.append({
                    'entry_time': entry_index,
                    'signal': sig,
                    'side': 'LONG' if position==1 else 'SHORT',
                    'entry_price': entry_price,
                    'size_notional': position_size_notional,
                    'fee_entry': fee_entry
                })
        else:
            # exit checks intrabar current i
            high = df['high'].iat[i]
            low = df['low'].iat[i]
            exit_reason = None
            exit_price = None
            if position == 1:
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)
                if low <= sl_price:
                    exit_reason = 'SL'
                    exit_price = sl_price * (1 + config.slippage)
                elif high >= tp_price:
                    exit_reason = 'TP'
                    exit_price = tp_price * (1 - config.slippage)
            else:
                tp_price = entry_price * (1 - tp_pct)
                sl_price = entry_price * (1 + sl_pct)
                if high >= sl_price:
                    exit_reason = 'SL'
                    exit_price = sl_price * (1 - config.slippage)
                elif low <= tp_price:
                    exit_reason = 'TP'
                    exit_price = tp_price * (1 + config.slippage)

            if exit_reason is not None:
                price_return = (exit_price - entry_price) / entry_price
                pnl = price_return * position_size_notional * (1 if position==1 else -1)
                fee_exit = position_size_notional * fee_rate
                balance += pnl
                balance -= fee_exit
                trades[-1].update({
                    'exit_time': df.index[i],
                    'exit_price': exit_price,
                    'fee_exit': fee_exit,
                    'pnl': pnl,
                    'exit_reason': exit_reason,
                    'balance_after': balance
                })
                # reset
                position = 0
                entry_price = None
                position_size_notional = 0.0
                entry_index = None

        equity_curve.append({'time': df.index[i], 'balance': balance})

    equity_df = pd.DataFrame(equity_curve).set_index('time')
    trades_df = pd.DataFrame(trades)

    # safe summary even when no trades
    if trades_df.empty or 'pnl' not in trades_df.columns:
        total_trades = 0
        net_profit = balance - config.initial_balance
        winrate = np.nan
        avg_win = 0
        avg_loss = 0
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
        # debug prints
        print("DEBUG trades_df columns:", trades_df.columns)
        print("No trades executed.")
        return trades_df, equity_df, summary

    # else normal path
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    total_trades = len(trades_df)
    net_profit = balance - config.initial_balance
    winrate = len(wins) / total_trades if total_trades > 0 else np.nan
    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
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
    # debug
    print("DEBUG trades_df columns:", trades_df.columns)
    print(trades_df.head())
    return trades_df, equity_df, summary

def compute_max_drawdown(equity_series: pd.Series):
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    max_dd = drawdown.min()
    return max_dd

# --- Example usage ---
if __name__ == "__main__":
    # 1) load 1m OHLCV CSV
    df = load_ohlcv("/root/trade-control-system/backtest_result_data/TIAUSDT_1m_all_signals.xlsx")

    # 2) generate straddle setup
    straddle = signal_straddle_simple(df, lookback=8, range_threshold=0.0006)

    # 3) run backtest with dual straddle engine
    cfg = BacktestConfig(initial_balance=100.0, fee_taker=0.0004,
                         slippage=0.0006, risk_per_trade=0.01, leverage=3.0)

    trades_df, equity_df, summary = run_backtest_straddle(df, straddle, cfg,
                                                          tp_pct=0.001,
                                                          sl_pct=0.0015)

    print("Summary:", summary)
    trades_df.to_csv("trades_result.csv", index=False)
    equity_df.to_csv("equity_curve.csv")
