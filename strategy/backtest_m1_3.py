import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class BacktestConfig:
    initial_balance: float = 100.0
    risk_per_trade: float = 0.01  # 1% risk per trade
    leverage: float = 3.0
    fee_rate: float = 0.0004
    min_atr: float = 0.0005  # filter market flat
    tp_atr_mult: float = 0.8
    sl_atr_mult: float = 0.8
    max_hold_bars: int = 5  # maksimal hold 5 candle

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

def compute_ema(df, period=20):
    return df['close'].ewm(span=period, adjust=False).mean()

def dual_entry_backtest(df, cfg: BacktestConfig):
    balance = cfg.initial_balance
    trades = []
    equity_curve = []
    atr = compute_atr(df)
    ema = compute_ema(df)  # Filter trend
    
    open_positions = []  # Daftar posisi terbuka: (entry_time, entry_price, side, size, sl, tp)
    
    for i in range(1, len(df)):
        current_time = df.index[i]
        current_open = df['open'].iat[i]
        current_high = df['high'].iat[i]
        current_low = df['low'].iat[i]
        current_close = df['close'].iat[i]
        
        # Periksa exit untuk posisi terbuka
        new_open_positions = []
        for pos in open_positions:
            entry_time, entry_price, side, size, sl, tp = pos
            
            # Check for exit conditions
            exit_reason = None
            exit_price = None
            
            if side == 'LONG':
                # Check for SL
                if current_low <= sl:
                    exit_reason = 'SL'
                    exit_price = sl
                # Check for TP
                elif current_high >= tp:
                    exit_reason = 'TP'
                    exit_price = tp
                # Check for time-based exit
                elif (current_time - entry_time).total_seconds() / 60 >= cfg.max_hold_bars:
                    exit_reason = 'TIME'
                    exit_price = current_close
                    
            elif side == 'SHORT':
                # Check for SL
                if current_high >= sl:
                    exit_reason = 'SL'
                    exit_price = sl
                # Check for TP
                elif current_low <= tp:
                    exit_reason = 'TP'
                    exit_price = tp
                # Check for time-based exit
                elif (current_time - entry_time).total_seconds() / 60 >= cfg.max_hold_bars:
                    exit_reason = 'TIME'
                    exit_price = current_close
            
            # Jika ada alasan exit, tutup posisi
            if exit_reason:
                # Hitung PnL
                if side == 'LONG':
                    pnl = size * (exit_price - entry_price) / entry_price
                else:
                    pnl = size * (entry_price - exit_price) / entry_price
                
                # Kurangi fee
                fee = size * cfg.fee_rate
                pnl -= fee
                
                # Update balance
                balance += pnl
                
                # Catat trade
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'side': side,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'size': size,
                    'pnl': pnl,
                    'exit_reason': exit_reason
                })
            else:
                # Tambahkan posisi yang masih terbuka
                new_open_positions.append(pos)
        
        open_positions = new_open_positions
        
        # Skip jika ATR tidak valid atau market trending
        if np.isnan(atr.iat[i]) or atr.iat[i] < cfg.min_atr:
            equity_curve.append({'time': current_time, 'balance': balance})
            continue
            
        # Filter trend - hindari trading ketika harga jauh dari EMA
        if abs(current_close - ema.iat[i]) > 2 * atr.iat[i]:
            equity_curve.append({'time': current_time, 'balance': balance})
            continue
        
        # Hitung level entry
        long_level = current_close + atr.iat[i] * 0.2
        short_level = current_close - atr.iat[i] * 0.2
        
        # Hitung ukuran posisi berdasarkan risk management
        position_size = balance * cfg.risk_per_trade * cfg.leverage
        
        # Entry rules
        if current_high >= long_level and len(open_positions) == 0:
            # LONG entry
            entry_price = current_open  # Gunakan open candle saat ini
            sl = entry_price - atr.iat[i] * cfg.sl_atr_mult
            tp = entry_price + atr.iat[i] * cfg.tp_atr_mult
            
            open_positions.append((
                current_time, entry_price, 'LONG', position_size, sl, tp
            ))
            
        elif current_low <= short_level and len(open_positions) == 0:
            # SHORT entry
            entry_price = current_open  # Gunakan open candle saat ini
            sl = entry_price + atr.iat[i] * cfg.sl_atr_mult
            tp = entry_price - atr.iat[i] * cfg.tp_atr_mult
            
            open_positions.append((
                current_time, entry_price, 'SHORT', position_size, sl, tp
            ))
        
        equity_curve.append({'time': current_time, 'balance': balance})
    
    # Tutup semua posisi terbuka di akhir
    for pos in open_positions:
        entry_time, entry_price, side, size, sl, tp = pos
        exit_price = df['close'].iat[-1]
        
        if side == 'LONG':
            pnl = size * (exit_price - entry_price) / entry_price
        else:
            pnl = size * (entry_price - exit_price) / entry_price
        
        fee = size * cfg.fee_rate
        pnl -= fee
        balance += pnl
        
        trades.append({
            'entry_time': entry_time,
            'exit_time': df.index[-1],
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'pnl': pnl,
            'exit_reason': 'END'
        })
    
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve).set_index('time')
    
    if len(trades_df) > 0:
        winrate = (trades_df['pnl'] > 0).mean()
    else:
        winrate = 0
        
    summary = {
        'initial_balance': cfg.initial_balance,
        'final_balance': balance,
        'net_profit': balance - cfg.initial_balance,
        'total_trades': len(trades_df),
        'winrate': winrate,
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
    trades_df.to_csv("trades_dual_entry.csv", index=False)
    equity_df.to_csv("equity_dual_entry.csv")