# backtest_bot7.py
import pandas as pd
import numpy as np
import talib

# --- CONFIG ---
INITIAL_BALANCE = 20.0
RISK_PCT = 0.008  # 0.8%
LEVERAGE = 10
TP_ATR_MULT = 3.0
SL_ATR_MULT = 2.5
LEVEL_MULT = 1.0

# --- LOAD DATA ---
df = pd.read_csv("/home/julbot/trade-control-system/data_test/AVAXUSDT_5m-31.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# --- TAMBAH INDIKATOR ---
df['ema_fast'] = talib.EMA(df['close'], 20)
df['ema_slow'] = talib.EMA(df['close'], 50)
df['rsi'] = talib.RSI(df['close'], 14)
df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
df['plus_di'], df['minus_di'], df['adx'] = talib.PLUS_DI(df['high'], df['low'], df['close'], 14), \
                                            talib.MINUS_DI(df['high'], df['low'], df['close'], 14), \
                                            talib.ADX(df['high'], df['low'], df['close'], 14)

# --- FUNGSI REGIME (sama seperti di bot Anda) ---
def enhanced_trend_detection(row_idx, df):
    if row_idx < 30:
        return "MIXED"
    # Ambil data terbaru
    adx = df['adx'].iloc[row_idx]
    plus_di = df['plus_di'].iloc[row_idx]
    minus_di = df['minus_di'].iloc[row_idx]
    ema_fast = df['ema_fast'].iloc[row_idx]
    ema_slow = df['ema_slow'].iloc[row_idx]
    
    trend_score = 0
    if adx > 22:
        trend_score += 2
    elif adx > 16:
        trend_score += 1
    
    if ema_fast > ema_slow:
        trend_score += 2
    elif ema_fast < ema_slow:
        trend_score += 2
    
    di_diff = plus_di - minus_di
    if abs(di_diff) > 12:
        trend_score += 2
    elif abs(di_diff) > 6:
        trend_score += 1

    if trend_score >= 4:
        return "STRONG_TREND"
    elif trend_score <= 2:
        return "SIDEWAYS"
    else:
        return "MIXED"

# --- SIMULASI BACKTEST ---
balance = INITIAL_BALANCE
position = None
trades = []

for i in range(30, len(df)):
    if position is not None:
        # Cek exit
        current_price = df['close'].iloc[i]
        if position['side'] == 'LONG':
            if current_price >= position['tp'] or current_price <= position['sl']:
                pnl = (current_price - position['entry']) * position['qty'] * LEVERAGE
                balance += pnl
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': df['timestamp'].iloc[i],
                    'side': 'LONG',
                    'entry': position['entry'],
                    'exit': current_price,
                    'pnl': pnl,
                    'balance': balance
                })
                position = None
        else:  # SHORT
            if current_price <= position['tp'] or current_price >= position['sl']:
                pnl = (position['entry'] - current_price) * position['qty'] * LEVERAGE
                balance += pnl
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': df['timestamp'].iloc[i],
                    'side': 'SHORT',
                    'entry': position['entry'],
                    'exit': current_price,
                    'pnl': pnl,
                    'balance': balance
                })
                position = None
    
    if position is None:
        # Cek entry
        regime = enhanced_trend_detection(i, df)
        atr = df['atr'].iloc[i]
        close = df['close'].iloc[i]
        
        # Buat level breakout
        long_level = close + atr * LEVEL_MULT
        short_level = close - atr * LEVEL_MULT
        
        # Cek apakah breakout terjadi di candle ini
        broke_long = df['high'].iloc[i] >= long_level
        broke_short = df['low'].iloc[i] <= short_level
        
        if broke_long and regime in ["STRONG_TREND", "MIXED"]:
            # Hitung qty berdasarkan risk
            sl = long_level - atr * SL_ATR_MULT
            risk_per_unit = (long_level - sl) * LEVERAGE
            if risk_per_unit > 0:
                qty = (balance * RISK_PCT) / risk_per_unit
                tp = long_level + atr * TP_ATR_MULT
                position = {
                    'side': 'LONG',
                    'entry': long_level,
                    'tp': tp,
                    'sl': sl,
                    'qty': qty,
                    'entry_time': df['timestamp'].iloc[i]
                }
        elif broke_short and regime in ["STRONG_TREND", "MIXED"]:
            sl = short_level + atr * SL_ATR_MULT
            risk_per_unit = (sl - short_level) * LEVERAGE
            if risk_per_unit > 0:
                qty = (balance * RISK_PCT) / risk_per_unit
                tp = short_level - atr * TP_ATR_MULT
                position = {
                    'side': 'SHORT',
                    'entry': short_level,
                    'tp': tp,
                    'sl': sl,
                    'qty': qty,
                    'entry_time': df['timestamp'].iloc[i]
                }

# --- HASIL ---
trades_df = pd.DataFrame(trades)
if not trades_df.empty:
    win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)
    total_pnl = trades_df['pnl'].sum()
    print(f"Total Trades: {len(trades_df)}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Total PnL: {total_pnl:.4f} USDT")
    print(f"Final Balance: {balance:.4f} USDT")
    trades_df.to_csv("backtest_results.csv", index=False)
else:
    print("Tidak ada trade dieksekusi.")